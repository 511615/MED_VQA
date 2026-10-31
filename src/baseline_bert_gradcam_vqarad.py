import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =================== 基础配置 ===================

DEVICE = torch.device("cpu")   # 为了稳定，先用 CPU 做 Grad-CAM 即可
BERT_NAME = "bert-base-uncased"  # ⚠️ 如果你训练 baseline 时用的是别的 BERT，这里要改掉

MODEL_PATH = "./vqa_baseline_bert_finetune/vqa_baseline_bert_finetune.pt"
ANSWER2ID_PATH = "./vqa_baseline_bert_finetune/answer2id.json"

GRADCAM_OUT_DIR = "./gradcam_baseline_bert"
os.makedirs(GRADCAM_OUT_DIR, exist_ok=True)

# 图像预处理（尽量跟你训练时一致）
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =================== 1. 复现 Baseline-BERT 模型结构 ===================

class VQABaselineBERT(nn.Module):
    def __init__(self, num_answers, bert_name=BERT_NAME):
        super().__init__()
        # 1) CNN 部分：ResNet18
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.cnn.fc.in_features
        # 最后一层 FC 改成 512 维特征
        self.cnn.fc = nn.Linear(in_features, 512)

        # 2) 文本部分：BERT + 线性投影到 512
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden_size = self.bert.config.hidden_size
        self.text_proj = nn.Linear(hidden_size, 512)

        # 3) 分类头：输入 512 维（融合后的特征）
        self.classifier = nn.Linear(512, num_answers)

    def forward(self, images, input_ids, attention_mask):
        # 图像特征 (B, 512)
        img_feat = self.cnn(images)

        # 文本特征：BERT 的 [CLS] -> 线性层 -> (B, 512)
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = out.last_hidden_state[:, 0, :]  # (B, hidden)
        text_feat = self.text_proj(cls)       # (B, 512)

        # 融合（这里用加法，保持 512 维）
        joint = img_feat + text_feat          # (B, 512)

        # 分类
        logits = self.classifier(joint)       # (B, num_answers)
        return logits



# =================== 2. Grad-CAM 核心类 ===================

class GradCAM:
    """
    通用 Grad-CAM 实现：
      - 注册 forward hook 拿到 feature map
      - 注册 backward hook 拿到梯度
      - 对某一个 logit 反向传播，得到 heatmap
    """
    def __init__(self, model: VQABaselineBERT, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # 注册 hook
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # 保存 forward 输出的 feature map
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output 是关于该层输出的梯度
        self.gradients = grad_output[0].detach()

    def generate(self, images, input_ids, attention_mask, target_class=None):
        """
        输入一张图片 + 问题，输出 heatmap (numpy 2D, 范围 0~1)
        """
        self.model.eval()
        self.model.zero_grad()

        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        logits = self.model(images, input_ids, attention_mask)  # (1, C)
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # 针对这个类别的 logit 反向传播
        score = logits[0, target_class]
        score.backward()

        # 取出保存的 梯度 和 特征图
        gradients = self.gradients[0]   # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # 1) 通道维度上，对梯度做 GAP -> 每个通道一个权重 alpha
        alpha = gradients.mean(dim=(1, 2))  # (C,)

        # 2) 用 alpha 加权求和特征图
        weighted = (alpha.view(-1, 1, 1) * activations).sum(dim=0)  # (H, W)

        # 3) ReLU
        cam = F.relu(weighted)

        # 4) 归一化到 0~1
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # 转为 numpy
        cam_np = cam.cpu().numpy()
        return cam_np, probs[0, target_class].item()


# =================== 3. 一些辅助函数：加载数据 / 预处理问题 ===================

def load_answer_mapping():
    with open(ANSWER2ID_PATH, "r", encoding="utf-8") as f:
        answer2id = json.load(f)
    # 反转成 id -> answer
    id2answer = {int(v): k for k, v in answer2id.items()}
    return answer2id, id2answer


def build_model_and_tokenizer():
    # 先加载 answer2id，知道类别数
    answer2id, id2answer = load_answer_mapping()
    num_answers = len(answer2id)

    print("类别数 num_answers =", num_answers)

    # 初始化模型
    model = VQABaselineBERT(num_answers=num_answers, bert_name=BERT_NAME)
    model.to(DEVICE)

    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("加载权重完成。missing keys:", missing)
    print("unexpected keys:", unexpected)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

    return model, tokenizer, id2answer


def preprocess_sample(example, tokenizer):
    """
    输入：datasets 里的一条样本
    输出：image_tensor, input_ids, attention_mask, gt_answer_str
    """
    image: Image.Image = example["image"]
    question: str = example["question"]
    answer: str = example["answer"]

    # 图像预处理
    img_tensor = IMG_TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)

    # 问题编码
    encoded = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    return img_tensor, input_ids, attention_mask, question, answer


# =================== 4. 画图并保存 ===================

def save_gradcam_figure(orig_pil: Image.Image, cam: np.ndarray, out_path: str,
                        title: str = ""):
    """
    orig_pil: 原始图片（PIL）
    cam: 2D numpy，范围 0~1
    """
    # 调整 cam 到和原图一样大小
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    cam_upsampled = F.interpolate(
        cam_tensor,
        size=orig_pil.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(orig_pil.convert("RGB"), cmap="gray")
    plt.imshow(cam_upsampled, cmap="jet", alpha=0.4)  # 热力图叠加
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Grad-CAM 图已保存到:", out_path)


# =================== 5. 主函数：对几条样本做 Grad-CAM ===================

def run_gradcam_on_examples(num_examples=5):
    print("加载 VQA-RAD 数据集...")
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    test_set = dataset["test"]
    print("Test 样本数:", len(test_set))

    model, tokenizer, id2answer = build_model_and_tokenizer()
    model.eval()

    # 目标层：ResNet 的最后一个卷积块（layer4）
    target_layer = model.cnn.layer4
    gradcam = GradCAM(model, target_layer)

    # 随便选几条 test 样本
    indices = list(range(len(test_set)))
    # 为了方便，可以固定前 num_examples 条
    indices = indices[:num_examples]

    for idx in indices:
        example = test_set[idx]
        img_tensor, input_ids, attention_mask, question, gt_answer = preprocess_sample(example, tokenizer)

        # 生成 Grad-CAM
        cam, prob = gradcam.generate(
            images=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_class=None,  # None 表示用模型预测的类别
        )

        # 重新跑一遍 forward 拿预测类别
        with torch.no_grad():
            logits = model(img_tensor.to(DEVICE),
                           input_ids.to(DEVICE),
                           attention_mask.to(DEVICE))
            pred_id = logits.argmax(dim=1).item()
        pred_answer = id2answer.get(pred_id, "<UNK>")

        print(f"\n样本 idx = {idx}")
        print("Question:", question)
        print("GT Answer:", gt_answer)
        print("Pred Answer:", pred_answer, f"(prob={prob:.3f})")

        out_file = os.path.join(GRADCAM_OUT_DIR, f"gradcam_baseline_idx{idx}.jpg")
        save_gradcam_figure(
            orig_pil=example["image"],
            cam=cam,
            out_path=out_file,
            title=f"Q: {question}\nGT: {gt_answer} | Pred: {pred_answer}",
        )


if __name__ == "__main__":
    run_gradcam_on_examples(num_examples=5)
