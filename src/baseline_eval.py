import json

import torch
import torch.nn as nn
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image

# ======= 基本设置，要和 baseline_version.py 保持一致 =======

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
OTHER_TOKEN = "<other>"
MAX_Q_LEN = 20   # 跟你 baseline 里的一样

# 图像预处理（和训练时保持一致）
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========= 1. 文本编码（和训练时一样） =========

def encode_question(text, word2idx):
    text = text.lower().strip()
    tokens = text.split()
    ids = []
    for t in tokens[:MAX_Q_LEN]:
        ids.append(word2idx.get(t, word2idx.get(UNK_TOKEN)))
    while len(ids) < MAX_Q_LEN:
        ids.append(word2idx.get(PAD_TOKEN))
    return ids

def encode_answer(ans, answer2idx):
    ans = ans.lower().strip()
    return answer2idx.get(ans, answer2idx[OTHER_TOKEN])

# ========= 2. 定义模型结构（必须和训练时完全一致） =========

class VQABaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, num_classes, pad_idx):
        super().__init__()
        # 图像编码器：ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.cnn = resnet
        cnn_out_dim = 512

        # 问题编码：Embedding + LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)

        # 融合 + 分类
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + lstm_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions):
        img_feat = self.cnn(images)                 # [B, 512]
        embedded = self.embedding(questions)        # [B, L, E]
        _, (h_n, _) = self.lstm(embedded)          # h_n: [num_layers, B, H]
        q_feat = h_n[-1]                            # [B, H]
        fused = torch.cat([img_feat, q_feat], dim=1)
        logits = self.classifier(fused)             # [B, num_classes]
        return logits

# ========= 3. 加载模型 + 词典 =========

def load_baseline_model_and_dicts():
    # 1) 读取词典
    with open("baseline_word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    with open("baseline_answer2idx.json", "r", encoding="utf-8") as f:
        answer2idx = json.load(f)

    # 2) 建模型实例并加载权重
    model = VQABaseline(
        vocab_size=len(word2idx),
        embed_dim=256,          # 要和 baseline_version.py 里的设置一致
        lstm_hidden=256,
        num_classes=len(answer2idx),
        pad_idx=word2idx[PAD_TOKEN],
    )
    state_dict = torch.load("baseline_model_best.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, word2idx, answer2idx

# ========= 4. 在 VQA-RAD 测试集上计算准确率 =========

def eval_baseline_on_vqarad():
    print("使用设备:", DEVICE)

    # 加载数据集
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    test_split = dataset["test"]
    print("测试样本数:", len(test_split))

    # 加载模型和词典
    model, word2idx, answer2idx = load_baseline_model_and_dicts()

    total = 0
    correct = 0

    for i, example in enumerate(test_split):
        image = example["image"]
        question = example["question"]
        gt_answer = example["answer"]

        # 图像 → tensor
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = image.convert("RGB")
        img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

        # 问题 → ID 序列
        q_ids = torch.tensor(
            [encode_question(question, word2idx)],
            dtype=torch.long
        ).to(DEVICE)

        # 真实答案 → 类别 ID
        gt_id = encode_answer(gt_answer, answer2idx)

        with torch.no_grad():
            logits = model(img_tensor, q_ids)
            pred_id = logits.argmax(dim=1).item()

        if pred_id == gt_id:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(test_split)}] 当前准确率: {correct/total:.4f}")

    acc = correct / total if total > 0 else 0.0
    print(f"\nBaseline 模型在整个测试集上的准确率: {acc:.4f}")
    return acc

if __name__ == "__main__":
    eval_baseline_on_vqarad()
