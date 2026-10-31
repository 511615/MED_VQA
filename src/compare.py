import json
import re

import torch
import torch.nn as nn
from torchvision import transforms, models
from datasets import load_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# ========== 通用设置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", DEVICE)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
OTHER_TOKEN = "<other>"
MAX_Q_LEN = 20      # 跟 baseline 训练时一致
MAX_A_LEN = 8       # 跟 BLIP 微调时一致
BLIP_MODEL_DIR = "./blip/blip_medvqa_vqarad"   # 你微调后保存的目录

# 图像预处理（baseline 用）
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========== 一些小工具：文本标准化 + EM / F1 计算 ==========

def normalize_text(s: str) -> str:
    """用来做 token-level F1 前的简单归一化。"""
    s = s.lower().strip()
    # 去掉标点（只保留字母、数字、空格）
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # 合并多余空格
    s = re.sub(r"\s+", " ", s)
    return s

def compute_em_and_f1(pred: str, gt: str):
    """
    输入：预测答案字符串、真实答案字符串
    输出：EM（0/1），F1（0~1）
    """
    # ------- EM（严格字符串匹配，小写+去空格） -------
    pred_strict = pred.strip().lower()
    gt_strict = gt.strip().lower()
    em = 1.0 if pred_strict == gt_strict else 0.0

    # ------- token-level F1 -------
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)

    pred_tokens = pred_norm.split() if pred_norm else []
    gt_tokens = gt_norm.split() if gt_norm else []

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return em, 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return em, 0.0

    # 计算 token overlap（多重集）
    common = 0
    gt_counts = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1
    for t in pred_tokens:
        if gt_counts.get(t, 0) > 0:
            common += 1
            gt_counts[t] -= 1

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return em, f1

# ========== baseline：模型定义 + 加载 + 推理 ==========

def encode_question(text, word2idx):
    text = text.lower().strip()
    tokens = text.split()
    ids = []
    for t in tokens[:MAX_Q_LEN]:
        ids.append(word2idx.get(t, word2idx.get(UNK_TOKEN)))
    while len(ids) < MAX_Q_LEN:
        ids.append(word2idx.get(PAD_TOKEN))
    return ids

class VQABaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, num_classes, pad_idx):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.cnn = resnet
        cnn_out_dim = 512

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + lstm_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions):
        img_feat = self.cnn(images)
        embedded = self.embedding(questions)
        _, (h_n, _) = self.lstm(embedded)
        q_feat = h_n[-1]
        fused = torch.cat([img_feat, q_feat], dim=1)
        logits = self.classifier(fused)
        return logits

def load_baseline_model():
    with open("baseline_word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    with open("baseline_answer2idx.json", "r", encoding="utf-8") as f:
        answer2idx = json.load(f)

    idx2answer = {v: k for k, v in answer2idx.items()}

    model = VQABaseline(
        vocab_size=len(word2idx),
        embed_dim=256,
        lstm_hidden=256,
        num_classes=len(answer2idx),
        pad_idx=word2idx[PAD_TOKEN],
    )
    state_dict = torch.load("baseline_model_best.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, word2idx, idx2answer

def predict_baseline_answer_str(image, question, model, word2idx, idx2answer):
    # 图像处理
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

    with torch.no_grad():
        logits = model(img_tensor, q_ids)
        pred_id = logits.argmax(dim=1).item()

    pred_answer = idx2answer[pred_id]  # 这是一个字符串，比如 "yes" / "no" / "<other>" / 具体短语
    return pred_answer

# ========== BLIP：加载微调后模型 + 推理 ==========

def load_blip_finetuned():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_DIR)
    model = BlipForQuestionAnswering.from_pretrained(BLIP_MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return processor, model

def predict_blip_answer_str(image, question, processor, model):
    if not isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_length=MAX_A_LEN,
            num_beams=3
        )
    pred = processor.decode(out_ids[0], skip_special_tokens=True)
    return pred

# ========== 主评估函数：两个模型统一文本指标 ==========

def main():
    print("加载 VQA-RAD 测试集...")
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    test_split = dataset["test"]
    print("测试样本数:", len(test_split))

    print("加载 baseline 模型...")
    baseline_model, word2idx, idx2answer = load_baseline_model()

    print("加载微调后的 BLIP 模型...")
    blip_processor, blip_model = load_blip_finetuned()

    # 统计
    b_em_sum = 0.0
    b_f1_sum = 0.0
    blip_em_sum = 0.0
    blip_f1_sum = 0.0
    total = 0

    for i, example in enumerate(test_split):
        image = example["image"]
        question = example["question"]
        gt_answer = example["answer"]  # 真实答案文本

        # baseline 预测文本
        pred_b = predict_baseline_answer_str(image, question, baseline_model, word2idx, idx2answer)
        # BLIP 预测文本
        pred_blip = predict_blip_answer_str(image, question, blip_processor, blip_model)

        # 统一用 EM + F1 评估
        em_b, f1_b = compute_em_and_f1(pred_b, gt_answer)
        em_blip, f1_blip = compute_em_and_f1(pred_blip, gt_answer)

        b_em_sum += em_b
        b_f1_sum += f1_b
        blip_em_sum += em_blip
        blip_f1_sum += f1_blip
        total += 1

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(test_split)}]")
            print(f"  Baseline 预测: {pred_b} | GT: {gt_answer}")
            print(f"  BLIP     预测: {pred_blip} | GT: {gt_answer}")

    b_em = b_em_sum / total
    b_f1 = b_f1_sum / total
    blip_em = blip_em_sum / total
    blip_f1 = blip_f1_sum / total

    print("\n====== 统一文本指标评估结果（整个测试集） ======")
    print(f"Baseline  - EM: {b_em:.4f}, Token F1: {b_f1:.4f}")
    print(f"BLIP      - EM: {blip_em:.4f}, Token F1: {blip_f1:.4f}")

if __name__ == "__main__":
    main()
