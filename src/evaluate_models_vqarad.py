import os
import json
from collections import Counter
import re
import string

import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BlipProcessor, BlipForQuestionAnswering
from torchvision import models, transforms
from tqdm import tqdm

# ===================== 一些超参数和路径配置 =====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", DEVICE)

# 这三个路径你需要根据自己电脑上的实际情况调整
BASELINE_MODEL_PATH = "./vqa_baseline_bert_finetune/vqa_baseline_bert_finetune.pt"
BASELINE_ANS2ID_PATH = "./vqa_baseline_bert_finetune/answer2id.json"
BLIP_MODEL_DIR = "./blip/blip_medvqa_longtrain"  # 你保存 blip 模型和 processor 的目录

MAX_Q_LEN = 32
MAX_A_LEN = 8
IMAGE_SIZE = 224


# ===================== 文本归一化 & F1 计算 =====================

def normalize_answer(s: str) -> str:
    """
    把答案统一成“便于比较”的形式：
    - 全小写
    - 去标点
    - 去 a/an/the
    - 合并多余空格
    """
    def lower(text):
        return text.lower()

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_single(pred: str, gold: str) -> float:
    """
    单样本 token-level F1
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + precision) if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# ===================== baseline 模型（ResNet + BERT） =====================

class VQABertBaseline(nn.Module):
    """
    这里结构和你训练 full-finetune BERT baseline 时保持一致：
    - ResNet-50 去掉最后一层 fc，当图像 backbone
    - BERT base 作为文本编码器
    - 融合后接一个全连接分类头
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # 图像 backbone（这里不用预训练权重，反正会从 state_dict 加载）
        resnet = models.resnet50(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)
        self.img_feat_dim = resnet.fc.in_features  # 2048

        # 文本 BERT
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.text_feat_dim = self.bert.config.hidden_size  # 通常是 768

        # 融合 + 分类
        fusion_dim = 512
        self.fusion = nn.Linear(self.img_feat_dim + self.text_feat_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, images, questions):
        # 图像特征
        img_feat = self.cnn_backbone(images)      # (B, 2048, 1, 1)
        img_feat = torch.flatten(img_feat, 1)     # (B, 2048)

        # 文本特征（BERT 的 [CLS] 向量）
        enc = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=MAX_Q_LEN,
            return_tensors="pt"
        ).to(images.device)

        bert_out = self.bert(**enc)
        last_hidden = bert_out.last_hidden_state      # (B, seq_len, hidden)
        cls_emb = last_hidden[:, 0, :]               # (B, hidden)

        # 融合
        fused = torch.cat([img_feat, cls_emb], dim=1)  # (B, 2048+768)
        fused = torch.relu(self.fusion(fused))         # (B, fusion_dim)
        logits = self.classifier(fused)                # (B, num_classes)
        return logits


# ===================== 统一评估函数 =====================

def evaluate_baseline_and_blip():
    print("加载 VQA-RAD 数据集...")
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    test_data = dataset["test"]

    # 1. 加载 answer2id（来自你训练 baseline 时保存的 json）
    print("加载 baseline 的答案映射 answer2id ...")
    with open(BASELINE_ANS2ID_PATH, "r", encoding="utf-8") as f:
        answer2id = json.load(f)
    # json 里是 {answer(str): id(int)}，这里反过来做 id -> answer
    id2answer = {int(v): k for k, v in answer2id.items()}
    num_classes = len(answer2id)
    print("answer2id 大小:", num_classes)

    # 2. 加载 Baseline-BERT 模型
    print("加载 Baseline-BERT 模型权重...")
    baseline_model = VQABertBaseline(num_classes=num_classes).to(DEVICE)
    state_dict = torch.load(BASELINE_MODEL_PATH, map_location=DEVICE)
    baseline_model.load_state_dict(state_dict)
    baseline_model.eval()

    # 3. 加载 BLIP 微调模型
    print("加载 BLIP 微调模型和 processor...")
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_DIR)
    blip_model = BlipForQuestionAnswering.from_pretrained(BLIP_MODEL_DIR).to(DEVICE)
    blip_model.eval()

    # 图像预处理（给 baseline 用）
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ===== 统计量初始化 =====
    total = 0

    # baseline：分类准确率 + 文本 EM/F1
    base_correct_cls = 0
    base_em_sum = base_f1_sum = 0.0
    base_closed_n = base_closed_em = base_closed_f1 = 0.0
    base_open_n = base_open_em = base_open_f1 = 0.0

    # BLIP：文本 EM/F1
    blip_em_sum = blip_f1_sum = 0.0
    blip_closed_n = blip_closed_em = blip_closed_f1 = 0.0
    blip_open_n = blip_open_em = blip_open_f1 = 0.0

    print("开始在测试集上同时评估 Baseline-BERT 和 BLIP ...")

    for idx, ex in enumerate(tqdm(test_data, desc="Eval on test")):
        img_pil = ex["image"]             # PIL Image
        question = ex["question"]
        gt_answer = ex["answer"]

        # ====== 用 GT 答案来推断 CLOSED / OPEN ======
        norm_gt = normalize_answer(gt_answer)
        if norm_gt in ("yes", "no"):
            ans_type_str = "CLOSED"
        else:
            ans_type_str = "OPEN"
        at_lower = ans_type_str.lower()

        # -------- baseline 预测 --------
        with torch.no_grad():
            img_tensor = test_transform(img_pil).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
            logits = baseline_model(img_tensor, [question])
            pred_id = torch.argmax(logits, dim=1).item()
            pred_answer_base = id2answer.get(pred_id, "<unk>")

        # baseline 分类标签的 ground truth（用于 classification accuracy）
        gt_label_id = answer2id.get(gt_answer, answer2id.get("<unk>"))
        if gt_label_id is None:
            # 理论上不会发生，如果发生就简单跳过该样本
            continue

        total += 1
        if pred_id == gt_label_id:
            base_correct_cls += 1

        # baseline 文本 EM/F1
        base_em = 1.0 if normalize_answer(pred_answer_base) == normalize_answer(gt_answer) else 0.0
        base_f1 = f1_score_single(pred_answer_base, gt_answer)
        base_em_sum += base_em
        base_f1_sum += base_f1

        if "close" in at_lower:   # CLOSED 问题
            base_closed_n += 1
            base_closed_em += base_em
            base_closed_f1 += base_f1
        elif "open" in at_lower:  # OPEN 问题
            base_open_n += 1
            base_open_em += base_em
            base_open_f1 += base_f1

        # -------- BLIP 预测 --------
        with torch.no_grad():
            blip_inputs = blip_processor(
                images=img_pil,
                text=question,
                return_tensors="pt"
            ).to(DEVICE)

            out_ids = blip_model.generate(
                **blip_inputs,
                max_length=MAX_A_LEN,
                num_beams=3
            )
            pred_answer_blip = blip_processor.decode(out_ids[0], skip_special_tokens=True)

        blip_em = 1.0 if normalize_answer(pred_answer_blip) == normalize_answer(gt_answer) else 0.0
        blip_f1 = f1_score_single(pred_answer_blip, gt_answer)
        blip_em_sum += blip_em
        blip_f1_sum += blip_f1

        if "close" in at_lower:
            blip_closed_n += 1
            blip_closed_em += blip_em
            blip_closed_f1 += blip_f1
        elif "open" in at_lower:
            blip_open_n += 1
            blip_open_em += blip_em
            blip_open_f1 += blip_f1

        # 每隔 80 条打印一个样例，方便你大致看看
        if (idx + 1) % 80 == 0:
            print(f"\n[样例 {idx+1}]")
            print("Q:", question)
            print("GT:", gt_answer, "| type:", ans_type_str)
            print("Baseline 预测:", pred_answer_base)
            print("BLIP     预测:", pred_answer_blip)

    # ======== 统计结果汇总 ========

    base_acc = base_correct_cls / total if total > 0 else 0.0
    base_em_all = base_em_sum / total if total > 0 else 0.0
    base_f1_all = base_f1_sum / total if total > 0 else 0.0

    blip_em_all = blip_em_sum / total if total > 0 else 0.0
    blip_f1_all = blip_f1_sum / total if total > 0 else 0.0

    print("\n==================== 统一评估结果 ====================")
    print(f"总样本数: {total}")

    print("\n--- Baseline-BERT (full finetune) ---")
    print(f"Classification Accuracy: {base_acc:.4f}")
    print(f"All     - EM: {base_em_all:.4f}, Token F1: {base_f1_all:.4f}")
    if base_closed_n > 0:
        print(f"Closed  - EM: {base_closed_em/base_closed_n:.4f}, Token F1: {base_closed_f1/base_closed_n:.4f}")
    if base_open_n > 0:
        print(f"Open    - EM: {base_open_em/base_open_n:.4f}, Token F1: {base_open_f1/base_open_n:.4f}")

    print("\n--- BLIP (fine-tuned) ---")
    print(f"All     - EM: {blip_em_all:.4f}, Token F1: {blip_f1_all:.4f}")
    if blip_closed_n > 0:
        print(f"Closed  - EM: {blip_closed_em/blip_closed_n:.4f}, Token F1: {blip_closed_f1/blip_closed_n:.4f}")
    if blip_open_n > 0:
        print(f"Open    - EM: {blip_open_em/blip_open_n:.4f}, Token F1: {blip_open_f1/blip_open_n:.4f}")

    print("\n评估完成。你可以把这些数字整理成表格写入报告。")


if __name__ == "__main__":
    evaluate_baseline_and_blip()
