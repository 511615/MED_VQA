import os
import json
from collections import Counter
import re
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from tqdm import tqdm

# ===================== 基本配置 =====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", DEVICE)

BATCH_SIZE = 4          # 显存不够就改成 2
EPOCHS = 10              # 部分微调可以稍微多一点 epoch
LR_BERT = 5e-6          # 只训后 4 层，可以用 1e-5
LR_HEAD = 1e-4          # 融合层/分类头的学习率
MAX_ANSWERS = 300       # 训练集中最常见的 300 个答案，其余归为 <unk>
MAX_Q_LEN = 32          # 问题最大 token 长度
OUTPUT_DIR = "./vqa_baseline_bert_partial_finetune"  # 模型保存目录


# ===================== 文本归一化 & F1 计算 =====================

def normalize_answer(s: str) -> str:
    """
    把答案字符串统一成“便于比较”的形式：
    - 全部转小写
    - 去掉标点
    - 去掉 a/an/the（英文冠词）
    - 规范空格
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
    计算单个样本的 token-level F1
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # 有一方为空
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ===================== 数据加载 & 答案词表 =====================

def load_vqa_rad():
    """
    从 HuggingFace 加载 VQA-RAD 数据集
    """
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    return dataset


def build_answer_vocab(train_split, max_answers=MAX_ANSWERS):
    """
    基于训练集的答案构建一个“答案词表”（分类标签），只保留最常见的 max_answers 个答案，
    其他答案统一本成 <unk>。
    """
    all_answers = [ex["answer"] for ex in train_split]
    counter = Counter(all_answers)

    # 按出现频次从高到低排序
    sorted_answers = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_answers is not None:
        sorted_answers = sorted_answers[:max_answers]

    # 保留这些高频答案，再加一个 <unk>
    answer_list = [a for a, _ in sorted_answers]
    answer_list = ["<unk>"] + answer_list

    answer2id = {a: i for i, a in enumerate(answer_list)}
    id2answer = {i: a for a, i in answer2id.items()}

    print(f"答案词表大小（含 <unk>）: {len(answer2id)}")
    print("Top 10 最频繁的答案:", sorted_answers[:10])
    return answer2id, id2answer, counter


# ===================== Dataset 封装 =====================

IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class VQARADDataset(Dataset):
    def __init__(self, hf_split, answer2id, transform=None):
        """
        hf_split: HuggingFace 的一个 split（train 或 test）
        answer2id: 答案 -> 标签 id 的映射
        transform: 图像预处理
        """
        self.data = hf_split
        self.answer2id = answer2id
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        image = ex["image"]  # PIL Image
        if self.transform is not None:
            image = self.transform(image)

        question = ex["question"]
        answer = ex["answer"]

        # 答案 -> 标签 id，不在词表里的统一成 <unk>
        if answer in self.answer2id:
            label = self.answer2id[answer]
        else:
            label = self.answer2id["<unk>"]

        # answer_type 字段，有些样本可能为 None，这里统一转成字符串
        raw_ans_type = ex["answer_type"] if "answer_type" in ex else ""
        if raw_ans_type is None:
            raw_ans_type = ""
        ans_type = str(raw_ans_type)

        sample = {
            "image": image,          # Tensor
            "question": question,    # str
            "answer": answer,        # str
            "label": label,          # int
            "answer_type": ans_type  # 始终是字符串（"CLOSED"/"OPEN"/""）
        }
        return sample


# ===================== 模型：ResNet + BERT（部分微调版） =====================

class VQABertPartialBaseline(nn.Module):
    def __init__(self, num_classes, freeze_cnn=True, unfreeze_last_n_layers=4):
        super().__init__()

        # 1️⃣ 图像编码器：ResNet-50（预训练）
        resnet = models.resnet50(pretrained=True)
        # 去掉最后的 fc，只保留到 avgpool
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # 输出 (B, 2048, 1, 1)
        self.img_feat_dim = resnet.fc.in_features  # 2048

        if freeze_cnn:
            print(">>> 冻结 CNN backbone 参数")
            for p in self.cnn_backbone.parameters():
                p.requires_grad = False

        # 2️⃣ 文本编码器：BERT
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.text_feat_dim = self.bert.config.hidden_size  # 通常是 768

        # 先全部冻结
        for p in self.bert.parameters():
            p.requires_grad = False

        # 只解冻最后 unfreeze_last_n_layers 层 encoder + pooler
        total_layers = len(self.bert.encoder.layer)
        n = min(unfreeze_last_n_layers, total_layers)
        if n > 0:
            print(f">>> 只解冻 BERT 的最后 {n} 层 encoder 及 pooler")
            # 解冻最后 n 层
            for layer in self.bert.encoder.layer[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # 解冻 pooler
            if hasattr(self.bert, "pooler"):
                for p in self.bert.pooler.parameters():
                    p.requires_grad = True
        else:
            print(">>> BERT 全部冻结（unfreeze_last_n_layers <= 0）")

        # 3️⃣ 融合 + 分类
        fusion_dim = 512
        self.fusion = nn.Linear(self.img_feat_dim + self.text_feat_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, images, questions):
        """
        images: Tensor (B, 3, H, W)
        questions: 长度为 B 的 list[str]
        """

        # 图像特征
        img_feat = self.cnn_backbone(images)         # (B, 2048, 1, 1)
        img_feat = torch.flatten(img_feat, 1)        # (B, 2048)

        # 文本编码（BERT）
        enc = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=MAX_Q_LEN,
            return_tensors="pt"
        ).to(images.device)  # 把 input_ids / attention_mask 都放到同一设备

        bert_out = self.bert(**enc)
        last_hidden = bert_out.last_hidden_state     # (B, seq_len, hidden_size)
        cls_emb = last_hidden[:, 0, :]              # (B, hidden_size) 对应 [CLS]

        # 融合
        fused = torch.cat([img_feat, cls_emb], dim=1)  # (B, 2048 + 768)
        fused = torch.relu(self.fusion(fused))         # (B, fusion_dim)
        logits = self.classifier(fused)                # (B, num_classes)
        return logits


# ===================== 评估函数 =====================

def evaluate(model, data_loader, id2answer):
    model.eval()
    total = 0
    correct_cls = 0
    em_sum = 0.0
    f1_sum = 0.0

    closed_n = closed_em = closed_f1 = 0.0
    open_n = open_em = open_f1 = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Eval"):
            images = batch["image"].to(DEVICE)
            questions = batch["question"]         # list[str]
            labels = batch["label"].to(DEVICE)    # (B,)
            answers = batch["answer"]             # list[str]
            ans_types = batch["answer_type"]      # list[str]

            logits = model(images, questions)
            preds = torch.argmax(logits, dim=1)   # (B,)

            total += labels.size(0)
            correct_cls += (preds == labels).sum().item()

            for i in range(len(answers)):
                pred_id = preds[i].item()
                pred_str = id2answer[pred_id]
                gt_str = answers[i]

                em = 1.0 if normalize_answer(pred_str) == normalize_answer(gt_str) else 0.0
                f1 = f1_score_single(pred_str, gt_str)

                em_sum += em
                f1_sum += f1

                ans_type = ans_types[i]
                if ans_type:  # 非空字符串
                    at_lower = ans_type.lower()
                    if "close" in at_lower:   # "CLOSED"
                        closed_n += 1
                        closed_em += em
                        closed_f1 += f1
                    elif "open" in at_lower: # "OPEN"
                        open_n += 1
                        open_em += em
                        open_f1 += f1

    acc = correct_cls / total if total > 0 else 0.0
    em_all = em_sum / total if total > 0 else 0.0
    f1_all = f1_sum / total if total > 0 else 0.0

    print("\n====== Baseline-BERT (partial finetune) 评估结果 ======")
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"All     - EM: {em_all:.4f}, Token F1: {f1_all:.4f}")

    if closed_n > 0:
        print(f"Closed  - EM: {closed_em/closed_n:.4f}, Token F1: {closed_f1/closed_n:.4f}")
    if open_n > 0:
        print(f"Open    - EM: {open_em/open_n:.4f}, Token F1: {open_f1/open_n:.4f}")

    return {
        "accuracy": acc,
        "all": (em_all, f1_all),
        "closed": (closed_em / closed_n if closed_n > 0 else None,
                   closed_f1 / closed_n if closed_n > 0 else None),
        "open": (open_em / open_n if open_n > 0 else None,
                 open_f1 / open_n if open_n > 0 else None),
    }


# ===================== 训练主流程 =====================

def train_baseline_bert_partial_finetune():
    print("加载 VQA-RAD 数据集...")
    raw_dataset = load_vqa_rad()
    train_raw = raw_dataset["train"]
    test_raw = raw_dataset["test"]

    print("构建答案词表...")
    answer2id, id2answer, counter = build_answer_vocab(train_raw, max_answers=MAX_ANSWERS)
    num_classes = len(answer2id)

    print("构建 PyTorch Dataset 和 DataLoader...")
    train_ds = VQARADDataset(train_raw, answer2id, transform=train_transform)
    test_ds = VQARADDataset(test_raw, answer2id, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("初始化模型（冻结 CNN，只解冻 BERT 最后若干层）...")
    model = VQABertPartialBaseline(
        num_classes=num_classes,
        freeze_cnn=True,
        unfreeze_last_n_layers=4   # 解冻最后 4 层
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # 区分 BERT 参数和其他参数，设置不同学习率
    bert_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            bert_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": LR_BERT},
            {"params": head_params, "lr": LR_HEAD},
        ]
    )

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
            images = batch["image"].to(DEVICE)
            questions = batch["question"]
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images, questions)
            loss = criterion(logits, labels)
            loss.backward()

            # 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} 平均训练损失: {avg_loss:.4f}")

        print("在测试集上评估（Accuracy + EM/F1 + closed/open）...")
        evaluate(model, test_loader, id2answer)

    # 保存模型和答案映射
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "vqa_baseline_bert_partial_finetune.pt")
    torch.save(model.state_dict(), model_path)
    print("模型权重已保存到:", model_path)

    mapping_path = os.path.join(OUTPUT_DIR, "answer2id.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(answer2id, f, ensure_ascii=False, indent=2)
    print("答案映射 answer2id 已保存到:", mapping_path)


if __name__ == "__main__":
    train_baseline_bert_partial_finetune()
