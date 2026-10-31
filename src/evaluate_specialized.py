import os
import json
import string
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# ===================== 1. 全局配置 (Configuration) =====================

# 显卡设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# 参数设置
BATCH_SIZE = 4  # 如果显存不够，改小这个数字（比如 4 或 2）
EPOCHS = 15  # 想要更高准确率，可以改成 10 或 15
LEARNING_RATE = 2e-5
MAX_Q_LEN = 32
BERT_MODEL_NAME = "Microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
OUTPUT_DIR = "./vqa_med_final_output"

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 2. 工具函数 (Utils) =====================

def normalize_text(s: str) -> str:
    """标准化文本：小写、去标点"""

    def lower(text): return text.lower()

    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)

    def fix_spaces(text): return " ".join(text.split())

    return fix_spaces(remove_articles(remove_punc(lower(s))))


# ===================== 3. 数据集处理 (Dataset) =====================

class VQADataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform, answer2id=None, is_train=True):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_train = is_train

        # 如果是训练集，我们需要构建 answer2id 字典
        if is_train and answer2id is None:
            all_answers = [normalize_text(item['answer']) for item in self.data]
            # 统计最高频的答案，这里简单起见，取所有出现过的答案
            unique_answers = sorted(list(set(all_answers)))
            self.answer2id = {ans: i for i, ans in enumerate(unique_answers)}
            print(f"Dataset built. Total unique answers (classes): {len(self.answer2id)}")
        else:
            self.answer2id = answer2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 处理图片
        image = item['image'].convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 2. 处理问题
        question = normalize_text(item['question'])
        encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=MAX_Q_LEN,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        # 3. 处理答案 (Label)
        answer_text = normalize_text(item['answer'])
        label = self.answer2id.get(answer_text, -1)  # 如果测试集中出现了训练集没见过的答案，标为 -1

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "answer_text": answer_text  # 用于评估
        }


# ===================== 4. 模型定义 (Model) =====================

class MedicalVQA(nn.Module):
    def __init__(self, num_classes):
        super(MedicalVQA, self).__init__()

        # 图像端：DenseNet121
        print("Initializing DenseNet...")
        self.cnn = models.densenet121(weights="DEFAULT")
        cnn_feature_dim = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Identity()  # 去掉原本的分类头

        # 文本端：PubMedBERT
        print("Initializing PubMedBERT...")
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
        text_feature_dim = self.bert.config.hidden_size

        # 融合层
        self.classifier = nn.Sequential(
            nn.Linear(cnn_feature_dim + text_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # 图像特征
        img_feats = self.cnn(images)
        # 文本特征
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = bert_out.last_hidden_state[:, 0, :]  # 取 [CLS] token

        # 拼接
        combined = torch.cat((img_feats, text_feats), dim=1)

        # 分类
        logits = self.classifier(combined)
        return logits


# ===================== 5. 主流程 (Main) =====================

def main():
    # --- A. 准备数据 ---
    print("\n[Step 1] Loading Data...")
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 图片预处理
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = VQADataset(dataset['train'], tokenizer, transform=train_tfm, is_train=True)
    # 重要：测试集必须使用训练集的 answer2id 字典
    test_ds = VQADataset(dataset['test'], tokenizer, transform=eval_tfm, answer2id=train_ds.answer2id, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- B. 立刻保存字典 (关键修复!) ---
    print(f"\n[Step 2] Saving answer2id mapping to {OUTPUT_DIR}...")
    with open(os.path.join(OUTPUT_DIR, "answer2id.json"), "w") as f:
        json.dump(train_ds.answer2id, f)

    # --- C. 初始化模型 ---
    print("\n[Step 3] Initializing Model...")
    num_classes = len(train_ds.answer2id)
    model = MedicalVQA(num_classes).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- D. 训练循环 ---
    print(f"\n[Step 4] Starting Training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in loop:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images, input_ids, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # 每个 epoch 结束后保存一次模型
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))

    # --- E. 最终评估 ---
    print("\n[Step 5] Final Evaluation on Test Set...")
    model.eval()

    correct = 0
    total = 0
    closed_correct = 0
    closed_total = 0
    open_correct = 0
    open_total = 0

    id2answer = {v: k for k, v in train_ds.answer2id.items()}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits = model(images, input_ids, mask)
            preds = torch.argmax(logits, dim=1)

            # 逐个样本统计
            for i in range(len(preds)):
                pred_idx = preds[i].item()
                label_idx = labels[i].item()

                # 如果这个答案在训练集中没出现过 (label_idx == -1)，那肯定算错
                if label_idx == -1:
                    total += 1
                    # 统计为 open (因为 yes/no 肯定在训练集里)
                    open_total += 1
                    continue

                is_correct = (pred_idx == label_idx)
                if is_correct: correct += 1
                total += 1

                # 判断是 Closed 还是 Open 问题 (根据真实答案文本)
                gt_text = id2answer[label_idx]
                if gt_text in ["yes", "no"]:
                    closed_total += 1
                    if is_correct: closed_correct += 1
                else:
                    open_total += 1
                    if is_correct: open_correct += 1

    # --- F. 输出结果 ---
    print("\n================ FINAL RESULTS ================")
    print(f"Overall Accuracy: {correct / total * 100:.2f}%")
    if closed_total > 0:
        print(f"Closed (Yes/No):  {closed_correct / closed_total * 100:.2f}%")
    if open_total > 0:
        print(f"Open-Ended:       {open_correct / open_total * 100:.2f}%")
    print(f"Model and Dictionary saved to: {OUTPUT_DIR}")
    print("===============================================")


if __name__ == "__main__":
    main()