import os
import json
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ===================== 基本配置 =====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 50  # 每多少个 batch 打印一次训练信息

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
OTHER_TOKEN = "<other>"

MAX_Q_LEN = 20   # 问题最长 20 个词
TOP_K_ANS = 100  # 取出现最多的前 100 个答案作为类别

# ===================== 步骤 1：加载数据 =====================

def load_vqa_rad():
    """
    使用 Hugging Face datasets 加载 VQA-RAD 数据集。
    需要联网下载一次，之后走本地缓存。
    """
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    return dataset["train"], dataset["test"]

# ===================== 步骤 2：构建词典和答案标签 =====================

def build_vocab_and_answers(train_split):
    """
    从训练集统计：
    - 问题中的词 -> 构建 word2idx 词典
    - 答案文本 -> 构建 answer2idx 标签映射
    """
    all_words = []
    all_answers = []

    for ex in train_split:
        q = ex["question"].lower().strip()
        a = ex["answer"].lower().strip()
        all_words.extend(q.split())
        all_answers.append(a)

    word_counter = Counter(all_words)
    answer_counter = Counter(all_answers)

    # 构建词典：保留出现次数 >= 2 的词
    vocab = [PAD_TOKEN, UNK_TOKEN]
    for w, c in word_counter.items():
        if c >= 2:
            vocab.append(w)

    word2idx = {w: i for i, w in enumerate(vocab)}

    # 构建答案标签：取出现最多的 TOP_K_ANS 个，其他归为 OTHER_TOKEN
    most_common_answers = [a for a, c in answer_counter.most_common(TOP_K_ANS)]
    answer2idx = {a: i for i, a in enumerate(most_common_answers)}
    answer2idx[OTHER_TOKEN] = len(answer2idx)

    return word2idx, answer2idx

def encode_question(text, word2idx):
    """
    把一句问题变成固定长度的 ID 序列。
    不足补 PAD，多了截断。
    """
    text = text.lower().strip()
    tokens = text.split()
    ids = []
    for t in tokens[:MAX_Q_LEN]:
        ids.append(word2idx.get(t, word2idx[UNK_TOKEN]))
    while len(ids) < MAX_Q_LEN:
        ids.append(word2idx[PAD_TOKEN])
    return ids

def encode_answer(ans, answer2idx):
    """
    把文本答案映射成类别 ID。
    不在常见答案表中的，统一映射到 OTHER_TOKEN 类别。
    """
    ans = ans.lower().strip()
    return answer2idx.get(ans, answer2idx[OTHER_TOKEN])

# ===================== 步骤 3：定义 Dataset 和 DataLoader =====================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class VQARadDataset(Dataset):
    def __init__(self, hf_split, word2idx, answer2idx, transform=None):
        self.data = hf_split
        self.word2idx = word2idx
        self.answer2idx = answer2idx
        self.transform = transform or image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # 图像
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = image.convert("RGB")
        image = self.transform(image)

        # 问题 → ID 序列
        q_ids = torch.tensor(
            encode_question(row["question"], self.word2idx),
            dtype=torch.long
        )
        # 答案 → 类别 ID
        ans_id = torch.tensor(
            encode_answer(row["answer"], self.answer2idx),
            dtype=torch.long
        )

        return image, q_ids, ans_id

# ===================== 步骤 4：定义 CNN+LSTM 模型 =====================

class VQABaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, num_classes, pad_idx):
        super().__init__()

        # 图像编码器：ResNet18，使用 ImageNet 预训练权重
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()  # 去掉最后的分类层，只保留 512 维特征
        self.cnn = resnet
        cnn_out_dim = 512

        # 问句编码器：Embedding + LSTM
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
        # 图像特征
        img_feat = self.cnn(images)  # [B, 512]

        # 文本特征：取 LSTM 最后一层的 hidden state
        embedded = self.embedding(questions)        # [B, L, E]
        _, (h_n, _) = self.lstm(embedded)           # h_n: [num_layers, B, H]
        q_feat = h_n[-1]                            # [B, H]

        # 融合
        fused = torch.cat([img_feat, q_feat], dim=1)
        logits = self.classifier(fused)             # [B, num_classes]
        return logits

# ===================== 步骤 5：训练 & 评估函数 =====================

def train_one_epoch(model, dataloader, optimizer, criterion, epoch_index):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, questions, answers) in enumerate(dataloader):
        images = images.to(DEVICE)
        questions = questions.to(DEVICE)
        answers = answers.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, questions)
        loss = criterion(logits, answers)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == answers).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % PRINT_EVERY == 0:
            print(f"[Epoch {epoch_index+1} | Step {batch_idx+1}] "
                  f"loss={loss.item():.4f}")

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, questions, answers in dataloader:
            images = images.to(DEVICE)
            questions = questions.to(DEVICE)
            answers = answers.to(DEVICE)

            logits = model(images, questions)
            preds = logits.argmax(dim=1)

            total_correct += (preds == answers).sum().item()
            total_samples += images.size(0)

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return acc

# ===================== 步骤 6：推理（部署接口） =====================

def build_idx2answer(answer2idx):
    return {v: k for k, v in answer2idx.items()}

def answer_question_baseline(model, image_path, question_text,
                             word2idx, answer2idx):
    """
    给定一张图片路径和一个问题字符串，返回模型预测的答案。
    """
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = image_transform(img).unsqueeze(0).to(DEVICE)

    q_ids = torch.tensor(
        [encode_question(question_text, word2idx)],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor, q_ids)
        pred_id = logits.argmax(dim=1).item()

    idx2answer = build_idx2answer(answer2idx)
    return idx2answer[pred_id]

# ===================== 步骤 7：主函数（从头到尾跑一遍） =====================

def main():
    print("使用设备:", DEVICE)

    # 1. 加载原始数据
    print("加载 VQA-RAD 数据集...")
    train_split, test_split = load_vqa_rad()
    print("训练样本数:", len(train_split))
    print("测试样本数:", len(test_split))

    # 2. 构建词典 & 答案标签
    print("构建词典和答案标签...")
    word2idx, answer2idx = build_vocab_and_answers(train_split)
    print("词典大小:", len(word2idx))
    print("答案类别数:", len(answer2idx))

    # 3. 构建 Dataset 和 DataLoader
    train_dataset = VQARadDataset(train_split, word2idx, answer2idx)
    test_dataset = VQARadDataset(test_split, word2idx, answer2idx)

    train_loader = DataLoader(train_dataset, batch_size=16,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=0)

    # 4. 定义模型、损失函数、优化器
    model = VQABaseline(
        vocab_size=len(word2idx),
        embed_dim=256,
        lstm_hidden=256,
        num_classes=len(answer2idx),
        pad_idx=word2idx[PAD_TOKEN]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 5. 训练多个 epoch
    EPOCHS = 10
    best_test_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch
        )
        test_acc = evaluate(model, test_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"test_acc={test_acc:.4f}")

        # 简单选最好的模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "baseline_model_best.pth")
            print(f"保存当前最优模型，test_acc={best_test_acc:.4f}")

    # 6. 保存词典和答案映射，方便之后部署复用
    with open("baseline_word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)
    with open("baseline_answer2idx.json", "w", encoding="utf-8") as f:
        json.dump(answer2idx, f, ensure_ascii=False, indent=2)

    print("训练完成！最优测试准确率:", best_test_acc)
    print("模型保存在 baseline_model_best.pth，词典保存在 baseline_word2idx.json 等文件中。")

if __name__ == "__main__":
    main()
