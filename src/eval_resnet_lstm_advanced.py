import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
import re
import string
from collections import Counter
from tqdm import tqdm

# ================= 配置区域 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_Q_LEN = 20
MODEL_PATH = "baseline_model_best.pth"
WORD2IDX_PATH = "baseline_word2idx.json"
ANS2IDX_PATH = "baseline_answer2idx.json"


# ===========================================

# ================= 1. 移植过来的核心评测函数 (来自你上传的 BERT 脚本) =================

def normalize_answer(s: str) -> str:
    """
    标准化答案：转小写、去标点、去冠词、规范空格
    """

    def lower(text): return text.lower()

    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text): return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_single(pred: str, gold: str) -> float:
    """计算单个样本的 Token-level F1"""
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
    return 2 * precision * recall / (precision + recall)


# ================= 2. 模型定义 (ResNet18 + LSTM) =================

class VQABaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, num_classes, pad_idx):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.cnn = resnet
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 + lstm_hidden, 512),
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
        return self.classifier(fused)


# ================= 3. 辅助函数 =================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def encode_question(text, word2idx):
    text = text.lower().strip()
    tokens = text.split()
    ids = []
    for t in tokens[:MAX_Q_LEN]:
        ids.append(word2idx.get(t, word2idx.get("<unk>")))
    while len(ids) < MAX_Q_LEN:
        ids.append(word2idx.get("<pad>"))
    return ids


# ================= 4. 主评测逻辑 =================

def evaluate_advanced():
    print(f"正在加载资源... 设备: {DEVICE}")

    # 1. 加载字典
    with open(WORD2IDX_PATH, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    with open(ANS2IDX_PATH, "r", encoding="utf-8") as f:
        answer2idx = json.load(f)
    # 创建反向字典用于将预测ID转回文本
    id2answer = {v: k for k, v in answer2idx.items()}

    # 2. 加载模型
    model = VQABaseline(
        vocab_size=len(word2idx),
        embed_dim=256,
        lstm_hidden=256,
        num_classes=len(answer2idx),
        pad_idx=word2idx.get("<pad>", 0)
    )

    # 针对那个 FutureWarning (weights_only=False)，这里加上 weights_only=True 更安全（如果你 pytorch 版本较新）
    # 如果报错说没有 weights_only 参数，就删掉它，保持原来的写法
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        # 如果 PyTorch 版本旧不支持 weights_only 参数
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 3. 加载数据
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    test_data = dataset["test"]

    # 统计变量
    total = 0
    correct_cls = 0  # 严格的 Acc (ID 匹配)
    em_sum = 0.0  # 宽松的 EM (文本 Normalize 后匹配)
    f1_sum = 0.0  # F1 Score

    # 分类统计 (Open vs Closed)
    closed_stats = {'n': 0, 'em': 0.0, 'f1': 0.0, 'acc': 0}
    open_stats = {'n': 0, 'em': 0.0, 'f1': 0.0, 'acc': 0}

    print("开始评测...")

    mismatch_count = 0

    for i, example in enumerate(tqdm(test_data)):
        # --- 数据预处理 ---
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)  # 防止有些格式是 numpy 数组
        image = image.convert("RGB")

        img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

        q_ids = torch.tensor([encode_question(example["question"], word2idx)], dtype=torch.long).to(DEVICE)

        gt_raw_text = example["answer"]

        # === 【修复点在这里】 ===
        # 安全获取 answer_type，防止 key 不存在报错
        ans_type = str(example.get("answer_type", "") or "").lower()

        # --- 模型推理 ---
        with torch.no_grad():
            logits = model(img_tensor, q_ids)
            pred_id = logits.argmax(dim=1).item()

        # --- 获取预测文本 ---
        pred_text = id2answer.get(pred_id, "<unk>")

        # --- 计算指标 1: 严格 Acc ---
        gt_id = answer2idx.get(gt_raw_text.lower().strip(), answer2idx.get("<other>", -1))
        is_acc_correct = (pred_id == gt_id)
        if is_acc_correct:
            correct_cls += 1

        # --- 计算指标 2 & 3: EM 和 F1 ---
        em = 1.0 if normalize_answer(pred_text) == normalize_answer(gt_raw_text) else 0.0
        f1 = f1_score_single(pred_text, gt_raw_text)

        em_sum += em
        f1_sum += f1

        # --- 诊断 ---
        if not is_acc_correct and em == 1.0:
            mismatch_count += 1
            if mismatch_count <= 5:
                print(f"\n[发现差异样本] 真实: '{gt_raw_text}' | 预测: '{pred_text}'")
                print(f"   -> Acc 判定: ❌ (GT ID: {gt_id}, Pred ID: {pred_id})")
                print(f"   -> EM  判定: ✅")

        # --- Open / Closed 统计 ---
        total += 1
        if "close" in ans_type:
            closed_stats['n'] += 1
            closed_stats['em'] += em
            closed_stats['f1'] += f1
            if is_acc_correct: closed_stats['acc'] += 1
        elif "open" in ans_type:
            open_stats['n'] += 1
            open_stats['em'] += em
            open_stats['f1'] += f1
            if is_acc_correct: open_stats['acc'] += 1
        # 如果既不是 open 也不是 closed，就只计入 total，不计入子分类

    # ================= 输出结果 =================
    print("\n" + "=" * 40)
    print("       最终评测报告 (ResNet+LSTM)       ")
    print("=" * 40)

    print(f"Total Samples: {total}")
    print(f"Strict Accuracy (原始): {correct_cls / total:.4f}")
    print(f"Exact Match (新指标):   {em_sum / total:.4f}")
    print(f"Token F1 Score (新指标):{f1_sum / total:.4f}")

    print("-" * 20)
    if closed_stats['n'] > 0:
        print(f"Closed Questions ({closed_stats['n']}):")
        print(f"  Acc: {closed_stats['acc'] / closed_stats['n']:.4f}")
        print(f"  EM:  {closed_stats['em'] / closed_stats['n']:.4f}")
        print(f"  F1:  {closed_stats['f1'] / closed_stats['n']:.4f}")

    print("-" * 20)
    if open_stats['n'] > 0:
        print(f"Open Questions ({open_stats['n']}):")
        print(f"  Acc: {open_stats['acc'] / open_stats['n']:.4f}")
        print(f"  EM:  {open_stats['em'] / open_stats['n']:.4f}")
        print(f"  F1:  {open_stats['f1'] / open_stats['n']:.4f}")

    print("\n[分析结果]")
    print(f"共有 {mismatch_count} 个样本被 Acc 判错，但被 EM 判对。")


if __name__ == "__main__":
    evaluate_advanced()