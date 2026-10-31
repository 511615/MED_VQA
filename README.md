# Medical Visual Question Answering

基于 VQA-RAD 数据集的医学视觉问答实验项目。项目使用 PyTorch 对比传统 CNN-LSTM 基线与预训练视觉语言模型在医学影像问答任务上的表现。

## 项目内容

- 使用 ResNet18 提取医学影像特征，Embedding + LSTM 编码问题文本，构建 CNN-LSTM 基线模型。
- 使用 BERT 和 BLIP 进行视觉语言建模与医学领域微调实验。
- 统一计算 Accuracy、Exact Match（EM）和 Token-level F1，并区分封闭式与开放式问题。
- 使用 Grad-CAM 和错误样本分析模型关注区域、回答偏差及视觉语言模型的幻觉问题。

## 实验结果

在 VQA-RAD 测试集上，BLIP 的 Exact Match 从基线的 33.70% 提升至 37.69%，Token-level F1 从 33.70% 提升至 41.46%。预训练视觉语言模型在开放式医学描述问题上回答更完整，但复杂影像仍可能出现过度解读。

## 目录结构

```text
src/       训练、评测、对比和 Grad-CAM 脚本
docs/      项目说明文档
reports/   项目报告和实验材料
configs/   运行配置预留目录
```

## 运行说明

```bash
pip install -r requirements.txt
python src/baseline_version.py
python src/evaluate_models_vqarad.py
```

部分脚本会从 Hugging Face 加载 `flaviagiammarino/vqa-rad` 数据集，并需要本地下载预训练模型。训练得到的权重、数据集缓存和运行日志不纳入仓库，请根据脚本中的路径配置运行环境。

## 说明

本仓库用于展示模型训练、对比评测和可解释性分析流程，不包含在线推理前端或生产部署服务。
