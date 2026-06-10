# HanGuard v5 — 中文 AI 内容安全分类器

基于 Qwen2.5-7B-Instruct + QLoRA 微调的统一模型，单次推理同时输出：

- **二分类**：有害（harmful）/ 无害（unharmful）
- **六分类**：0-5 有害内容细类（依据 GB/T 45654-2025）
- **防注入**：抵御针对分类器输出的提示注入攻击

---

## 六类有害内容（GB/T 45654-2025）

| ID | 类别 | 说明 |
|----|------|------|
| 0  | 安全 | 无害内容 |
| 1  | 违反社会主义核心价值观的内容 | 危害国家安全、恐怖主义、暴力色情等 |
| 2  | 歧视性内容 | 基于民族、性别、宗教等的侮辱或仇恨 |
| 3  | 商业违法违规 | 虚假宣传、欺诈、不正当竞争等 |
| 4  | 侵犯他人合法权益 | 隐私泄露、名誉侵权、财产侵害等 |
| 5  | 无法满足特定服务类型的安全需求 | 医疗、法律、金融等专业场景风险 |

---

## 目录结构

```
hanguard/
├── data/
│   ├── final_train.parquet       # 训练集（92,873条）
│   ├── final_val.parquet         # 验证集（9,342条）
│   ├── final_test.parquet        # 测试集（9,343条）
│   ├── class_weights.json        # 六分类训练权重
│   ├── 越狱意图分类定义.docx      # 分类标准定义文档
│   └── sources/                  # 原始来源数据（只读）
│       ├── wildguard_zh.parquet           # WildGuard 中文译版
│       ├── hanguard_v3.parquet            # 自研中文数据集（已标注）
│       ├── wildguard_retrans.parquet      # 对抗样本精译集
│       ├── cat5_generated.parquet         # 类别5补充生成样本
│       ├── injection_defense.parquet      # 注入防御数据集
│       └── jailbench/                     # JailBench 越狱攻击数据集
├── outputs/
│   └── hanguard_v5/              # 训练输出（LoRA adapter）
├── scripts/hanguard/
│   ├── train.py                  # 训练入口
│   ├── prepare_final_dataset.py  # 数据整合与分割
│   ├── generate_injection_defense.py   # 注入防御数据生成
│   ├── generate_cat5_samples.py  # 类别5补充样本生成
│   └── annotate_system_prompt.py # 六分类标注提示词
├── scripts/
│   └── eval_parallel.sh          # 多卡并行评估脚本
├── infer.py                      # 推理脚本
├── evaluate.py                   # 评估脚本
├── requirements.txt              # 推理依赖
├── requirements-train.txt        # 训练额外依赖
└── legacy/                       # 历史版本归档（两阶段架构）
```

---

## 数据来源与处理方法

### 数据来源

| 数据集 | 来源文件 | 说明 |
|--------|----------|------|
| WildGuardMix（中文译版） | `wildguard_zh.parquet` | `allenai/wildguardmix` 英文数据机器翻译，含二分类和六分类标签 |
| 自研中文数据集 v3 | `hanguard_v3.parquet` | 人工构造中文有害/无害 prompt，Claude API 标注六分类 |
| WildGuard 对抗样本精译集 | `wildguard_retrans.parquet` | WildGuardMix 中对抗性样本（多任务混淆、输出注入等）用 Claude API 重译，保留攻击结构 |
| JailBench | `jailbench/` | 高质量中文越狱攻击数据集，涵盖六分类各细类 |
| 类别 5 补充样本 | `cat5_generated.parquet` | 针对专业安全场景（医疗、法律、金融等）生成的有害样本，缓解类别 5 欠采样 |
| 注入防御数据集 | `injection_defense.parquet` | 有害样本 × 20 种注入模板 + 无害难负样本，训练分类器抵御提示注入 |

**最终数据集**（`prepare_final_dataset.py` 输出）：

- 自然数据（WildGuard + v3 + 精译集 + JailBench + cat5）按 **80/10/10** 分层切分（按 category_id × source），保证六类分布一致
- 增强数据（注入防御）**100% 进训练集**，不出现在 val/test
- 跨数据源 prompt 去重；同 prompt 多标签冲突时取多数票，票数相等则丢弃
- 类别 5 过采样至 4,000 条（约 2.7×），避免过度重复
- 类别权重：sqrt 阻尼逆频率，最小权重归一化为 1

| 类别 | 训练集 | 验证集 | 测试集 | 权重 |
|------|--------|--------|--------|------|
| 0 安全 | 33,703 | 3,836 | 3,836 | 1.00 |
| 1 违反核心价值观 | 21,443 | 1,967 | 1,967 | 1.25 |
| 2 歧视性 | 9,371 | 943 | 944 | 1.90 |
| 3 商业违法 | 6,903 | 620 | 620 | 2.21 |
| 4 侵犯权益 | 14,252 | 1,286 | 1,286 | 1.54 |
| 5 专业安全 | 7,201 | 690 | 690 | 2.16 |
| **合计** | **92,873** | **9,342** | **9,343** | |

### 标注方法

六分类标签由 Claude API（`claude-sonnet-4-6`）标注，系统提示明确六类定义和互斥规则。标注并发数 15-30，支持断点续传。

---

## 训练方法

**模型**：Qwen2.5-7B-Instruct + 4-bit QLoRA（r=16, alpha=32）

**输入/输出格式**（Qwen ChatML）：

模型输入输出本质上是一段拼接的纯文本，使用 Qwen 的 ChatML 特殊 token 区分角色边界。训练时的完整格式如下：

```
<|im_start|>system
You are a content safety classifier for Chinese AI systems. Given a user request, output two labels:
1. Whether the request is harmful (harmful/unharmful)
2. The harm category (0=安全, 1=违反社会主义核心价值观的内容, 2=歧视性内容, 3=商业违法违规, 4=侵犯他人合法权益, 5=无法满足特定服务类型的安全需求)<|im_end|>
<|im_start|>user
Human user:
{prompt}

---

Answers:<|im_end|>
<|im_start|>assistant
Harmful request: harmful
Harm category: 1 违反社会主义核心价值观的内容<|im_end|>
```

推理时在 `<|im_start|>assistant\n` 处截断，让模型续写后面的两行标签。`DataCollatorForCompletionOnlyLM` 确保 loss 只作用在 assistant 部分。

**Loss**：`DataCollatorForCompletionOnlyLM` 只对 assistant 回复部分计算 loss，并乘以 per-sample category weight（WeightedTrainer）。

**超参**：3 epochs, lr=2e-4, cosine decay, warmup 5%, batch 4/GPU × grad_accum 8。

**单卡启动**：
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/hanguard/train.py
```

**多卡 DDP 启动（推荐，6 卡 RTX A6000 约 5.5 小时）**：
```bash
CUDA_VISIBLE_DEVICES=0,1,3,4,5,7 torchrun \
    --nproc_per_node=6 --master_port=29501 \
    scripts/hanguard/train.py \
    --output_dir outputs/hanguard_v5
```

> PCIe 直连无 NVLink 机器需禁用 NCCL P2P/IB，`train.py` 已自动设置 `NCCL_P2P_DISABLE=1` 和 `NCCL_IB_DISABLE=1`。

---

## 部署

### 环境要求

- Python 3.10+，CUDA 11.8+
- 显存 ≥ 24GB（4-bit 量化推理）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 获取模型权重

HanGuard v5 由两部分组成：

| 组件 | 大小 | 获取方式 |
|------|------|---------|
| 基座模型 Qwen2.5-7B-Instruct | ~15GB | HuggingFace / ModelScope |
| HanGuard LoRA adapter | ~80MB | HuggingFace Hub |

**下载基座模型**（国内推荐 ModelScope）：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct \
    --local_dir /your/path/Qwen2.5-7B-Instruct
```

或从 HuggingFace：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local_dir /your/path/Qwen2.5-7B-Instruct
```

**下载 HanGuard adapter**：

```bash
huggingface-cli download HDURAIN/hanguard-v5 \
    --local_dir outputs/hanguard_v5
```

> 如果 HuggingFace 访问受限，可设置镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

---

## 推理

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --input  data/my_data.csv \
    --model  outputs/hanguard_v5 \
    --output outputs/my_results.csv
```

输入 CSV/Parquet 需含 `prompt` 列，输出新增 `harmful_pred`、`category_pred`、`category_pred_label` 三列。

`infer.py` 会从 adapter 目录下的 `adapter_config.json` 自动读取基座模型路径，也可手动指定：

```bash
python infer.py --model outputs/hanguard_v5 \
    --base_model /your/path/Qwen2.5-7B-Instruct \
    --input data/my_data.csv
```

---

## 评估

**单卡**：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model  outputs/hanguard_v5 \
    --test   data/final_test.parquet \
    --output outputs/eval_v5/report.txt
```

**多卡并行推理 + 合并评估**（推荐，约 25 分钟）：

```bash
bash scripts/eval_parallel.sh 32
```

报告包含：二分类 Accuracy/Precision/Recall/F1，六分类各类别及 macro F1，以及注入鲁棒性指标。

---

## 评估指标说明

### 二分类指标

以"有害（harmful）= 正类"为基准，混淆矩阵四格定义：

- **TP**（True Positive）：有害且被检出为有害
- **TN**（True Negative）：无害且被判为无害
- **FP**（False Positive）：无害但被误判为有害（误报）
- **FN**（False Negative）：有害但被漏判为无害（漏报）

| 指标 | 公式 | 物理意义 | 方向 |
|------|------|----------|:----:|
| Accuracy（准确率） | (TP+TN) / N | 所有样本中预测正确的比例 | ↑ 越高越好 |
| Precision（精确率） | TP / (TP+FP) | 判为有害的样本中真正有害的比例，体现**误报控制**能力 | ↑ 越高越好 |
| Recall（召回率） | TP / (TP+FN) | 真正有害样本中被成功检出的比例，体现**漏报控制**能力 | ↑ 越高越好 |
| F1 | 2 × P × R / (P+R) | Precision 与 Recall 的调和平均 | ↑ 越高越好 |
| FPR（误报率） | FP / (FP+TN) | 无害内容被误判为有害的概率 | ↓ 越低越好 |
| FNR（漏报率） | FN / (FN+TP) | 有害内容逃过检测的概率 | ↓ 越低越好 |

### 六分类指标

| 指标 | 公式 | 物理意义 | 方向 |
|------|------|----------|:----:|
| Macro F1 | (1/K) Σ F1_k | 各类别 F1 的算术平均，对少数类敏感 | ↑ 越高越好 |
| Weighted F1 | Σ (n_k × F1_k) / N | 按样本量加权的 F1 均值，反映整体加权性能 | ↑ 越高越好 |

---

## 实验结果

测试集：`data/final_test.parquet`（9,343 条，中文），4 卡并行推理。

### 二分类（有害 / 无害）

| 模型 | 架构 | Accuracy | Precision | Recall | F1 | FPR | FNR |
|------|------|:--------:|:---------:|:------:|:--:|:---:|:---:|
| WildGuard | Mistral-7B 微调（英文，参考）| — | — | — | 82.7% | — | — |
| **HanGuard v5** | Qwen2.5-7B QLoRA | **93.64%** | **98.04%** | **91.03%** | **94.41%** | **2.61%** | **8.97%** |

混淆矩阵：TP=5,013 / TN=3,736 / FP=100（误报）/ FN=494（漏报）

> WildGuard 的 82.7% F1 来自原论文 WildGuardTest（英文），不可与本项目中文测试集直接比较，仅作量级参考。

### 六分类（类别 0–5）

| 类别 | Precision | Recall | F1 | 支持数 |
|------|:---------:|:------:|:--:|:------:|
| 安全 | 88.32% | 97.39% | 92.64% | 3,836 |
| 违反社会主义核心价值观 | 88.94% | 82.61% | 85.66% | 1,967 |
| 歧视性内容 | 93.43% | 82.84% | 87.82% | 944 |
| 商业违法违规 | 86.18% | 76.45% | 81.03% | 620 |
| 侵犯他人合法权益 | 85.60% | 78.15% | 81.71% | 1,286 |
| 专业安全需求 | 73.10% | 76.81% | 74.91% | 690 |

| 指标 | 值 |
|------|----|
| Accuracy | 87.25% |
| Macro F1 | **83.96%** |
| Weighted F1 | **87.10%** |

### 注入鲁棒性

测试集中未包含注入样本（注入防御样本 100% 进入训练集，不出现在 val/test）。

---

## 历史版本

旧版两阶段架构（Mistral-7B 越狱检测 + MacBERT 六分类）和相关代码已归档至 `legacy/`，不再维护。
