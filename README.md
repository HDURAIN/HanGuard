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
│   ├── final_train.parquet       # 训练集（130,029条）
│   ├── final_val.parquet         # 验证集（12,091条）
│   ├── final_test.parquet        # 测试集（12,092条）
│   ├── class_weights.json        # 六分类训练权重
│   ├── 越狱意图分类定义.docx      # 分类标准定义文档
│   └── sources/                  # 原始来源数据（只读）
│       ├── train_wildguard_zh.parquet         # WildGuard 中文译版
│       ├── train_hanguard_v3_labeled.parquet  # 自研中文数据集（已标注）
│       ├── wildguard_injection_retrans_labeled.parquet  # 对抗样本精译集
│       └── wildguard_en/         # 英文原版 WildGuardMix
├── outputs/
│   └── hanguard_v5/              # 训练输出（训练后生成）
├── scripts/hanguard/
│   ├── train_v5.py               # 训练入口
│   ├── prepare_final_dataset.py  # 数据整合与分割
│   ├── generate_injection_defense.py   # 注入防御数据生成
│   ├── retranslate_injection_samples.py # 对抗样本重译
│   ├── translate_wildguard.py    # WildGuard 中文翻译
│   ├── download_training_data.py # 下载 WildGuardMix
│   ├── annotate_system_prompt.py # 六分类标注提示词
│   ├── annotate_v3_zh.py         # v3 数据集六分类标注
│   ├── annotate_retrans.py       # 重译集六分类标注
│   └── annotate_others.py        # 其他数据集标注
├── infer_v5.py                   # 推理脚本
├── evaluate_v5.py                # 评估脚本
├── requirements.txt              # 推理依赖
├── requirements-train.txt        # 训练额外依赖
└── legacy/                       # 历史版本归档（两阶段架构）
```

---

## 数据来源与处理方法

### 数据来源

| 数据集 | 条数 | 说明 |
|--------|------|------|
| WildGuardMix（中文译版） | 86,759 | `allenai/wildguardmix` 英文数据机器翻译，含二分类标签和六分类标签 |
| 自研中文数据集 v3 | 29,065 | 人工构造中文有害/无害 prompt，Claude API 标注六分类 |
| WildGuard 对抗样本精译集 | 5,037 | WildGuardMix 中对抗性样本（多任务混淆、输出注入等）用 Claude API 重译，保留攻击结构 |
| v3 前缀格式修复集 | 10,637 | 针对 v3 "请以……开始回答" 格式偏置的数据增强（去前缀 + 添加无害前缀样本） |
| 注入防御数据集 | 18,230 | 有害样本 × 20 种注入模板 + 无害难负样本，训练分类器抵御提示注入 |

**最终数据集**（`prepare_final_dataset.py` 输出）：

- 自然数据（WildGuard + v3 + 精译集）按 **80/10/10** 分层切分（按 category_id），保证六类分布一致
- 增强数据（前缀修复 + 注入防御）**100% 进训练集**，不出现在 val/test
- 类别 5 欠采样严重（3,516条），训练集过采样至 8,000 条
- 类别权重：sqrt 阻尼逆频率，最小权重归一化为 1

| 类别 | 训练集 | 验证集 | 测试集 | 权重 |
|------|--------|--------|--------|------|
| 0 安全 | 58,645 | 6,350 | 6,350 | 1.00 |
| 1 违反核心价值观 | 30,671 | 2,581 | 2,582 | 1.38 |
| 2 歧视性 | 8,396 | 842 | 842 | 2.64 |
| 3 商业违法 | 8,460 | 778 | 778 | 2.63 |
| 4 侵犯权益 | 15,857 | 1,269 | 1,269 | 1.92 |
| 5 专业安全 | 8,000 | 271 | 271 | 2.71 |
| **合计** | **130,029** | **12,091** | **12,092** | |

### 标注方法

六分类标签由 Claude API（`claude-sonnet-4-6`）标注，系统提示明确六类定义和互斥规则。标注并发数 15-30，支持断点续传。

---

## 训练方法

**模型**：Qwen2.5-7B-Instruct + 4-bit QLoRA（r=16, alpha=32）

**输入格式**（ChatML）：
```
<|im_start|>system
You are a content safety classifier...
<|im_start|>user
Human user:
{prompt}

---

Answers:
<|im_start|>assistant
Harmful request: harmful
Harm category: 违反社会主义核心价值观的内容
```

**Loss**：`DataCollatorForCompletionOnlyLM` 只对 assistant 回复部分计算 loss，并乘以 per-sample category weight（WeightedTrainer）。

**超参**：3 epochs, lr=2e-4, cosine decay, warmup 5%, batch 4/GPU × grad_accum 8。

**单卡启动**：
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/hanguard/train_v5.py
```

**多卡 DDP 启动（推荐，6 卡约 23 小时）**：
```bash
CUDA_VISIBLE_DEVICES=0,1,3,4,5,7 torchrun \
    --nproc_per_node=6 \
    scripts/hanguard/train_v5.py \
    --output_dir outputs/hanguard_v5
```

---

## 安装

```bash
# 推理依赖
pip install -r requirements.txt

# 训练额外依赖
pip install -r requirements-train.txt
```

**环境要求**：Python 3.10+，CUDA 11.8+，显存 ≥ 24GB（推理 4-bit 量化）

---

## 推理

```bash
CUDA_VISIBLE_DEVICES=0 python infer_v5.py \
    --input  data/my_data.csv \
    --model  outputs/hanguard_v5 \
    --output outputs/my_results.csv
```

输入 CSV/Parquet 需含 `prompt` 列，输出新增 `harmful_pred`、`category_pred`、`category_pred_label` 三列。

---

## 评估

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_v5.py \
    --model  outputs/hanguard_v5 \
    --test   data/final_test.parquet \
    --output outputs/eval_v5/report.txt
```

报告包含：二分类 Accuracy/Precision/Recall/F1，六分类各类别及 macro F1，以及注入鲁棒性指标。

---

## 实验结果

> 训练完成后填入

| 指标 | 值 |
|------|----|
| 二分类 F1 | — |
| 六分类 macro F1 | — |
| 六分类 accuracy | — |

---

## 历史版本

旧版两阶段架构（Mistral-7B 越狱检测 + MacBERT 六分类）和相关代码已归档至 `legacy/`，不再维护。
