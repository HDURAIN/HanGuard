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

推理时在 `<|im_start|>assistant\n` 处截断，让模型续写后面的两行标签。`DataCollatorForCompletionOnlyLM` 确保 loss 只作用在 assistant 部分（即上面的最后两行）。

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
CUDA_VISIBLE_DEVICES=0 python infer_v5.py \
    --input  data/my_data.csv \
    --model  outputs/hanguard_v5 \
    --output outputs/my_results.csv
```

输入 CSV/Parquet 需含 `prompt` 列，输出新增 `harmful_pred`、`category_pred`、`category_pred_label` 三列。

> `infer_v5.py` 会从 adapter 目录下的 `adapter_config.json` 自动读取基座模型路径，
> 也可手动指定：`--model outputs/hanguard_v5 --base_model /your/path/Qwen2.5-7B-Instruct`（需在脚本中添加该参数）。

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
| Precision（精确率） | TP / (TP+FP) | 判为有害的样本中真正有害的比例，体现**误报控制**能力；Precision 低 → 频繁误伤无害内容 | ↑ 越高越好 |
| Recall（召回率） | TP / (TP+FN) | 真正有害样本中被成功检出的比例，体现**漏报控制**能力；Recall 低 → 有害内容频繁绕过 | ↑ 越高越好 |
| F1 | 2 × P × R / (P+R) | Precision 与 Recall 的调和平均，惩罚两者差距，综合衡量准确与覆盖的权衡 | ↑ 越高越好 |
| FPR（误报率） | FP / (FP+TN) | 无害内容被误判为有害的概率，直接影响正常用户体验 | ↓ 越低越好 |
| FNR（漏报率） | FN / (FN+TP) | 有害内容逃过检测的概率，直接影响安全风险敞口 | ↓ 越低越好 |

> Precision ↔ Recall 存在天然权衡：提高 Recall（减少漏报）往往会降低 Precision（增加误报），F1 是两者的平衡点。

### 六分类指标

| 指标 | 公式 | 物理意义 | 方向 |
|------|------|----------|:----:|
| Macro F1 | (1/K) Σ F1_k | 各类别 F1 的算术平均，**对每个类别一视同仁**，对少数类（如类别 5）敏感，适合评估长尾性能 | ↑ 越高越好 |
| Weighted F1 | Σ (n_k × F1_k) / N | 按各类别样本量加权的 F1 均值，反映**整体加权性能**，受多数类主导 | ↑ 越高越好 |

---

## 实验结果

测试集：`data/final_test.parquet`（12,092 条，中文），4 卡并行推理。

### 二分类基线对比（有害 / 无害）

| 模型 | 架构 | 测试集 | Accuracy | Precision | Recall | F1 | FPR | FNR |
|------|------|--------|:--------:|:---------:|:------:|:--:|:---:|:---:|
| WildGuard | Mistral-7B 微调（英文）| WildGuardTest（英文，参考）| — | — | — | 82.7% | — | — |
| WildGuard | Mistral-7B 微调（英文）| 本项目中文测试集 | TBD | TBD | TBD | TBD | TBD | TBD |
| GPT-4o | 通用大模型（zero-shot）| 本项目中文测试集 | TBD | TBD | TBD | TBD | TBD | TBD |
| HanGuard v4（旧）| Mistral-7B + MacBERT 两阶段 | 本项目中文测试集 | TBD | TBD | TBD | TBD | TBD | TBD |
| **HanGuard v5（本文）** | Qwen2.5-7B QLoRA 单模型 | 本项目中文测试集 | **96.87%** | **98.57%** | **95.55%** | **97.04%** | **1.61%** | **4.45%** |

> WildGuard 的 82.7% F1 来自原论文 WildGuardTest（英文），与本项目中文测试集不可直接比较，仅作量级参考。  
> TBD 项待在统一测试集上运行评估脚本后填入。

混淆矩阵（HanGuard v5）：TP=6,212 / TN=5,501 / FP=90（误报）/ FN=289（漏报）

### 六分类（类别 0–5，HanGuard v5）

| 指标 | 值 |
|------|----|
| Macro F1 | 0.11 |
| Weighted F1 | 0.36 |
| Accuracy | 52.51% |

> **六分类结果无效，存在训练 bug。** 详见下方分析。六分类对比表待修复后重新训练补充。

### 注入鲁棒性

测试集中未包含注入样本（注入样本 100% 进入训练集），无法在本轮评估。

---

## 已知问题与修复计划

### 六分类崩溃根因

`train_v5.py` 的 gold output 模板使用纯中文类别标签：

```
Harm category: {category_label}   # 例：违反社会主义核心价值观的内容
```

导致模型在生成时未能学会"从 6 个类别中选一个"的离散决策，而是将 system prompt 中描述的所有类别标签依次列出。`parse_output` 的正则取第一行"Harm category:"的内容，第一行始终是"安全"（无数字），最终所有预测退化为 category=0，macro F1 = 0.11。

**修复方案（已提交）**：将训练模板改为：

```
Harm category: {category_id} {category_label}
```

使输出包含数字 ID，模型可学习精确类别选择。同步修复 `parse_output` 支持中文标签→ID 映射（兼容旧模型）。

**待完成**：使用修复后的模板重新训练，预期六分类 macro F1 显著提升。

---

## 历史版本

旧版两阶段架构（Mistral-7B 越狱检测 + MacBERT 六分类）和相关代码已归档至 `legacy/`，不再维护。
