# HanGuard 实验结果归档 20261015

本目录集中保存截至 2026 年 10 月 15 日已有的实验指标、分析报告、实验协议和数据验收结果。已完成结果包括历史 LoRA 二分类、类别查询聚合、同一冻结 LoRA 的多层读出，以及训练内部的 A/B 诊断。当前新版数据上的全量无 LoRA 实验尚未产生最终准确率，不能用这些历史成绩代替。

归档仅包含报告、聚合指标、协议和状态快照，不包含完整模型、特征缓存、原始语料或逐条预测。`artifacts/` 保留来源文件在仓库中的相对路径；原报告中指向未归档权重、缓存或逐条文件的链接可能不可用。

- [实验目录与状态](index.json)
- [文件来源与哈希](manifest.json)
- [聚合指标表](metrics_summary.csv)
- [方法差值与比较口径](comparisons.csv)

文件校验可在仓库根目录运行 `python3 experiment_results/20261015/verify_archive.py`，核对归档哈希、指标来源、差值和本说明的文件链接。

以下准确率、F1 和差值均以百分比或百分点表示；“均值 ± SD”是训练种子之间的均值与样本标准差。不同数据版本、任务或评测范围的分数不能直接相减作为方法收益。

## 历史 LoRA 二分类

历史翻译修复版数据共 77,686 条，测试集 7,778 条。六种配置均已完成，以下是 seed 42、验证集选择阈值后的结果。

| 配置 | 测试准确率 | 有害 F1 | 有害召回率 | 无害误报率 |
|---|---:|---:|---:|---:|
| E03 LoRA 加末层 MLP | 96.824 | 96.950 | 96.296 | 2.594 |
| E04 LoRA 加多层融合 | 97.043 | 97.164 | 96.640 | 2.513 |
| E13 LoRA 加固定位置监督 | 96.940 | 97.075 | 96.860 | 2.972 |
| E14 LoRA 加动态伪标签 | 96.927 | 97.053 | 96.517 | 2.621 |
| E15 LoRA 加融合及固定位置监督 | 97.043 | 97.163 | 96.591 | 2.459 |
| E16 LoRA 加融合及动态伪标签 | 96.940 | 97.072 | 96.762 | 2.864 |

E04 相对 E03 为约 +0.219 个百分点。它们各自训练 LoRA，不能据此单独识别冻结表示中的跨层信息收益；单种子也不能证明跨训练种子稳定。既有测试身份已经参与历史探索。

入口：[实验报告](artifacts/outputs/hanguard_repaired_core_20260928/report.md)、[完整聚合结果](artifacts/outputs/hanguard_repaired_core_20260928/summary.json)、[协议](artifacts/outputs/hanguard_repaired_core_20260928/protocol.json)。

## 两来源类别查询聚合

这是有害文本的五分类读出实验，使用中文整理语料与 JailBench，不含 WildGuard。三种类别头共用同一冻结骨干、LoRA 和二分类门控；五分类测试为 1,828 条有害文本，六类端到端测试为 3,452 条文本。

| 类别头 | 头参数量 | 五分类准确率 | 五分类 Macro F1 | 六类端到端准确率 |
|---|---:|---:|---:|---:|
| 末 token MLP | 705,655 | 79.45 ± 0.81 | 79.02 ± 0.96 | 88.79 ± 0.43 |
| 可学习类别 query | 706,177 | 82.57 ± 2.18 | 81.63 ± 1.95 | 90.45 ± 1.10 |
| 类别描述 query | 706,177 | 81.95 ± 2.37 | 81.02 ± 1.85 | 90.12 ± 1.23 |

以上使用 seeds 42、43、44。可学习 query 相对 MLP 的五分类准确率平均提高 3.12 个百分点，三个种子均为正；描述 query 相对可学习 query 平均低 0.62 个百分点。这里的收益来自全文 token 聚合与类别头变化，不能称为多层融合收益，也没有改变共享的有害识别模型。

入口：[实验报告](artifacts/outputs/hanguard_two_source_primary_20260928/report.md)、[结果解读](artifacts/outputs/hanguard_two_source_primary_20260928/interpretation.md)、[完整聚合结果](artifacts/outputs/hanguard_two_source_primary_20260928/report.json)、[协议](artifacts/outputs/hanguard_two_source_primary_20260928/protocol.json)。

## 同一冻结 LoRA 的静态融合诊断

以下始终使用同一个历史 E03 seed 42 LoRA，只改变读出头。`repeat_last` 将末层重复输入，与两种融合头匹配参数量、拓扑和同 seed 初始化；它是容量与训练结构对照，不能保证有效表示容量完全相同。

| 读出头 | 头参数量 | 测试准确率 | 有害 F1 | AUROC |
|---|---:|---:|---:|---:|
| 末层 MLP | 327,937 | 96.829 ± 0.164 | 96.955 ± 0.162 | 99.587 ± 0.016 |
| repeat_last | 1,656,709 | 96.859 ± 0.270 | 96.979 ± 0.282 | 99.595 ± 0.009 |
| 固定层融合 | 1,656,709 | 96.884 ± 0.045 | 97.009 ± 0.043 | 99.603 ± 0.008 |
| 验证集选层融合 | 1,656,709 | 96.867 ± 0.136 | 96.992 ± 0.145 | 99.606 ± 0.007 |

以上为三个 head seeds。固定融合相对 `repeat_last` 的准确率差值为 +0.026 ± 0.225 个百分点，选层融合为 +0.009 ± 0.372 个百分点，均有种子方向翻转。固定层为 8/16/24/32，验证集探索选层为 17/18/28/32。

逐层 probe 的错误并集存在互补，但 oracle 不能直接部署，也不是尚未训练的其他方法的严格上界。追加的全 32 层线性 logit stack 相对 L32 probe 曾得到约 +0.360 个百分点；该结果属于查看主结果后追加的探索诊断，与尚未完成的严格 nested group OOF 项目分开记录。

入口：[主报告](artifacts/outputs/hanguard_frozen_lora_layer_diagnostics_20261012/report.md)、[诊断解读](artifacts/outputs/hanguard_frozen_lora_layer_diagnostics_20261012/analysis/diagnosis.md)、[主结果](artifacts/outputs/hanguard_frozen_lora_layer_diagnostics_20261012/results.json)、[事后 stack](artifacts/outputs/hanguard_frozen_lora_layer_diagnostics_20261012/analysis/probe_logit_stacking.json)、[协议](artifacts/outputs/hanguard_frozen_lora_layer_diagnostics_20261012/protocol.json)。

## 输入相关动态门控

仍使用上述同一个冻结 LoRA，主结果固定阈值为 0.5，主要容量匹配对照使用全部五个 head seeds 42 至 46。

| 动态头 | 测试准确率 | 有害 F1 | 相对 dynamic_repeat_last 准确率差值 | 提升种子 |
|---|---:|---:|---:|---:|
| dynamic_repeat_last | 96.930 ± 0.075 | 97.057 ± 0.075 | — | — |
| dynamic_fixed | 96.912 ± 0.116 | 97.034 ± 0.117 | −0.018 ± 0.131 | 2/5 |
| dynamic_selected | 97.020 ± 0.108 | 97.145 ± 0.100 | +0.090 ± 0.114 | 4/5 |

动态门控没有形成几个百分点的准确率提升。动态头与旧静态头的次要比较只使用真实共有的 seeds 42、43、44；它们的参数量不同。评估器在 test barrier 后修正过静态种子交集处理，归档同时保留修订说明和已有预测精确重算审计。

入口：[报告](artifacts/outputs/hanguard_dynamic_gate_diagnostics_20261012/report.md)、[结果](artifacts/outputs/hanguard_dynamic_gate_diagnostics_20261012/results.json)、[评估修订](artifacts/outputs/hanguard_dynamic_gate_diagnostics_20261012/evaluation_amendment.json)、[重算审计](artifacts/outputs/hanguard_dynamic_gate_diagnostics_20261012/evaluation_primary_reproducibility.json)。

## A/B 训练内部诊断

该实验将训练数据进一步划分为 A 与 B。`base_last` 不使用 LoRA，两个 `adapted` 视图的 LoRA 只使用 A；B 的 20,048 条样本通过 group OOF 获得读出预测。所有选层、检查点和阈值在各折内部开发集确定，结果用于定位原因，不是正式测试成绩。

`best_single` 表示先由逐层线性 probe 选择层，再训练该层的 MLP；它不是训练全部 32 个单层 MLP 后选出的最佳 MLP。该口径与当前全量无 LoRA 实验的单层搜索不同。

| 冻结表示 | 末层 MLP | probe 选层单层 MLP | 固定层融合 | 固定融合加所选单层锚点 |
|---|---:|---:|---:|---:|
| 原始骨干末 token | 92.508 | 93.446 | 93.501 | 93.805 |
| A 上训练 LoRA 后的末 token | 96.364 | 96.663 | 96.713 | 96.483 |
| A 上训练 LoRA 后的均值池化 | 95.396 | 96.289 | 96.104 | 96.159 |

固定融合相对 `best_single` 在三个视图分别为 +0.055、+0.050、−0.185 个百分点，不能把相对较弱末层基线的全部提升归因于多层信息。跨表示对比同时涉及 LoRA 和池化方式，置信区间以固定训练产物为条件，不包含完整训练流程方差。

入口：[原始骨干报告](artifacts/outputs/hanguard_ab_diagnosis_20261012/readouts/base_last/analysis/report.md)、[LoRA 末 token 报告](artifacts/outputs/hanguard_ab_diagnosis_20261012/readouts/adapted_last/analysis/report.md)、[LoRA 均值池化报告](artifacts/outputs/hanguard_ab_diagnosis_20261012/readouts/adapted_mean/analysis/report.md)、[错误分析](artifacts/outputs/hanguard_ab_diagnosis_20261012/error_analysis/results.json)、[阈值敏感性](artifacts/outputs/hanguard_ab_diagnosis_20261012/independent_cpu_all_views_threshold_sensitivity_20261013/results.json)。

## 全文条件 query 与稀疏 token 读出

用户在查看中途表现后停止该项目；只完成计划五折中的两折，合计 8,022 条 B 内部 OOF 样本。以下只描述已完成部分。

| 读出方法 | 部分 OOF 准确率 | 相对 last_mlp 差值 |
|---|---:|---:|
| last_mlp | 96.796 | — |
| repeat_last | 96.784 | −0.012 |
| static_query_softmax | 96.572 | −0.224 |
| global_query_softmax | 96.634 | −0.162 |
| global_query_entmax15 | 96.510 | −0.287 |
| gated_attention_softmax | 96.609 | −0.187 |

这不是完整五折结果；停止决定受中途表现影响，不据此给出确认性显著性结论。与旧缓存的数值等价性检查未通过，表中方法只在该次共同缓存内比较。稀疏注意力位置也不等于经过标注或因果验证的有害证据。

入口：[停止后的聚合分析](artifacts/outputs/hanguard_sparse_token_readout_20261013_v3/stopped_analysis/results.json)、[停止记录](artifacts/outputs/hanguard_sparse_token_readout_20261013_v3/user_stop_request.json)、[WildGuard 相似性诊断](artifacts/outputs/hanguard_sparse_token_readout_20261013_v3/stopped_analysis/wildguard_similarity_diagnostic/results.json)、[协议](artifacts/outputs/hanguard_sparse_token_readout_20261013_v3/protocol.json)。

## 数据审计与整改的先后关系

2026 年 9 月 28 日完成历史 WildGuard 全文重译和流程 QA，得到 77,686 条修复数据。该 QA 验证了来源、生成及固定样本，但不等于全量语义认证。[历史修复报告](artifacts/outputs/hanguard_translation_repair_20260928/report.md)

10 月 12 日早期的错误定向审计关注 WildGuard 共错与类别来源，其样本不能估计总体错误率；其中对翻译作用的判断仅适用于当时审阅范围。[定向审计](artifacts/outputs/hanguard_targeted_error_label_audit_20261012/report.md)

随后对固定 79 条模型分歧样本的原文对照发现 6 条明确局部意义错误、21 条轻微措辞问题、3 条源语境歧义、45 条未见明确意义问题和 4 条无公开原文。该集合由模型分歧选取，不能外推全库翻译错误率，也不能将局部错译直接等同于公开二分类标签错误。[79 条报告](artifacts/outputs/hanguard_disagreement79_translation_analysis_20261012/report.md)

独立全量审计进一步发现类别冲突筛选影响二分类覆盖、旧长度规则遗漏完整 JailBench 包装家族，以及成族翻译与分组问题。200 条简单随机语义样本中有 5 条重要语义偏移；这项 2.5% 抽样统计不是二分类标签错误率。[全量审计](artifacts/outputs/hanguard_full_data_audit_20261012/report.md)、[审计聚合结果](artifacts/outputs/hanguard_full_data_audit_20261012/audit_summary.json)

整改最终发布三来源 83,502 条数据，训练/验证/测试为 66,802/8,350/8,350；已修正 218 个独立翻译单位，补回覆盖并重新分组、划分与验收。原公开 binary 标签及来源原文未修改；仍不能认证所有文本语义与来源标注绝对正确。数据整改没有直接生成模型准确率，也不能与历史版分数相减计算算法收益。

入口：[整改报告](artifacts/outputs/hanguard_data_remediation_20261012/report.md)、[独立主集验收](artifacts/outputs/hanguard_data_remediation_20261012/independent_release_qa.json)、[独立来源审计](artifacts/outputs/hanguard_data_remediation_20261012/independent_release_source_audit.json)、[控制视图验收](artifacts/outputs/hanguard_data_remediation_20261012/independent_release_views_qa.json)、[训练入口验收](artifacts/outputs/hanguard_data_remediation_20261012/activation_verification.json)、[新版主集说明](artifacts/data/three_source_remediated_20261012/README.md)、[最终发布 QA](artifacts/data/three_source_remediated_20261012/release_qa.json)。

另有 [JailBench 有害 seed 长度占比与任务结构统计](artifacts/outputs/hanguard_task_fit_research_20261013/jailbench_seed_fraction_audit/results.json)，用于描述信号与包装的关系，不是新增分类效果证明。

## 未完成项目与当前实验

- **新版全量无 LoRA 对照**：原始 Qwen3.5-4B 冻结骨干，比较末层 MLP、全部 32 层 MLP 中由验证集选择的单层、固定多层融合及容量/锚点对照。训练与验证特征已完成，当前分类头按新资源策略等待空卡，测试集尚未开启，没有最终准确率。归档中的状态是静态快照，不会随后台任务更新。[协议](artifacts/outputs/hanguard_frozen_base_full_readouts_20261013_v5/protocol.json)、[状态快照](artifacts/outputs/hanguard_frozen_base_full_readouts_20261013_v5/state.json)
- **独立 LoRA 多 seed**：已登记三个独立 LoRA seeds，但三个 run 未形成最终 selection 和 test 结果；遗留 `training` 状态不能当作当前进程仍活跃的证据。现有冻结 LoRA 上的多个 head seeds 不能替代该实验。[协议](artifacts/outputs/hanguard_e03_lora_multiseed_20261012/protocol.json)
- **严格 nested group OOF logit stack**：仅见部分基础 probe 折训练记录，未形成最终选择与评估结果。它与已经归档的事后线性 stack 是两个项目。[协议](artifacts/outputs/hanguard_oof_probe_logit_stack_20261012/protocol.json)

各历史测试身份已参与方法探索；新版整改也不会自动将其变成全新的外部盲测。逐来源、逐层和多组配对检验属于探索分析，具体重采样单位、阈值规则及多重比较限制以各实验协议和报告为准。
