# 训练与实验脚本导航

截至2026-10-25，本目录100个Python文件按用途与状态归档索引。源码保留原位置：已登记的实验协议绑定文件路径和SHA，移动或改写会破坏追溯。这里区分当前证据入口、部署实现和历史工具，不自动启动任何实验。

## 先从这里开始

| 目的 | 入口 |
|---|---|
| 写论文、复制四组结果表 | [论文实验主稿](../../docs/paper_experiments_20261025.md) · [论文结果包](../../experiment_results/20261025_paper_ready/README.md) |
| 核查容量、选层、逐类和效率对照 | [补充证据](../../docs/paper_supporting_evidence_20261025.md) |
| 查看主实验完成状态与协议 | [主实验归档](../../experiment_results/20261022_clean_ablation/README.md) · [实验总索引](../../docs/experiment_index.md) |
| 核查现成安全模型原生能力 | [原生Qwen3Guard结果](../../experiment_results/20261025_qwen3guard_clean_test_v2/README.md) |
| 训练或导出部署包 | [训练指南](../../docs/training.md) · [项目说明](../../README.md) |

主线方法统一称**类别查询注意力聚合（CQA）＋固定多层残差融合（MLF）**。20261022实验在同一数据与Qwen3.5-4B骨干上完成二分类/五分类各9组×3种子，共54个正式头；32层×2任务×3种子，共192个线性probe。四组主消融为普通MLP、仅MLF、仅CQA、CQA＋MLF。融合层为L8/16/24/32，各层系数固定为0.25；投影与输出映射参与训练。二分类、独立五分类和双阶段系统分别使用8,350、2,180、3,704条测试范围，不能混用分母。

## 常用检查命令

从仓库根目录运行。下列命令只查询状态、导出已有结果或校验，不启动训练或模型推理；本机使用`.venv-hanguard`，在其他环境需替换为满足项目依赖的Python。

```bash
# 当前论文材料及其来源校验
.venv-hanguard/bin/python scripts/hanguard/package_paper_experiment_materials.py --verify

# 从已有结果重新导出CSV/LaTeX与写作材料清单
.venv-hanguard/bin/python scripts/hanguard/package_paper_experiment_materials.py

# 主实验轻量归档完整性
.venv-hanguard/bin/python scripts/hanguard/package_clean_ablation_results.py --verify

# 本机当前主实验、原生Guard、补充实验状态
.venv-hanguard/bin/python scripts/hanguard/clean_ablation_study.py status --output outputs/hanguard_clean_ablation_20261022
.venv-hanguard/bin/python scripts/hanguard/qwen3guard_clean_eval.py status --output outputs/qwen3guard_clean_test_20261025_v2
.venv-hanguard/bin/python scripts/hanguard/paper_followup_study.py status --output outputs/hanguard_paper_followups_20261025
```

状态命令需要对应本机`outputs/`文件；仅有Git轻量归档的环境应直接阅读结果包。`package_clean_ablation_results.py`不带`--verify`会刷新归档，不能把它理解为只读检查。当前实验已完成，历史队列的`run`不列入日常命令。新增实验应使用新输出目录及登记协议，先核查CLI和资源占用，不覆盖旧记录。

## 当前论文实验与结果导出

| 脚本 | 用途与状态 |
|---|---|
| [package_paper_experiment_materials.py](package_paper_experiment_materials.py) | 纯CPU：从已完成结果重算四组消融、原生Guard及补充对照，导出CSV/LaTeX并校验写作稿；当前论文首选入口。 |
| [package_clean_ablation_results.py](package_clean_ablation_results.py) | 纯CPU：刷新/校验20261022主实验轻量证据包，保留注册源码、选择记录与无原文预测。 |
| [clean_ablation_study.py](clean_ablation_study.py) | 主实验登记、调度与状态入口；本轮已完成。`init/run/worker`会登记或执行实验，整理仓库时不用运行。 |
| [clean_ablation_features.py](clean_ablation_features.py) | 从同一次骨干前向提取末层有效token与全部层末位置状态，校验缓存身份。 |
| [clean_ablation_heads.py](clean_ablation_heads.py) | 四组主方法、同容量重复末层、共享注意力和选层单层等九组分类组件。 |
| [clean_ablation_training.py](clean_ablation_training.py) | 全层线性probe、验证选层、可恢复训练、检查点/阈值锁定及测试汇总。 |
| [clean_ablation_io.py](clean_ablation_io.py) | 阻止新实验意外读取旧实验产物的I/O边界。 |
| [clean_ablation_runtime.py](clean_ablation_runtime.py) | 分类组件与特征提取的进程内确定性设置。 |

## 原生Qwen3Guard对照

| 脚本 | 用途与状态 |
|---|---|
| [qwen3guard_clean_eval.py](qwen3guard_clean_eval.py) | 官方生成式原生判定、分块收据、解析、双映射评分与原生结果归档；正式使用`20261025_v2`输出。 |
| [qwen3guard_shared_runner.py](qwen3guard_shared_runner.py) | 当时已授权的显存预算共享卡修订调度；不改变原生模板、模型和评分，不是新增方法。 |
| [qwen3guard_clean_queue.py](qwen3guard_clean_queue.py) | 原始空卡调度器及队列共同实现；共享修订入口复用此模块。 |

## 已完成的论文补充实验

| 脚本 | 用途与状态 |
|---|---|
| [paper_followup_study.py](paper_followup_study.py) | 补充实验统一登记、调度、状态及归档；本轮已完成，不需要重复启动队列。 |
| [paper_pooling_controls.py](paper_pooling_controls.py) | 同划分的均值池化/独立共享注意力普通分类器；保留强对照，支撑增益归因。 |
| [paper_external_classifier.py](paper_external_classifier.py) | MacBERT普通五分类的本地监督参考，骨干参与训练；不是原生国标五分类安全模型。 |
| [paper_error_efficiency.py](paper_error_efficiency.py) | 既有预测的CPU错误分析，以及真实在线端到端GPU效率测量；效率分支会加载骨干。 |
| [paper_followup_recovery.py](paper_followup_recovery.py) | 当时解决MacBERT剩余测试锁争用的恢复包装器，保留执行记录用途。 |

## 部署训练与共用实现

| 脚本 | 用途与状态 |
|---|---|
| [train.py](train.py) · [current_training.py](current_training.py) | 部署训练入口及注册/运行实现；使用范围和CLI见[训练指南](../../docs/training.md)，与当前论文固定骨干消融区分。 |
| [export_model.py](export_model.py) | 导出已选择的有害识别骨干和风险类型头；输出为部署包，不是论文四组结果包。 |
| [repaired_study.py](repaired_study.py) · [repaired_heads.py](repaired_heads.py) · [repaired_references.py](repaired_references.py) · [report_repaired_study.py](report_repaired_study.py) | 历史二分类训练、融合/读出、固定比较规则和报告；部署训练及旧实验仍会引用。 |
| [multilabel_heads.py](multilabel_heads.py) · [multilabel_study.py](multilabel_study.py) | 类别查询组件及历史特征/研究工具；旧文件名不代表当前任务是多标签，当前两任务均为softmax互斥分类。 |
| [primary_category_contract.py](primary_category_contract.py) · [primary_category_study.py](primary_category_study.py) · [primary_category_queue.py](primary_category_queue.py) · [primary_category_report.py](primary_category_report.py) | 既有单主类风险五分类的合约、训练、调度和报告；保留部署/历史依赖。 |

## 数据合约、构建与翻译审计工具

| 脚本 | 用途与状态 |
|---|---|
| [remediated_dataset_contract.py](remediated_dataset_contract.py) · [remediated_binary_study.py](remediated_binary_study.py) | 正式发布的标签/划分合约和整改数据二分类包装器。 |
| [prepare_two_source_primary.py](prepare_two_source_primary.py) | 按既有划分导出两来源类型监督，不自动标注、不重划分。 |
| [translation_provenance.py](translation_provenance.py) | 追溯翻译来源与文本身份，生成独立审计材料。 |
| [repair_wildguard_translation.py](repair_wildguard_translation.py) | 按源文本翻译并保存可恢复记录；属于数据生成操作，可能调用模型。 |
| [translation_literal_guard.py](translation_literal_guard.py) | 翻译前后保护并恢复代码字面量，不执行文本中的代码。 |
| [assemble_translation_reviews.py](assemble_translation_reviews.py) | 汇集既有文本复核记录，不自动作出质量判断。 |
| [merge_translation_repair.py](merge_translation_repair.py) | 合并生产翻译、身份与复核结果。 |
| [audit_translation_repair.py](audit_translation_repair.py) · [audit_translation_repair_outputs.py](audit_translation_repair_outputs.py) | 模型无关初审与独立CPU回读核验。 |
| [build_translation_repaired_dataset.py](build_translation_repaired_dataset.py) | 构建隔离的数据候选发布，不覆盖当前正式发布。 |

## 历史：早期四组重跑与缓存排查

| 脚本 | 用途与状态 |
|---|---|
| [four_arm_experiment_study.py](four_arm_experiment_study.py) · [four_arm_experiment_training.py](four_arm_experiment_training.py) · [four_arm_query_heads.py](four_arm_query_heads.py) · [four_arm_token_cache.py](four_arm_token_cache.py) | 20261019早期四组实验与缓存实现；因特征不一致停止，由20261022新实验取代。 |
| [four_arm_feature_mismatch_diagnostic.py](four_arm_feature_mismatch_diagnostic.py) · [four_arm_feature_replay_queue.py](four_arm_feature_replay_queue.py) | 上述缓存不一致排查与重放；不是当前论文的恢复入口。 |
| [package_four_arm_experiments.py](package_four_arm_experiments.py) | 只打包/校验上述历史证据，不将其成绩填入当前主表。 |

## 历史：冻结骨干五组与调度版本

| 脚本 | 用途与状态 |
|---|---|
| [frozen_base_readout_study.py](frozen_base_readout_study.py) · [frozen_base_readout_heads.py](frozen_base_readout_heads.py) | 旧五组科学实现与分类组件，已登记产物仍绑定其源码。 |
| [frozen_base_parallel_study.py](frozen_base_parallel_study.py) · [frozen_base_parallel_features.py](frozen_base_parallel_features.py) · [frozen_base_resumable_heads.py](frozen_base_resumable_heads.py) | 旧并行提取、实验和可恢复训练实现。 |
| [frozen_base_parallel_queue.py](frozen_base_parallel_queue.py) · [frozen_base_parallel_queue_v2.py](frozen_base_parallel_queue_v2.py) · [frozen_base_parallel_queue_v3.py](frozen_base_parallel_queue_v3.py) · [frozen_base_parallel_queue_v4.py](frozen_base_parallel_queue_v4.py) | 历史调度版本，不是同时运行的四个队列；保留路径供协议追溯。 |
| [frozen_base_readout_queue.py](frozen_base_readout_queue.py) · [frozen_base_readout_admission.py](frozen_base_readout_admission.py) · [frozen_base_readout_admission_v2.py](frozen_base_readout_admission_v2.py) · [frozen_base_cooperative_runner.py](frozen_base_cooperative_runner.py) | 旧任务队列、入场与协作运行包装器。 |
| [frozen_base_locked_test_resume.py](frozen_base_locked_test_resume.py) · [frozen_base_post_validation_recovery.py](frozen_base_post_validation_recovery.py) · [frozen_base_recovery_coordinator.py](frozen_base_recovery_coordinator.py) · [frozen_base_reviewed_head_restart.py](frozen_base_reviewed_head_restart.py) · [frozen_base_test_monitor.py](frozen_base_test_monitor.py) | 旧锁定测试、异常恢复、重启及监控工具；不能依据旧状态文件认定作业仍在运行。 |

## 历史：固定层、probe与A/B诊断

| 脚本 | 用途与状态 |
|---|---|
| [frozen_lora_layer_diagnostics.py](frozen_lora_layer_diagnostics.py) · [dynamic_gate_diagnostics.py](dynamic_gate_diagnostics.py) | 同一冻结适配器下的逐层/容量/动态门控分析。 |
| [diagnostic_ab_split.py](diagnostic_ab_split.py) | 仅在训练数据内构建分组隔离的A/B诊断划分。 |
| [ab_diagnostic_experiment.py](ab_diagnostic_experiment.py) · [ab_diagnostic_queue.py](ab_diagnostic_queue.py) · [ab_diagnostic_followup.py](ab_diagnostic_followup.py) · [ab_diagnostic_dual_gpu_followup.py](ab_diagnostic_dual_gpu_followup.py) · [ab_diagnostic_gap_single_followup.py](ab_diagnostic_gap_single_followup.py) | A/B诊断及其后续调度，不是正式留出测试实验。 |
| [ab_frozen_readout_diagnostics.py](ab_frozen_readout_diagnostics.py) · [ab_readout_parallel_queue.py](ab_readout_parallel_queue.py) · [ab_probe_rescue_audit.py](ab_probe_rescue_audit.py) · [ab_diagnostic_error_analysis.py](ab_diagnostic_error_analysis.py) | A/B读出、全层probe、救回/新增错误与互补审计。 |
| [oof_probe_logit_stack.py](oof_probe_logit_stack.py) · [multiseed_fixed_layer_replication.py](multiseed_fixed_layer_replication.py) · [finalize_e03_lora_multiseed.py](finalize_e03_lora_multiseed.py) · [external_holdout_evaluation.py](external_holdout_evaluation.py) | 旧交叉拟合、独立适配器种子、固定层复制与外部留出补充；结果按各自协议解释。 |

## 历史：子集与细组件消融

| 脚本 | 用途与状态 |
|---|---|
| [selected_subset_lora_study.py](selected_subset_lora_study.py) · [build_selected_test_subset.py](build_selected_test_subset.py) | 历史固定测试子集构建与对照；当前主实验不读取该子集或其模型。 |
| [fragment_clean_lora_study.py](fragment_clean_lora_study.py) · [build_fragment_filtered_dataset.py](build_fragment_filtered_dataset.py) | 已停用的片段筛选方案；前者仍被历史子集包装器复用，不能删除依赖文件。 |
| [nolora_fusion_ablation.py](nolora_fusion_ablation.py) · [nolora_ablation_queue.py](nolora_ablation_queue.py) | 旧缓存/旧数据范围上的细组件与种子消融，已完成；不作为当前主实验队列。 |
| [analyze_fusion_reference_results.py](analyze_fusion_reference_results.py) · [package_fusion_ablation_results.py](package_fusion_ablation_results.py) | 历史结果CPU复核与轻量证据打包；完整集/子集范围分别保留。 |

## 已停止、搁置及一次性维护工具

| 脚本 | 用途与状态 |
|---|---|
| [sparse_token_feature_cache.py](sparse_token_feature_cache.py) · [sparse_token_readout_heads.py](sparse_token_readout_heads.py) · [sparse_token_readout_experiment.py](sparse_token_readout_experiment.py) · [sparse_token_readout_queue.py](sparse_token_readout_queue.py) | 稀疏token读出探索，已按决定停止；部分OOF输出不能替代完成的正式实验。 |
| [qwen3guard_supervised_features.py](qwen3guard_supervised_features.py) · [qwen3guard_supervised_features_v2.py](qwen3guard_supervised_features_v2.py) · [qwen3guard_supervised_study.py](qwen3guard_supervised_study.py) · [qwen3guard_supervised_study_v2.py](qwen3guard_supervised_study_v2.py) · [qwen3guard_supervised_training.py](qwen3guard_supervised_training.py) | 附加监督适配实验；仅保留补充审计，不是Qwen3Guard原生能力，不能用于原生对比表。 |
| [encoder_guard_native_prepare.py](encoder_guard_native_prepare.py) | 搁置中的CPU输入准备草稿；没有模型推理或对比成绩，不恢复执行。 |
| [cleanup_repository.py](cleanup_repository.py) | 20260928固定清理清单工具；包含实际删除入口，本次整理未执行，不是日常清理命令。 |

## 维护约定

- `outputs/`保留本机数据副本、特征、权重、预测、日志和进程状态；`experiment_results/`保留可版本管理的轻量证据；脚本导航不代替冻结协议。
- 不修改注册源码以迎合旧哈希，不移动其绑定路径，不覆盖正式数据发布。新代码或新配置另行登记。
- 原生Guard结果与本地监督适配严格分开；原生五分类N/A不能用追加分类头的分数填补。
- 查询聚合使用的共用模块可能有历史“multilabel”命名；当前论文是二分类与单标签五分类，任务定义以实际训练器为准。
- 进程清理和GPU调度需核对创建者及进程身份；维护文档不会授权停止其他用户进程。

数据范围见[数据说明](../../docs/data.md)，已归档实验及历史完成状态见[实验总索引](../../docs/experiment_index.md)。
