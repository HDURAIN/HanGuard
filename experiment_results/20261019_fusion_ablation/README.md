# 多层融合结果与消融归档（20261019）

主要研究路线采用冻结原始 Qwen3.5-4B 的无 LoRA 读出。以下两种测试范围必须分开报告，所有同范围方法使用完全相同的样本、既有验证集阈值和检查点。

| 范围 | 最佳单层 L19 | 锚点多层融合 | 差值 |
|---|---:|---:|---:|
| 原始完整测试集，8,350 条 | 94.5868% | 95.1976% | +0.6108 个百分点 |
| 固定排除 122 条后的子集，8,228 条 | 94.5066% | 96.6091% | +2.1026 个百分点 |
| 同一子集上的新冻结 LoRA 对照，8,228 条 | 97.2654% | 97.4842% | +0.2188 个百分点 |

122 条子集通过查看旧无 LoRA 模型错误后确定：排除 WildGuard 中“最佳单层正确、锚点融合错误”的 122 条。`posthoc_error_selected_test_subset` 是结果依赖的事后子集，+2.1026 是该固定子集上可复算的描述性增益，不能写成未经筛选测试集上的提升或数据质量整改收益。训练与验证样本没有因此删除。原始完整测试结果同时保留。

新补充消融在 `outputs/hanguard_nolora_ablation_20261019` 登记。缺失结果以 `pending` 表示，不能把计划当成已经完成的实验。新结果沿用固定 122 个 ID，不再按新模型错误更改子集。

| 实验 | 状态 | 来源 |
|---|---|---|
| `nolora_original` | complete | `outputs/hanguard_frozen_base_full_readouts_20261013_v5/analysis/results.json` |
| `lora_selected122` | complete | `outputs/hanguard_selected122_lora_20261018/readouts/analysis/results.json` |
| `nolora_fixed_selected122` | complete | `data/three_source_wildguard122_subset_20261018/old_model_subset_metrics.json` |
| `nolora_additional_ablation` | complete | `outputs/hanguard_nolora_ablation_20261019` |
| `nolora_reference_analysis` | complete | `outputs/hanguard_nolora_reference_analysis_20261019` |

- [汇总结果 CSV](results_summary.csv)：准确率、F1 等以百分比表示，差值以百分点表示；`evaluation_scope` 与 `threshold_policy` 区分口径。
- [实验状态索引](index.json)：区分已完成与待补充项。
- [SHA256 清单](manifest.json)：保存归档文件哈希、来源路径与来源哈希；转换过的逐样本文件注明转换方式。
- [源码依赖索引](source_dependencies.json)：协议及队列绑定的仓库内小型 Python 源码，以及静态分析得到的本项目 import 依赖；保留源码原相对路径与已登记 SHA256 的匹配情况。
- 已有结果的 CPU 复核报告、逐来源结果、阈值敏感性、错误互补与 32 层验证 probe 表位于 `artifacts/outputs/hanguard_nolora_reference_analysis_20261019/`。对应分析、消融、调度和打包源码快照位于 `artifacts/scripts/hanguard/`。
- `artifacts/` 保留来源相对路径。逐样本 CSV 仅保留 ID、标签、来源、长度及预测数值；完整中文、英文文本、权重、大型特征缓存、运行临时文件均不归档。

在仓库根目录刷新或校验：

```bash
python scripts/hanguard/package_fusion_ablation_results.py
python scripts/hanguard/package_fusion_ablation_results.py --verify
```

刷新只写归档目录，不移动或删除实验原目录，不重写已注册协议、科学源码、模型或数据。归档里的路径和协议可能引用本机未归档的大文件；重放训练仍需原始数据、基座和特征。

## 组件消融

相对增益按“该消融准确率 − 同设备同预算完整结构 `anchor_replay` 准确率”计算；负值表示移除该组件后下降。旧 GPU seed 42 结果不作为 CPU 组件消融的直接基线。层组可在训练前登记，准确率只有完成测试评估后才显示。

### full_test_8350

| 方法 | 状态 | 准确率 % | 相对 replay 增益/百分点 | 参数量（总/可训练） | 层组 | 锚点层 | 设备 |
|---|---|---:|---:|---:|---|---|---|
| `anchor_replay` | completed | 94.9820 | +0.0000 | 1,656,709/1,656,709 | 8/16/24/32 | 19 | cpu |
| `uniform_weights` | completed | 95.1856 | +0.2036 | 1,656,709/1,656,705 | 8/16/24/32 | 19 | cpu |
| `no_anchor` | completed | 95.0299 | +0.0479 | 1,328,901/1,328,901 | 8/16/24/32 | none | cpu |
| `shared_projection` | completed | 94.9940 | +0.0120 | 672,517/672,517 | 8/16/24/32 | 19 | cpu |
| `top4` | completed | 94.9341 | -0.0479 | 1,656,709/1,656,709 | 15/17/18/19 | 19 | cpu |
| `two_layers` | completed | 95.1737 | +0.1916 | 1,000,579/1,000,579 | 16/32 | 19 | cpu |
| `mean_pool` | completed | 93.9760 | -1.0060 | 327,937/327,937 | 8/16/24/32 | none | cpu |

### selected_test_8228

| 方法 | 状态 | 准确率 % | 相对 replay 增益/百分点 | 参数量（总/可训练） | 层组 | 锚点层 | 设备 |
|---|---|---:|---:|---:|---|---|---|
| `anchor_replay` | completed | 96.1351 | +0.0000 | 1,656,709/1,656,709 | 8/16/24/32 | 19 | cpu |
| `uniform_weights` | completed | 96.2810 | +0.1458 | 1,656,709/1,656,705 | 8/16/24/32 | 19 | cpu |
| `no_anchor` | completed | 96.0379 | -0.0972 | 1,328,901/1,328,901 | 8/16/24/32 | none | cpu |
| `shared_projection` | completed | 95.8921 | -0.2431 | 672,517/672,517 | 8/16/24/32 | 19 | cpu |
| `top4` | completed | 95.5518 | -0.5834 | 1,656,709/1,656,709 | 15/17/18/19 | 19 | cpu |
| `two_layers` | completed | 96.0501 | -0.0851 | 1,000,579/1,000,579 | 16/32 | 19 | cpu |
| `mean_pool` | completed | 94.7739 | -1.3612 | 327,937/327,937 | 8/16/24/32 | none | cpu |

## 三个 head seed 的核心五臂

仅 seeds 42、43、44 全部完成后显示三种子均值 ± 样本标准差；seed 42 复用既有模型。缺失种子不以计划结果填入均值。

| 范围 | 方法 | 状态 | 已完成 seeds | 准确率均值 ± SD / % |
|---|---|---|---|---:|
| full_test_8350 | `last` | completed | 42/43/44（3/3） | 93.8882 ± 0.1635 |
| full_test_8350 | `best_single` | completed | 42/43/44（3/3） | 94.6028 ± 0.0850 |
| full_test_8350 | `repeat_best` | completed | 42/43/44（3/3） | 94.6427 ± 0.2093 |
| full_test_8350 | `fusion_fixed` | completed | 42/43/44（3/3） | 95.1377 ± 0.1882 |
| full_test_8350 | `fusion_fixed_best_anchor` | completed | 42/43/44（3/3） | 95.1617 ± 0.0622 |
| selected_test_8228 | `last` | completed | 42/43/44（3/3） | 94.4336 ± 0.1589 |
| selected_test_8228 | `best_single` | completed | 42/43/44（3/3） | 94.8307 ± 0.2970 |
| selected_test_8228 | `repeat_best` | completed | 42/43/44（3/3） | 94.9400 ± 0.2675 |
| selected_test_8228 | `fusion_fixed` | completed | 42/43/44（3/3） | 95.9083 ± 0.0574 |
| selected_test_8228 | `fusion_fixed_best_anchor` | completed | 42/43/44（3/3） | 96.0703 ± 0.4698 |

### 逐 seed 锚点融合主增益

| 范围 | seed | 状态 | 最佳单层准确率 % | 锚点融合准确率 % | 融合增益/百分点 |
|---|---:|---|---:|---:|---:|
| full_test_8350 | 42 | completed | 94.5868 | 95.1976 | +0.6108 |
| full_test_8350 | 43 | completed | 94.5269 | 95.1976 | +0.6707 |
| full_test_8350 | 44 | completed | 94.6946 | 95.0898 | +0.3952 |
| selected_test_8228 | 42 | completed | 94.5066 | 96.6091 | +2.1026 |
| selected_test_8228 | 43 | completed | 94.8955 | 95.8556 | +0.9601 |
| selected_test_8228 | 44 | completed | 95.0899 | 95.7462 | +0.6563 |

`full_test_8350` 三种子融合增益均值 ± 样本 SD：+0.5589 ± 0.1449 个百分点。

`selected_test_8228` 三种子融合增益均值 ± 样本 SD：+1.2397 ± 0.7626 个百分点。

## 消融覆盖

当前 20/20 项已完成，逐项状态和证据路径见 [覆盖清单](ablation_coverage.csv)。组件数值见 [组件表](component_ablation.csv)，三种子汇总见 [种子汇总](seed_summary.csv)，逐种子差值见 [种子增益](seed_gains.csv)。
