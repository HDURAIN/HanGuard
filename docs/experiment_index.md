# 实验与结果入口

按包浏览请看[结果目录](../experiment_results/README.md)；按用途定位实现请看[脚本导航](../scripts/hanguard/README.md)。下方保留实验原始范围和历史状态，不代表应重新启动旧队列。

论文实验部分的当前入口为[20261025实验主稿](paper_experiments_20261025.md)、[补充证据](paper_supporting_evidence_20261025.md)及[论文结果包](../experiment_results/20261025_paper_ready/README.md)。统一命名“类别查询注意力聚合（CQA）＋固定多层残差融合（MLF）”，四组主消融准确率分别为二分类93.85/94.91/95.86/96.12%、五分类77.28/77.89/82.20/83.13%；由[论文表格打包工具](../scripts/hanguard/package_paper_experiment_materials.py)重算并校验。此入口只整理完成的实验，不改变下方原始归档。

当前主实验为 **20261022 全新消融重跑**，只读取正式数据发布与原始 Qwen3.5-4B 基座：原文重新分词，统一提取全文 token 与全部层末 token，重新训练与选层。历史实验特征、检查点、预测、结果和测试排除名单均不作为输入。主四组为普通 MLP、固定等权多层融合、类别向量、类别向量＋融合；另有四组机制对照及验证集选层单层 MLP。二分类、五分类各 9 组 × 3 个种子，共 54 个正式头；全层线性 probe 共 192 个。骨干冻结，无 LoRA。

二分类使用 66,802/8,350/8,350，测试只报告完整 8,350 条；已知有害类型监督使用 17,440/2,180/2,180；完整系统在 3,704 条已知最终标签样本上派生三个组合表。数据范围来自既有正式标签，不用旧实验的 122 条排除名单，也不根据新结果删样本。全部选层、检查点和阈值只依据新验证预测，锁定后再开放测试。

新入口为 [clean_ablation_study.py](../scripts/hanguard/clean_ablation_study.py)，新证据包位于 [20261022_clean_ablation](../experiment_results/20261022_clean_ablation/README.md)，通过 [package_clean_ablation_results.py](../scripts/hanguard/package_clean_ablation_results.py) 刷新及 `--verify` 校验。源码注册哈希、状态、逐种子表和预测均来自新目录；未完成成绩保留 `pending`。

| 实验 | 本机产物 | 状态及口径 |
|---|---|---|
| Qwen3Guard-Gen-4B 原始能力：正式外部对照 | `outputs/qwen3guard_clean_test_20261025_v2/` | 已完成并复核；官方单 user 模板直接生成，无新增分类头或本地训练。完整 8,350 条，推理前固定的 Unsafe-only / Unsafe＋Controversial 两种二分类映射准确率分别为 90.8623% / 94.8263%；原生九类与本地五类无可靠直接映射，五分类为 N/A。见[原版结果](../experiment_results/20261025_qwen3guard_clean_test_v2/README.md)及[原始能力对比](../experiment_results/20261025_qwen3guard_clean_test_v2/ORIGINAL_CAPABILITY_COMPARISON.md) |
| Qwen3Guard 附加监督适配：补充实验 | `outputs/qwen3guard_supervised_comparison_20261025_v2/` | 18/18 头已完成并核验；这是误将“补齐二/五分类”扩展为训练普通分类头的附加任务，不作为用户要求的原始 Qwen3Guard 外部基线。97.4371% / 83.3333% 只代表共享注意力监督适配，不能替代原生结果。科学协议、原始数值和预测保留，见[用途纠正](../experiment_results/20261025_qwen3guard_supervised_comparison_v2/EVALUATION_SCOPE.md)及[附加实验结果](../experiment_results/20261025_qwen3guard_supervised_comparison_v2/README.md) |
| 三项论文补充实验 | `outputs/hanguard_paper_followups_20261025/` | 两个通用全文读出×二分类/五分类×三seed，共12头；MacBERT五分类三seed；CPU错误分析及空卡在线效率，见[证据包](../experiment_results/20261025_paper_followups/README.md) |
| 当前主实验：全新缓存、四组＋机制对照＋单层基线 | `outputs/hanguard_clean_ablation_20261022/` | 不读取旧实验产物；54 个正式头＋192 个全层 probe，见[新证据包](../experiment_results/20261022_clean_ablation/README.md) |
| 历史未完成四组：固定等权融合 × 类别向量 | `outputs/hanguard_four_arm_experiments_20261019/` | 因新旧缓存不一致停止；只保留排查记录，不作为新实验输入 |
| 新版无 LoRA 五组历史对照 | `outputs/hanguard_frozen_base_full_readouts_20261013_v5/analysis/results.json` | 已完成；完整 8,350 条，最佳单层 94.5868%，锚点融合 95.1976%，+0.6108 个百分点 |
| 同一无 LoRA 模型固定 122 条子集 | `data/three_source_wildguard122_subset_20261018/old_model_subset_metrics.json` | 已完成重算；8,228 条，94.5066% → 96.6091%，+2.1026 个百分点；结果依赖的事后子集 |
| 同一子集上的新 LoRA 五组对照 | `outputs/hanguard_selected122_lora_20261018/readouts/analysis/results.json` | 已完成；同一冻结 LoRA，97.2654% → 97.4842%，+0.2188 个百分点 |
| 无 LoRA 组件及 head seed 消融 | `outputs/hanguard_nolora_ablation_20261019/` | 状态以该目录 `queue_state.json` 和结果归档 `index.json` 为准；完整集与固定子集双表 |
| 已有结果 CPU 独立复核 | `outputs/hanguard_nolora_reference_analysis_20261019/` | 已完成；逐来源、阈值敏感性、错误互补、32 层验证读出与现有参数量对照 |
| 历史已归档实验 | [20261015 归档](../experiment_results/20261015/README.md) | 历史 LoRA、类别 query、层 probe、动态门控、A/B 诊断及稀疏 token 读出；不可混用数据版本 |

以下为 20261019 及更早实验的历史说明，不是当前重跑的输入或执行步骤。

历史未完成四组的轻量证据包位于 `experiment_results/20261019_four_arm_experiments/`，由 [旧四组打包工具](../scripts/hanguard/package_four_arm_experiments.py)生成。其固定等权设置为 L8/L16/L24/L32 各 0.25，无最佳单层锚点；旧 +2.1026 个百分点不能填进固定等权结果。

20261019 旧协议的二分类还包含固定的 8,228 条事后子集；这一范围不进入 20261022 新重跑。旧协议的经典完整系统 2×2 为 binary gate `baseline/fusion_only` × type head `baseline/category_only`，类型阶段也加入融合的扩展组合另表报告。各自保留分母，未知类型不重标。

旧四组入口 [four_arm_experiment_study.py](../scripts/hanguard/four_arm_experiment_study.py) 仅保留审计，不继续该作业；新的全流程改用上方 `clean_ablation_study.py`。

旧五组结果与已完成的 20/20 项细组件/seed 补充消融的轻量证据包统一写入 `experiment_results/20261019_fusion_ablation/`，由 [打包工具](../scripts/hanguard/package_fusion_ablation_results.py)生成，包含协议、结果、无原文逐样本预测、状态索引和 SHA256 清单。归档是运行时点快照，使用以下命令刷新及校验：

```bash
python scripts/hanguard/package_fusion_ablation_results.py
python scripts/hanguard/package_fusion_ablation_results.py --verify
```

补充消融复用既有冻结基座特征；新 seed 的检查点、最佳单层及阈值继续由验证集选择。测试子集沿用固定 122 个 ID，不按新结果追加筛选。完整集和子集差值分列，LoRA 对照另列。具体组件、训练预算及完成状态以新实验协议和结果为准。

已完成的补充训练矩阵为 seed 43、44 各自完整重跑五个核心对照，以及 seed 42 的六个组件对照：等权融合、去掉直接锚点、共享分支投影、验证 BCE 排名前四层、双层 L16/L32、四层原始特征均值后 MLP。组件组额外重跑完整锚点结构 `anchor_replay`，以控制设备和共同训练预算；它不替换旧 seed 42 的 +2.1026 结果。每项报告实际参数量及训练设备。三种子均值同时保留逐种子结果。

旧补充队列已全部完成，以下继续命令仅用于其原登记作业恢复；同一作业只运行一个协调器，先核对 `queue_state.json` 和进程记录。队列会使用没有其他计算进程的空卡；全部显卡忙时最多运行一个 8 线程 CPU 读出任务，其余空卡出现后可继续并行。

```bash
python scripts/hanguard/nolora_ablation_queue.py run \
  --output outputs/hanguard_nolora_ablation_20261019
```

原数据整改发布位于 [三来源数据说明](../data/three_source_remediated_20261012/README.md)，122 条子集策略位于 [子集说明](../data/three_source_wildguard122_subset_20261018/README.md)。误按“片段”理解产生的 `three_source_fragment_clean_20261018` 与 `hanguard_fragment_clean_lora_20261018` 已停用，未用于本轮结果。

本次仓库整理增加导航和结果快照，不搬动已有实验目录或源码。旧协议绑定原文件路径与哈希，历史运行入口保留用于追溯；当前实验只使用 [脚本导航](../scripts/hanguard/README.md)列出的当前入口。部署模型与研究读出的训练范围不同，不能直接用研究表格替换旧推理包成绩。
