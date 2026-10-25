# 实验结果与证据导航

当前论文主线为**类别查询注意力聚合（CQA）＋固定多层残差融合（MLF）**。写作先读[论文实验主稿](../docs/paper_experiments_20261025.md)和[论文结果包](20261025_paper_ready/README.md)，再查对应原始证据。下表状态根据2026-10-25已有归档核实；完成状态不表示调度器仍在运行。

## 当前写作与复现入口

| 优先级 | 结果包 | 状态 | 用途 |
|---|---|---|---|
| 1 | [20261025_paper_ready](20261025_paper_ready/README.md) | 已生成并校验 | 论文四组主消融、原生Guard比较的CSV/LaTeX；逐种子、逐类别和补充对照索引 |
| 2 | [20261022_clean_ablation](20261022_clean_ablation/README.md) | `complete` | 当前核心实验：两个任务各9臂×3种子，共54个正式分类头，192个全层probe；含协议、选模记录和预测 |
| 3 | [20261025_qwen3guard_clean_test_v2](20261025_qwen3guard_clean_test_v2/README.md) | `complete`，8,350条安全标签有效率100% | **当前原生Qwen3Guard外部基线**，按官方接口直接生成，无本地监督分类头 |
| 4 | [20261025_paper_followups](20261025_paper_followups/README.md) | `complete` | 两种通用聚合对照共12头、MacBERT监督五分类3种子、错误分析与在线效率；用于补充材料及归因检查 |

主四组A/B/C/D依次对应基础MLP、MLP＋MLF、CQA、CQA＋MLF。准确率及标准差以[主表](20261025_paper_ready/tables/main_ablation.csv)为准，逐种子结果见[seed_results.csv](20261025_paper_ready/tables/seed_results.csv)。组合相对MLP的提升与在CQA上增加MLF的增量须分开，见[配对增益表](20261025_paper_ready/tables/paired_deltas.csv)，不能把组合总收益全部称为融合收益。

测试范围分别为安全二分类 **8,350** 条、独立风险五分类 **2,180** 条。完整双阶段系统的 **3,704** 条是另一固定评价范围，不替代上述任一任务的分母。主表均值来自seed 42/43/44，不从中挑最高种子。

原生Guard保留推理前固定的两种安全等级合并规则：仅Unsafe为有害、Unsafe与Controversial均为有害。两种规则均进入[原生比较表](20261025_paper_ready/tables/native_guard_binary.csv)；其原生类别不与本地五类直接对应，五分类为N/A。外部模型原生推理与本地监督分类器的比较边界见[原始能力说明](20261025_qwen3guard_clean_test_v2/ORIGINAL_CAPABILITY_COMPARISON.md)。Encoder-Guard当前搁置，本目录没有其对比成绩。

## 补充适配与历史记录

这些目录保留研究过程和失败原因，不用于替换当前主表；目录中的历史`pending`状态不表示应恢复训练。

| 结果包 | 已核实状态 | 保留用途 |
|---|---|---|
| [20261025_qwen3guard_supervised_comparison_v2](20261025_qwen3guard_supervised_comparison_v2/README.md) | `complete`，18/18个分类头 | Guard骨干上的**额外本地监督适配**；不是原版能力，不作为当前原生外部基线，见[用途纠正](20261025_qwen3guard_supervised_comparison_v2/EVALUATION_SCOPE.md) |
| [20261025_qwen3guard_supervised_comparison](20261025_qwen3guard_supervised_comparison/runtime_failure_summary.json) | `failed`；有效特征收据0，无测试结果 | BF16批量形状校准未通过，随后停止并由v2替代；保留失败记录和旧协议 |
| [20261025_qwen3guard_clean_test](20261025_qwen3guard_clean_test/README.md) | 已被替代；完成分块0，无有效成绩 | 首批生成在结果序列化时失败；修复后在独立v2目录重跑，见[superseded.json](20261025_qwen3guard_clean_test/superseded.json) |
| [20261019_four_arm_experiments](20261019_four_arm_experiments/README.md) | 归档状态`pending`，任务已停止 | 旧四臂实验的缓存一致性排查；没有三种子最终主表，当前使用20261022重跑 |
| [20261019_fusion_ablation](20261019_fusion_ablation/README.md) | 归档内5组研究均`complete` | 旧融合、组件、种子和参考分析；层结构与评价范围有别，不混入当前固定等权消融 |
| [20261015](20261015/README.md) | 历史快照，包含完成、部分完成及未完成项目 | 数据审计、翻译、早期训练和诊断追溯；逐项状态见[index.json](20261015/index.json)，不能将快照中的计划视为完成结果 |

## 这里保存什么

`experiment_results/`保存可随代码管理的结果证据：协议与源码快照、选择记录、结果JSON、数值预测、表格、失败记录和SHA256清单。其组织是按实验日期和用途归档，不移动已登记路径，也不修改原始协议以匹配后续代码。

`outputs/`保存本机完整运行目录：权重、token特征、完整输入/原生输出、日志和调度状态；`models/`保存模型文件，完整语料位于`data/`下相应发布目录。上述大产物由Git忽略，轻量归档不能代替它们重放训练。新环境缺少本机产物时，可以查归档指标和已有预测，不能据此声称已重新训练或验证了全部原始数据。

现有数值Parquet只含ID、来源、标签、长度、概率或预测，不含完整输入和生成文本。历史审计另保留少量文本举例，协议也可能记录本机绝对路径和进程身份；因此不将整个目录描述为“完全无文本或环境信息”。当前目录没有模型权重或大型特征缓存；较大的CSV是全层probe及分类头数值预测，不按扩展名直接删除。

## 校验与刷新

以下命令从仓库根目录执行，只做CPU证据核对，不启动训练或GPU推理：

```bash
# 当前论文表格：核验来源、重新计算指标，并核对写作稿和表格SHA。
.venv-hanguard/bin/python scripts/hanguard/package_paper_experiment_materials.py --verify

# 核心实验归档：核验现有清单及数值证据。
.venv-hanguard/bin/python scripts/hanguard/package_clean_ablation_results.py --verify

# 两个20261019历史归档。
.venv-hanguard/bin/python scripts/hanguard/package_four_arm_experiments.py --verify
.venv-hanguard/bin/python scripts/hanguard/package_fusion_ablation_results.py --verify

# 20261015快照：仅依赖Python标准库，可直接核验归档。
python experiment_results/20261015/verify_archive.py
```

当前论文表格与核心打包工具省略`--verify`时会**刷新派生归档**；Guard的`summarize`及补充实验的`archive`也会重写相应导出。日常检查使用只读校验，科学来源仍以各包的协议、结果和清单为准。不同年代的清单结构不同，不把它们当作同一实验协议。

更多入口见[实验索引](../docs/experiment_index.md)、[脚本导航](../scripts/hanguard/README.md)和[补充证据说明](../docs/paper_supporting_evidence_20261025.md)。
