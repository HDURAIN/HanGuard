# hanguard 论文实验材料（20261025）

主线为类别查询注意力聚合（CQA）＋固定多层残差融合（MLF），二分类与单标签五分类分别进行四组消融。

- [实验正文草稿](../../docs/paper_experiments_20261025.md)：方法命名、复现设置、四组主表与原生 Qwen3Guard 对照。
- [补充证据与写作边界](../../docs/paper_supporting_evidence_20261025.md)：参数匹配、选层、替代聚合、逐类分析与效率限制。
- [主消融 CSV](tables/main_ablation.csv)／[LaTeX](tables/main_ablation.tex)：二分类 F1 为有害类 F1，五分类采用 Macro-F1。
- [逐 seed 结果](tables/seed_results.csv)／[增益表](tables/paired_deltas.csv)／[增益 LaTeX](tables/paired_deltas.tex)。
- [原生 Qwen3Guard 二分类 CSV](tables/native_guard_binary.csv)／[LaTeX](tables/native_guard_binary.tex)：两种预先固定的原生安全标签合并规则同时保留；原生五分类 N/A。
- [五类 P/R/F1](tables/five_class_metrics.csv)、[全部机制对照](tables/supplementary_controls.csv)及 supplementary/ 中已有配对区间、来源/长度分层与在线效率记录。

表格均由固定结果中的三个 seed 重新计算；百分数与百分点分开。LaTeX 表使用 booktabs。
完整组合相对普通末位置 MLP 的准确率增益为二分类 +2.2675 pp、五分类 +5.8563 pp；这不是融合单模块的贡献。
组合相对仅 CQA 的平均增益为 +0.2555 pp、+0.9327 pp，存在种子差异。

此包只组织已完成证据，不重新训练、重新预测或修改数据。外部基线使用原始发布模型；额外监督适配结果留在原有归档，不混入原生比较。Encoder-Guard 对比保持搁置。
evidence.json 保存来源 SHA；manifest.json 校验本包与当前写作稿。指标是现有固定划分上的结果，不宣称整个项目从未查看测试集。

```bash
.venv-hanguard/bin/python scripts/hanguard/package_paper_experiment_materials.py
.venv-hanguard/bin/python scripts/hanguard/package_paper_experiment_materials.py --verify
```
