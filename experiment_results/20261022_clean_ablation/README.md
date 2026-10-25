# 全新缓存与训练的消融实验（20261022）

当前状态：**complete**。协议登记：**True**；注册源码核验：**passed**。

本轮只读取正式数据发布及原始 Qwen3.5-4B 基座，从原文重新分词，重新提取特征、训练、选层、选择检查点和阈值。历史实验的特征、预测、选择记录、权重、结果和测试排除名单均不作为输入；不沿用旧 122 条排除名单。

| 组别 | 方法 |
|---|---|
| `baseline` | L32 末 token MLP |
| `fusion_only` | 固定 L8/L16/L24/L32 各 0.25 的融合 |
| `category_only` | L32 全文 token 的类别查询读出 |
| `fusion_category` | 类别查询读出＋固定多层融合 |
| `fusion_repeat_last` | 融合结构保留，四路均输入 L32 |
| `category_fusion_repeat_last` | 组合结构保留，融合四路均输入 L32 |
| `category_shared` | 类别间共享 token 注意力分布 |
| `fusion_category_shared` | 共享读出＋固定多层融合 |
| `probe_selected_single` | 从新训练的 32 层线性 probe 通过验证集选层，再训练单层 MLP |

二分类、五分类各 9 组，seeds 42/43/44，共 **54 个正式分类头**；全部层 probe 共 **192 个**。骨干冻结，无 LoRA；全文 token 与各层句向量在同一次前向生成，所有组共用该轮新缓存。

测试范围分别为：二分类完整 **8,350** 条、原生已知有害类型 **2,180** 条、完整系统已知最终标签 **3,704** 条。两项监督任务与派生系统使用各自固定分母，未知类型不重新标注。测试在全体检查点及阈值由验证集锁定后开放。

完整系统报告三个组合：两阶段对应 A/A、B/B、C/C、D/D；经典组合 A/A、B/A、A/C、B/C；类型阶段也融合的原组合 A/A、B/B、A/C、B/D。每对字母分别表示二分类头／类型头。系统结果由同一批新预测派生，不增加训练。

三种子全部完成才显示均值及样本标准差；未完成项保留 `pending`。不将旧实验成绩填入本表。

## binary

| 方法 | 状态 | Accuracy / %（均值 ± SD） | Macro-F1 / %（均值 ± SD） |
|---|---|---:|---:|
| `baseline` | complete | 93.8483 ± 0.0997 | pending |
| `category_fusion_repeat_last` | complete | 95.8363 ± 0.2603 | pending |
| `category_only` | complete | 95.8603 ± 0.1540 | pending |
| `category_shared` | complete | 95.8483 ± 0.2732 | pending |
| `fusion_category` | complete | 96.1158 ± 0.2539 | pending |
| `fusion_category_shared` | complete | 96.1717 ± 0.1720 | pending |
| `fusion_only` | complete | 94.9142 ± 0.0421 | pending |
| `fusion_repeat_last` | complete | 93.7924 ± 0.2354 | pending |
| `probe_selected_single` | complete | 94.7146 ± 0.0566 | pending |

## category

| 方法 | 状态 | Accuracy / %（均值 ± SD） | Macro-F1 / %（均值 ± SD） |
|---|---|---:|---:|
| `baseline` | complete | 77.2783 ± 0.9516 | 76.0952 ± 0.8485 |
| `category_fusion_repeat_last` | complete | 82.4312 ± 0.0917 | 81.0182 ± 0.2595 |
| `category_only` | complete | 82.2018 ± 1.5950 | 80.9201 ± 1.8251 |
| `category_shared` | complete | 81.8196 ± 0.5473 | 80.5229 ± 0.7289 |
| `fusion_category` | complete | 83.1346 ± 0.8671 | 81.7905 ± 1.1373 |
| `fusion_category_shared` | complete | 82.7064 ± 0.8037 | 81.5472 ± 0.9038 |
| `fusion_only` | complete | 77.8899 ± 1.6225 | 76.3739 ± 1.6472 |
| `fusion_repeat_last` | complete | 77.3242 ± 0.9691 | 76.1503 ± 0.8433 |
| `probe_selected_single` | complete | 77.2783 ± 0.9516 | 76.0952 ± 0.8485 |

## corresponding_both_stages

| 方法 | 状态 | Accuracy / %（均值 ± SD） | Macro-F1 / %（均值 ± SD） |
|---|---|---:|---:|
| `baseline` | complete | 86.3661 ± 0.6280 | 79.8667 ± 0.7588 |
| `category_only` | complete | 89.1289 ± 0.9061 | 83.7704 ± 1.4886 |
| `fusion_category` | complete | 89.8398 ± 0.5017 | 84.6038 ± 0.9220 |
| `fusion_only` | complete | 86.6991 ± 0.9675 | 80.0761 ± 1.4030 |

## classic_gate_type_factorial

| 方法 | 状态 | Accuracy / %（均值 ± SD） | Macro-F1 / %（均值 ± SD） |
|---|---|---:|---:|
| `baseline` | complete | 86.3661 ± 0.6280 | 79.8667 ± 0.7588 |
| `category_only` | complete | 89.2459 ± 0.9750 | 83.8573 ± 1.5215 |
| `fusion_category` | complete | 89.2459 ± 0.8547 | 83.8773 ± 1.4522 |
| `fusion_only` | complete | 86.3301 ± 0.5010 | 79.8180 ± 0.6924 |

## fusion_in_both_stages_factorial

| 方法 | 状态 | Accuracy / %（均值 ± SD） | Macro-F1 / %（均值 ± SD） |
|---|---|---:|---:|
| `baseline` | complete | 86.3661 ± 0.6280 | 79.8667 ± 0.7588 |
| `category_only` | complete | 89.2459 ± 0.9750 | 83.8573 ± 1.5215 |
| `fusion_category` | complete | 89.7858 ± 0.4922 | 84.5843 ± 0.9170 |
| `fusion_only` | complete | 86.6991 ± 0.9675 | 80.0761 ± 1.4030 |

逐种子结果见 [results_summary.csv](results_summary.csv)，三种子汇总见 [seed_summary.csv](seed_summary.csv)，完成状态见 [coverage.csv](coverage.csv)。[index.json](index.json) 保存运行状态快照和源码核验；[manifest.json](manifest.json) 记录每项源文件及归档 SHA256。

`artifacts/` 只保存新研究的协议、状态、小型选择和结果 JSON、注册 Python 源码以及无原文的数值预测 CSV。不包含原文、特征、模型权重和日志。

```bash
python scripts/hanguard/package_clean_ablation_results.py
python scripts/hanguard/package_clean_ablation_results.py --verify
```
