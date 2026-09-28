# 当前数据与复现边界

训练入口使用两个正式发布目录，不读取 `archive`、完整修复档案或隔离集。

| 路径 | 内容 | 训练／验证／测试 |
|---|---|---|
| `data/three_source_translation_repaired` | WildGuard 中文修复、中文整理、JailBench | 62,155／7,753／7,778 |
| `data/hanguard_two_source_primary_20260928` | 后两个来源，原文、原标签与原划分不变 | 27,618／3,451／3,452 |
| 两来源有害子集 | 只对 category_id 1–5 计算五分类损失 | 15,578／1,875／1,828 |

二分类监督使用 `prompt_harm_label`。类型监督继承 `category_id`，不采用已停用的多标签试标或 WildGuard 意图重标草案。类型评估限两来源有害子集；安全记录保留用于最终门控输出和二分类评估。分类标准及来源差异见 [标准核查](primary_category_standard_review.md)。

## 保留的来源与质量记录

- `data/sources/`：原始输入资料，用于来源追溯，不直接作为默认训练入口。
- `data/sources/reference/`：原有中文语料表格和类别定义文档，原字节保留，仅从 `data/` 根目录移入来源资料目录。
- 修复数据的 `manifest.json`、`audit.json`、`release_qa.json`：冻结的发布、检查和最终 QA 记录。
- `full_repaired_archive.parquet`：全部 80,709 条历史范围记录及原文、新旧译文、修订依据。
- `quarantine.parquet`／CSV：3,023 条因质量疑点或重复／标签冲突而隔离的记录。
- `outputs/hanguard_translation_repair_20260928/`：原文映射、生成协议、双语复核、源代码快照和独立读回检查。
- 两来源发布的 `descriptions.json`：实验前冻结的类别描述，用于描述查询对照，不是重标规则。

修复主集 77,686 条；407 条超过旧 370-token 限制的合格全文保留，最长 3,202 token。这里的全文与无长度筛除指本次对既有 80,709 条记录的修复发布；最初整理语料时存在历史长度筛选。

72 条固定样本由助手做双语复核，另外保留问题回归与修订记录。这不是全量人工语义审校。精确重复和已知组关系隔离不保证所有未知语义改写或攻击模板互不重叠；既有测试身份已用于探索。

## 重新导出两来源数据

使用当前固定三来源发布重建一个新目录：

```bash
python scripts/hanguard/prepare_two_source_primary.py --help
```

按该工具的 `--source`、`--output` 参数指定输入和新目录；它保留来源原字段、文本、标签和划分，不重新随机切分。新分类实验应使用新输出目录登记协议，不能覆盖已完成研究的哈希记录。

## 历史复现资料

停用的数据已移出活动目录。修复前的三份 split 及必要元数据保存在 `archive/data_before_repair_20260928.tar.gz`，用于核对旧 manifest 中的输入哈希；旧实验摘要与代码另行压缩归档，见 [归档索引](../archive/README.md)。

默认后续训练直接使用已审计的正式发布，不需要恢复旧数据。若确需重放历史翻译构建，先在项目内单独的复现目录检查归档成员，再恢复历史输入与源代码快照，并使用翻译记录里的显式路径参数。历史 JSON 中的绝对路径和代码哈希是原执行证据，清理不会改写它们来冒充新流程。最早来源恢复的本地辅助文件见清理清单中的 `archive/source_provenance`；也可从已冻结的 `source_mapping.parquet` 重放后续翻译步骤。

来源谱系的补充诊断输入另记于 `archive/source_provenance/diagnostics_manifest.json`。若重新运行旧 token 风险诊断，还需要匹配的 NLLB tokenizer，并通过 `translation_provenance.py --tokenizer` 显式指定；该历史依赖不参与当前模型训练、推理或基于冻结映射的翻译修复重建。
