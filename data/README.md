# 数据版本导航

当前正式父数据为 **`three_source_remediated_20261012`**。20261022主消融使用与该发布绑定的固定划分和冻结副本；当前论文结果见[实验正文草稿](../docs/paper_experiments_20261025.md)，完整数据约定见[数据说明](../docs/data.md)。本导航不重新切分、筛选或修改任何发布文件。

## 当前正式发布

| 目录 | 内容 | 训练 / 验证 / 测试 |
|---|---|---:|
| [three_source_remediated_20261012](three_source_remediated_20261012/README.md) | 三来源完整二分类父数据：WildGuard中文、中文整理语料、JailBench | 66,802 / 8,350 / 8,350 |
| [hanguard_two_source_remediated_20261012](hanguard_two_source_remediated_20261012/README.md) | 同一父数据、同一划分的中文整理语料与JailBench已知类型记录，含安全类0 | 29,634 / 3,704 / 3,704 |
| 上述两来源视图中的已知有害类型 | `harmful`且已知类别1–5；用于独立风险五分类 | 17,440 / 2,180 / 2,180 |

二分类标签与类别监督独立：原二分类标签保留；未知或冲突类型用`category_id=-1`、`type_label_mask=False`表示，不能据类别字段反推有害性。两来源视图不是重新随机切分的独立数据集。安全类0用于二分类或门控输出，不参加五个风险类的交叉熵损失。

当前正式发布的关键证据：

- [三来源manifest](three_source_remediated_20261012/manifest.json)、[release QA](three_source_remediated_20261012/release_qa.json)与[分组划分规则](three_source_remediated_20261012/split_policy.json)。
- [两来源manifest](hanguard_two_source_remediated_20261012/manifest.json)与[release QA](hanguard_two_source_remediated_20261012/release_qa.json)。
- [五类语义与来源说明](hanguard_two_source_remediated_20261012/descriptions.json)：类别描述不是重新标注政策；当前CQA学习随机初始化的任务查询，不能因目录存在该文件就认定主方法使用描述文本编码。
- [主实验协议](../outputs/hanguard_clean_ablation_20261022/protocol.json)与[冻结数据副本](../outputs/hanguard_clean_ablation_20261022/data/)：实际使用的数据、版本和SHA以协议为准。`outputs/`中的完整实验文件仅本机保留，轻量证据见[主实验归档](../experiment_results/20261022_clean_ablation/README.md)。

三来源测试共8,350条，独立五分类测试2,180条，两来源门控评价3,704条。三种范围对应不同问题，不能互换分母或把独立五分类限定为二分类已判有害的样本。

## 历史版本与停用探索

| 目录 | 状态与用途 |
|---|---|
| [three_source_translation_repaired](three_source_translation_repaired/README.md) | 20260928翻译修复历史发布，训练/验证/测试62,155/7,753/7,778；其卡片保留当时说明，不是当前默认父数据 |
| [hanguard_two_source_primary_20260928](hanguard_two_source_primary_20260928/manifest.json) | 20260928两来源类型发布，关联旧类别实验与旧模型；不得与新划分混用 |
| [three_source_wildguard122_subset_20261018](three_source_wildguard122_subset_20261018/README.md) | 依赖旧模型错误的事后探索子集，仅保留选择依据与描述结果；不是当前主实验数据或独立质量修复发布 |
| [three_source_fragment_clean_20261018](three_source_fragment_clean_20261018/SUPERSEDED.json) | 已明确停用；保留原发布和停用原因，不能恢复为当前训练或评测输入 |

历史数据、实验与模型的身份关系保留原样。新划分与旧训练身份可能相交，不能用旧检查点或旧特征替代当前主实验中的重新提取与训练。

## 来源资料与本地大文件

[sources/](sources/)保存原始输入与来源追溯资料；[sources/reference/](sources/reference/)保存原有中文表格及类别定义参考。它们不是默认训练集。更早的原始三集和构建记录见[历史归档索引](../archive/README.md)。

Git保留数据卡、manifest、QA与必要定义；完整文本语料、parquet、逐条审计表和本地来源档案按仓库忽略规则留在本机。没有完整本地数据时可查已归档实验摘要，不能据摘要编造新的逐样本指标。

已有manifest、QA和实验协议绑定文件内容及路径，维护时不要覆盖已冻结发布。确需变更时建立新版本、新实验输出目录并重新审计，再在[实验索引](../docs/experiment_index.md)登记。
