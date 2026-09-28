# hanguard 中文翻译修复数据

当前状态：**qa_complete_with_documented_limitations**。见 [release_qa.json](release_qa.json)：全量结构、来源、规则核验及固定72条助手双语复核已完成；不代表全量真人语义审校。

构建时的 manifest/audit 保留 `candidate_pending_root_qa` 原始状态；后续完成的 QA 以 `release_qa.json` 为准，且绑定 manifest 与独立核验的 SHA256。当前统一训练入口已使用本发布，不修改冻结的构建记录。

全部 WildGuard 原有三集记录按统一规则重新翻译，其他来源保留原文本。原 split/base_id/group_id/标签不变；base_id 表示历史归属，不再要求等于新文本哈希。

原始 80709 条；完整档案 80709 条；候选三集 77686 条；明确隔离 3023 条。不设统一token长度上限；通过质量/重复检查的长文本完整保留。不截断，不回退旧译文。

prompt_over_legacy_limit仅标记是否超过旧实验370 tokens，长度不是翻译质量判据。完整档案超旧限582条，候选三集保留超旧限407条；长度分布及各来源/划分明细见manifest.length_statistics。

full_repaired_archive.parquet 保存所有记录及完整新旧文本、英文和修订理由；quarantine.parquet/.csv 保存所有隔离记录。normalized_prompt/sample_id/token数已重算。

修复中文的跨集重复、同集重复及标签冲突均整组隔离，明细见collision_groups.json。恢复英文同源后发现的跨集精确重复或标签冲突也整组隔离，不依据模型错误选删。明细另见english_collision_groups.json及english_cross_split_review.csv。

model_translation/model_translation_sha256保留模型原译；translation_source/correction_source及reviewer区分模型输出与有记录的双语复核修订，完整原/新审计保存在translation_metadata_json。Codex助手复核不是人工专家金标，保留原标签也不代表标签已验证；精确去重不保证语义近重复隔离。

当前统一训练入口为 `scripts/hanguard/train.py`，新实验须在新输出目录注册并按数据／骨干身份重新建立缓存。训练上限为 4,096 token，不静默截断主数据；旧 370-token 配置与旧缓存已经退出活动入口。译文和隔离后的测试样本发生过变化，新分数不能直接减去旧数据分数并宣称算法增益。参见[训练说明](../../docs/training.md)。


训练 62,155、验证 7,753、测试 7,778 条；407条超过旧370限制的合格全文已保留，最长3,202 token。

完整结果见 [修复报告](../../outputs/hanguard_translation_repair_20260928/report.md)。
