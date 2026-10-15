# 整改数据训练契约

二分类标签与类型监督已分离。新增 `remediated_dataset_contract.py` 和 `remediated_binary_study.py`；维护入口 `current_training.py` 在注册新 release 时检查新契约，明确注册新契约的 binary worker 使用新 wrapper。原 `repaired_study.py` 保持原字节内容，旧协议不自动切换 worker。

| category_id | prompt_harm_label | type_label_mask | 用途 |
|---|---|---|---|
| 0 | unharmful | False | 二分类安全监督 |
| 1..5 | harmful | True | 二分类有害监督和已知类型监督 |
| -1 | harmful 或 unharmful | False | 保留原二分类监督，不提供类型监督 |

新 manifest 使用 `dataset_contract="hanguard_binary_type_decoupled_v1"` 和明确的 `dataset_version`。每行必须有 `type_label_mask`，且它严格等于 `category_id > 0`。mask 缺失、未知类型误设 mask、已知类型与 binary 矛盾均拒绝；二分类标签只接受原来源的 harmful/unharmful，不从 category 重造标签。`category_annotation_status` 可由 builder 提供 retained/ambiguous_type 作为溯源，不参与 BCE。旧 release 的已知 0..5 分类可继续不提供 mask。

新 release 在 completed QA 写出之前不能注册。`release_qa.json` 必须包含完成状态 `qa_complete_with_documented_limitations`、`manifest_sha256`（兼容旧字段 `dataset_manifest_sha256`）以及覆盖三个 split 的 `split_sha256`。manifest、split 哈希都逐文件验证；双 manifest hash 字段同时存在时必须都一致。训练协议保存 dataset_contract、dataset_version、split hashes 和新模块源代码快照。

同一 group 可包含安全/有害对照；group 只承担跨 split 隔离。相同 canonical 全文具有冲突 binary 标签时拒绝，跨 split 的 ID、group、normalized 全文重叠仍拒绝。输入长度全量核查，超过注册 max_tokens 拒绝，不截断。原 primary-category 阶段继续只读已知 0..5，遇到 -1 明确拒绝，不将未知类型当作安全标签。

类型阶段另加父数据版本约束：新 category 子集声明 `parent_dataset_version` 后，必须与 binary parent 注册的 dataset_version、data_dir 和已验收 manifest 完全对应。每个子集 base_id 必须在父数据同一 split，prompt/source/binary/category 逐字段一致。父三集、manifest、release QA 哈希写入子协议依赖；拒绝新类型子集配旧父、旧类型数据配新父、跨 split 移动和文本/标签漂移。一般 group 可以含不同类别，guard 不以 group 是否单标签作条件。历史显式旧数据配旧父仍保留。

验证：`tests/test_remediated_dataset_contract.py`、`tests/test_current_training.py`、`tests/test_repaired_study.py` 共 70 个 CPU 测试通过，包括真实注册新父/子协议后输入漂移拒绝的集成验证。只读加载旧 frozen 和 dynamic protocol 均通过；原 repaired_study.py SHA-256 仍为 `ca86fc00cb43afb1d95d76d003a3733f355e6eae753e4916c2f006380712e249`，与锁定 frozen protocol 一致。未改任何旧实验 protocol/cache，也未启动训练。

新默认 binary 与 category 数据路由由 root 在新 release 独立 QA 完成之后一起激活；本次契约修改保留旧默认路径，现有 --data 可显式传入已验收新 release。新 category 需要重新训练并选择同版本 binary parent，不能沿用旧 LoRA、checkpoint 或阈值。重建 split、修复文本或新增覆盖属于新数据版本，后续须注册新训练协议并重建缓存，不能把前后数据版本分数差异归因于算法。
