# WildGuard / category3 定向错误、标签与后续实验审计

本报告只分析已经探索过的内部 test、登记本地可用数据与检查实验可执行性。内部 test 上的结果都是诊断性证据，不能重新包装成盲测结论。外部 holdout 本轮只生成索引和锁文件，没有运行模型预测。

## 审计结论

现有证据同时支持两个原因，而不是在“数据问题”和“训练参数问题”之间二选一：

1. **WildGuard hardest errors 中存在实质性的继承标签和政策边界噪声。** 20 个相关模型变体全部判错的 104 行中有 100 行来自 WildGuard；按 4 个模型族计，156 个共错中有 149 行来自 WildGuard。对优先级最高的 40 个 WildGuard 共错做单人初筛，11 个标为 `current_label_likely_wrong`，17 个标为 `ambiguous_policy_boundary`，12 个标为 `current_label_plausible`。这个样本由模型错误定向选取，只能说明噪声确实存在，不能估计全库标签错误率。
2. **翻译不是这批共错的主要解释。** 4,326 个 WildGuard test 行均能恢复到准确的英文父样本和原始二元标签；149 个 WildGuard 全模型族共错中，记录过翻译修正的为 0，超过旧 token 限制的只有 1。40 个初筛样本也都没有发现明显、足以改变标签的翻译错误。
3. **category3 的表面弱点主要由来源构成驱动。** category3 共 468 行：WildGuard 248、JailBench 112、Chinese curated 108。全部 13 个已汇总模型在 Chinese curated 的 108 行均为 100%；JailBench 多数只错 0–4 行，个别 probe/stack 错 6–7 行；WildGuard 则错 9–18 行。9 个 category3 全模型族共错全部来自 WildGuard。
4. **WildGuard 的 category3 不是可用的独立类别金标。** test 中 248 个当前 category3 行只有 177 个原始 subcategory 能套用历史固定规则；其中仅 81 个规则也映射到 3，96 个按规则会映射到别类，另有 71 个是无固定规则的 `others`。全部已接收 WildGuard 当前 category3 共 2,220 行：615 个 `others`、700 个与历史 category3 规则相符、905 个按该规则映射到别类。规则本身也不是人工复核金标，所以这里只能判定类别来源不完整、无法复现，不能反推哪一个 category_id 一定正确。二元 harmfulness 标签已独立核对，不因这个类别来源问题自动失效。
5. **训练和融合不稳定仍是独立问题。** 在同一 frozen E03 LoRA 上，三个 head seed 的 category3 错误数为：repeat-L32 `10/20/9`，fixed fusion `14/22/13`，selected fusion `13/20/16`。多层融合没有在该切片形成稳定增益，且常增加错误。全 test 上各 head seed 的排序也会翻转。这说明标签噪声不能解释全部现象；小样本切片、层间高度相关、融合器有效自由度、校准与 seed 方差仍需通过严格配对实验分开验证。

完整机器可读结果见 `summary.json`、`slice_model_metrics.csv`、`wildguard_subcategory_errors.csv` 和 `wildguard_subcategory_by_current_category.csv`。

## 标签复核方案

现有 `review_form_blinded.csv` 含 159 行，审阅者看不到继承标签、模型输出和入选原因；`review_key_with_predictions.parquet` 仅供复核完成后解盲。抽样由模型定向案例和固定 hash controls 组成。

执行时应满足以下验收条件：

1. 两名审阅者独立判断 `translation_fidelity`、binary harmfulness、primary category 与置信度，不查看 key 或模型结果。
2. 先冻结文字版政策边界和判定示例，再开始标注；成人性信息、抽象高危知识、危险自助医疗、通用 jailbreak 和比喻性语言要作为预先定义的边界类型记录。
3. 对所有分歧做第三方裁决，分别报告原始一致率和 Cohen's kappa；不能用模型多数票替代人工裁决。
4. 只有固定 hash controls 可用于其明确抽样框内的标签噪声率估计。模型定向案例只报告错误类型和案例数，不外推到全库。
5. 修正或查看这些标签后，当前 test 即成为开发数据；修正后的分数只能用于诊断，不能作为新盲测结果。

`preliminary_consensus_error_spotcheck.csv` 是单人初筛，作用是验证正式双人审阅是否值得执行，不是最终标签裁决。

## 同一 frozen LoRA 与多 LoRA seed

仓库原有 6 个 binary LoRA adapter 全部是 seed 42；E03、E04、E13–E16 的 mode/objective 不同，不能当作 6 个独立 LoRA seed。frozen E03 诊断中的 head seed 42/43/44 也只代表相同 adapter 上的 head 初始化，不能回答 LoRA seed 方差。

新实验 `outputs/hanguard_e03_lora_multiseed_20261012` 已注册为 fresh E03 seeds 42/43/44，并禁止所有 seed 完成选择前访问 test。审计时 seed 42 正在训练；42 也重训是必要的，因为当前代码与旧研究的代码哈希不同。原有完整 run 的中位总耗时约 2.428 GPU 小时；每个 adapter 的 32 层 feature cache 约 12.73 GB。当前约有 851 GB 可用，但文件系统已使用约 95%，应在生成每个缓存前检查剩余空间。

多 seed 结论的最低验收标准：

1. 每个 LoRA seed 独立训练；其后在该 seed 内冻结同一 adapter，repeat-L32 与多层 head 共用完全相同的 train/validation/test 行、优化预算、阈值规则和随机 seed 集合。
2. 容量对照同时报告名义参数数、输入特征矩阵秩与实际可训练自由度。若 repeat 分支使用相同输入和完全相同初始化，其输出会保持相同，只能称为“参数拓扑对照”，不能声称有效容量完全相同。
3. 层选择、正则、校准和阈值只使用 train OOF 与注册的 validation gate。若 stacked OOF 用于选择，outer heldout group 不能通过 inner base-logit 训练或 calibration score 回流，必须采用严格 nested group CV。
4. 每个 seed 先锁定 checkpoint、层、阈值和 validation 决策，再一次性访问 test。主报告给出 matched-seed delta 的均值、标准差、每 seed 方向和 group cluster CI；三个 seed 只描述稳定性，不作强总体显著性宣称。

## 外部 holdout 登记

不可变登记位于 `outputs/hanguard_external_holdouts_20261012`，状态为 `registered_unconsumed`，`prediction_artifacts` 为空。索引不复制 prompt；后续 evaluator 必须按 `source_file + source_row` 回取并复算 exact prompt hash、规范化 hash、`group_id` 和 `base_id`。规范化固定为 Unicode NFKC、casefold、删除 Unicode Cf、删除全部空白，不做语义匹配。

### Primary：official WildGuard English test

- 源文件 1,725 行；排除 26 个无 binary label 的行和 1 个与现有 archive 规范化重叠的行，留下 **1,698 行 / 1,698 groups**。
- 标签为 754 harmful、944 unharmful；源、索引、选择规则、顺序 ID 和实现脚本均已记录 SHA256。
- 只读 tokenizer 预检的 token 数为 min/median/p95/max = 6/45/276/630，0 行超过已注册的 4,096 token 上限；该预检没有运行模型 forward，也没有消费 registry。
- 这是当前仓库最好的未预测 external binary candidate，但它是英文，而现有主 test 是中文。直接评估混合了语言与领域迁移；若翻译，必须在看分数前锁定单一、label-blind 的翻译流程，不能根据 holdout 结果挑 direct 或 translated 版本。
- 标签和文件在仓库中可见，因此它是 untouched external holdout，不是严格隐藏盲测；也没有可用于五类评估的独立 category gold。

### Secondary：injection_defense stress set

- 18,230 行、**18,175 个规范化 groups**，55 个重复行，无冲突标签 group，与当前已接收 splits 的完整规范化 prompt 重叠为 0。
- 只读 tokenizer 预检的 token 数为 min/median/p95/max = 9/49/118/726，0 行超过 4,096；没有运行模型 forward。
- 只作为 adversarial stress set；生成或继承标签并非独立人工金标，内容由早期 HanGuard 样本和模板变换而来，零全文重叠不代表语义独立。
- 统计和重采样单位必须是 `group_id`，并与 official WildGuard 的结果分开呈现。

任何预测前必须冻结 adapter、head、seed 选择、层、校准、阈值和语言处理方案并写入 hash lock；首次预测后把 registry 标记为 consumed，不能再用这两个集合调参。

## 其他本地数据可用项与风险

| 数据 | 可用量 | 适用性 | 主要风险 |
|---|---:|---|---|
| `wildguard_en/train.parquet` 未入 archive 部分 | 6,319 行 / 2,194 规范化 groups | 额外同源诊断 | 16 个 group 内有冲突标签；与既有 WildGuard 同分布，不如 official test 独立 |
| repaired quarantine | 3,023 行，3,022 行来自 WildGuard；2,369 groups | 数据质量/冲突诊断 | 因质量规则被定向隔离，不能作随机 holdout |
| `cat5_generated.parquet` | 1,883 行 / groups | 类别覆盖与压力测试 | 合成数据、标签来源非独立 |
| `wildguard_retrans.parquet` | 5,037 行，5,022 groups；5,036 行与当前 split 全文不重叠 | 翻译敏感性诊断 | WildGuard 派生/生成，非新语义来源 |
| `hanguard_v3.parquet` | 29,065 行中 1,038 行全文规范化未见 | 补充错误分析 | 来源已大量进入现有数据，1,038 行也不构成独立来源 |

本地不存在严格隐藏的盲测。若需要可发表的独立最终结论，应由数据保管方持有未公开 labels，或新采集一批 prompt，经双人独立标注和第三方裁决后，仅在模型与分析计划冻结后开放一次。

## 证据边界

- 20 个模型变体高度相关，“全模型一致”只是审阅优先级证据，不是 20 个独立投票。
- 32 层矩阵、逐来源切片和所有配对 p 值均是探索性结果，未做多重检验校正，不能根据单个 p 值确认某一层更优。
- 当前证据能排除“只是翻译坏了”这一单一解释，也能排除“category3 在所有来源都学不会”这一单一解释；它不能仅靠 posthoc test 分析量化标签噪声和训练不稳定各自贡献多少。
