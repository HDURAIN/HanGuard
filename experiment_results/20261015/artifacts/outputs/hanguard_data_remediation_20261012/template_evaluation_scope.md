# 新版本的划分与模板评测范围

主划分使用 `group_split_policy.py` 的 `assign_group_splits`。它保留原始二分类标签和全部输入行，重新建立家族组，再按来源 × 二分类的联合计数接近 80% / 10% / 10% 分配 train、validation、test；随机种子固定为 20261012。旧 split 留作溯源，不能因为旧划分存在风格重复而删除整组数据。新划分会改变训练和评测样本，因此新旧版本分数不能直接当作算法改进量。

来源 × 类型（含类型未确定的 `-1`）另作为二级平衡目标。它只交换完整来源 × 二分类向量完全相同的组，因此各来源二分类计数和总行数严格不变。固定 seed 的类别 profile 和组哈希决定交换次序，不读取模型分数、不改标签。整组约束和非交换组可能留下少量类型配额残差，最终 `split_policy.json` 同时报告目标、实际计数和条件类型分布漂移。

同一历史父组、同一完整官方原文、同一 JailBench 父 seed，以及已确认的中文标点风格变体进入同一划分家族。原文近相似审计的 228 对也保守地连接成家族，以减少相同写作模式跨集的机会。这里的家族关系并不宣称请求语义相同；不同 payload 和不同标签均保留，不能把它们删成“重复样本”。中文标点规则遇到二分类不同会记录 flags，不自动合并或删除。

JailBench 主评测范围是 **unseen seed、seen wrapper**：同一来源父 seed 的所有 wrapper 留在一个主 split，全部 20 类 wrapper 可出现在各主 split。主划分没有根据 wrapper 把不同父 seed 连在一起，因此不会把 seed × wrapper 的整个图变成一个巨组。对其他来源，主划分测量的是所登记原文与风格家族之间的泛化；它不能证明所有潜在语义或写作模式都隔离。

另登记一个独立的纯 JailBench **template holdout view**。20 个 wrapper 按固定 seed 和 wrapper 内容哈希排序，整类分为 train 12 类、validation 4 类、test 4 类。`build_template_holdout_manifest` 返回固定 wrapper 归属和逐行 view mapping；这些归属不覆盖主 split。它的范围是 **已见公开 seed payload、新 wrapper**，有意复用公开父 seed，不声称同时测量新 seed 与新 wrapper。主协议和这个 view 应分别训练、选择和评测 checkpoint；在主训练已见过某个 wrapper 的 checkpoint，不能作为该 view 的“未见 wrapper”结果。

固定归属登记在 [template_holdout_protocol.json](template_holdout_protocol.json)。train 为 J03、J04、J05、J06、J07、J08、J11、J13、J14、J15、J17、J19；validation 为 J01、J02、J10、J18；test 为 J09、J12、J16、J20。绑定完整原始 CSV 时，分别有 6,480 / 2,160 / 2,160 行，每集复用全部 540 个公开 seed。新版本 builder 用实际保留的 base_id 单独物化 view，不把这些原始行数误写成任何尚未验收版本的结果。

| 评测范围 | 分组单位 | 各 split 的 wrapper | 父 seed 是否跨 split | 用途 |
| --- | --- | --- | --- | --- |
| 主划分 | 父 seed / 原文 / 登记风格家族 | JailBench 20 类均可出现 | JailBench 父 seed 不跨集 | 主方法比较和新请求泛化 |
| 独立 template holdout view | 完整 wrapper 家族 | 12 / 4 / 4 类 | 有意复用公开 seed | 新 wrapper 对有害识别的影响 |

JailBench 来源没有独立的原生二分类字段；项目继续保留其历史 `harmful` 规则。因此纯 JailBench view 是正类评测，支持有害召回等指标，单靠它无法评价安全请求的误报率或完整混合标签准确率。与原请求成对派生的安全格式对照保留 `parent_base_id`，用 `inherit_parent_splits` 继承主父样本的 group 和 split，单独登记和评测。它们不能依照模板文字把不同父样本重新连起来。

验收需保存完整的 group mapping、来源 × 二分类目标与实际计数、每来源标签比例的 train / test 漂移、全部 flags，以及两个评测范围各自的数据与 checkpoint 身份。主协议要求 group、官方原文身份和 JailBench 父 seed 跨 split 的计数全部为 0。
