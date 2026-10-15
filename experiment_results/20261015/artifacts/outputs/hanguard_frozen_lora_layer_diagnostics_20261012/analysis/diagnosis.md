# 同一冻结 LoRA：多层融合收益诊断

## 结论

这轮结果不支持“数据没有跨层互补信息”。同一个 E03_s42 冻结 LoRA 下，跨层错误互补很明显，而且一个简单的线性 logit stack 已能在 test 上把互补转成可测收益。当前小收益主要来自融合头的聚合方式、训练路径、验证阈值不稳定和 head seed 方差。数据侧存在明显的来源天花板与难度不均，但没有发现缓存、身份、split 或重复 group 泄漏问题。

## 协议

- 唯一骨干：E03_s42 LoRA，adapter SHA256 `5283c5c6b210f33539a06a0583135875a27d86b6914127d17ffba5658691ae25`。
- 缓存：32 个 block 的最后有效 token、final norm 之前的 raw state，float16；train/validation/test 分别 62,155 / 7,753 / 7,778 行。
- probe：每层同一个 2,561 参数线性分类器、相同初值和 batch 顺序；每层独立恢复最低 validation BCE 的 epoch。
- 容量对照：`repeat_last`、`fusion_fixed`、`fusion_selected` 均为 1,656,709 参数；同 seed 的初始状态逐张量相同。`last` 为 327,937 参数。
- 所有 checkpoint、阈值和选层均在 test barrier 前由 train/validation 固定。当前 test 身份过去已被探索，因此统计量属于诊断性复演。

## 1. 冻结表示中确有互补

逐层 probe 从浅到深持续提升，但最后一层不是唯一最优层。test accuracy：L25 为 96.799%，L32 为 96.696%；L20–L32 大多在 96.2%–96.8%。这排除了“只有最后一层保留任务信号”的解释。

错误互补上界更直接：

| probe 集合 | Validation oracle | Test oracle | Test 修正 L32 错误 | 独立 group |
|---|---:|---:|---:|---:|
| L8/L16/L24/L32 | 98.891% | 98.624% | 150 / 257（58.4%） | 146 |
| L17/L18/L28/L32 | 98.620% | 98.059% | 106 / 257（41.2%） | 103 |

浅层本身较弱，但错误更不相同。例如 test 上 L8 相对 L32 修正 108 条，同时损伤 570 条。这类信号需要按样本决定何时采用，固定全局权重难以直接利用。

作为事后可学习性检查，用 train 拟合 L2 logistic stack、仅用 validation 选择 C 和阈值，再冻结到 test：

| 输入 probe logits | Test accuracy | 相对 L32 | group bootstrap 95% CI | group permutation p |
|---|---:|---:|---:|---:|
| L32 | 96.696% | — | — | — |
| L8/L16/L24/L32 | 96.966% | +0.270pp | [+0.026, +0.517]pp | 0.0410 |
| L17/L18/L28/L32 | 96.940% | +0.244pp | [−0.026, +0.519]pp | 0.1000 |
| 全 32 层 | 97.056% | +0.360pp | [+0.103, +0.617]pp | 0.0083 |

这项 stack 是看过主实验结果后追加的探索性诊断，不能当新的确认实验；它仍说明 oracle 互补并非完全不可学习。

## 2. 额外容量几乎不是解释

三枚 head seed 的 test 均值：

| 头 | Test accuracy 均值 ± SD | 相对对照 |
|---|---:|---:|
| last | 96.829% ± 0.164 | — |
| repeat_last | 96.859% ± 0.270 | 比 last +0.030pp |
| fusion_fixed | 96.884% ± 0.045 | 比 repeat_last +0.026pp |
| fusion_selected | 96.867% ± 0.136 | 比 repeat_last +0.009pp |

增加约 5 倍头参数但重复使用 L32，只带来 +0.030pp。真实多层输入在平均上也只增加 +0.026pp / +0.009pp。容量本身既没有制造明显假收益，也没有解决互补利用问题。

seed 级差值不稳定：固定融合相对 `repeat_last` 为 +0.039 / +0.244 / −0.206pp；选层融合为 +0.141 / +0.296 / −0.411pp。两种融合都只有 2/3 seed 为正。seed 44 的选层融合显著变差（cluster CI [−0.689, −0.131]pp），所以当前不能声称稳定提升。

## 3. 阈值迁移放大了不稳定性

`fusion_selected` 的 test AUROC 在三枚 seed 都高于 `repeat_last`，平均 +0.0116pp；test BCE 在三枚 seed 也都更低，平均差 −0.00167。可是按 validation 最大准确率选出的硬阈值后，accuracy 平均只剩 +0.009pp。

seed 44 最典型：`repeat_last` 与选层融合的 validation 阈值分别为 0.430 和 0.707；后者的 test 事后最优阈值约 0.498。冻结 0.707 后相对容量对照为 −0.411pp，而使用预先固定的 0.5 时为 +0.077pp。选层融合的 0.5 阈值三 seed 差值为 +0.090 / −0.013 / +0.077pp，均值 +0.051pp。

因此模型的连续分数已有小而一致的改善，单个 validation 集上的离散 accuracy 最优点把少量校准波动放大成了正负翻转。

## 4. 当前融合结构与错误形态不匹配

融合头使用一组对所有样本相同的四层 softmax 权重；训练后固定融合的权重大致仍在 22%–28% 之间。它不能针对某个输入选择“这条信 L8、下一条信 L32”。probe 四格表显示中间层同时有大量修正与损伤，正需要输入相关的 gate。

融合残差输出又从零初始化。第一步只有 `fusion_output` 能得到有效梯度，四个 layer-value 分支和 layer logits 要等输出映射离开零点后才开始接收信号。与“每层先受独立监督，再做正则化 stack”相比，这条间接优化路径更难。线性 stack 的结果说明，先形成各层可判别 logits 再组合是更合适的基线。

validation 的互补贪心选层也不等于最易学习的融合集合。它要求单层 AUROC 接近 L32，选出 L17/L18/L28/L32；这些晚层彼此高度相关。固定的 L8/L16/L24/L32 虽包含较弱层，却有更大的 oracle 和更好的事后 stack 区间。选层目标应直接使用 group-OOF 的可学习 stack 增益，而不是 standalone AUROC 加 oracle rescue。

## 5. 数据侧发现

- 数据哈希、行数、ordered `base_id`、非空 `group_id`、跨 split 的 base/group 隔离、token 长度和缓存 finite/extent 全部通过。
- 用原 E03 小头读取缓存 L32，test 7,778 条硬判定与原在线推理 100% 一致，概率相关系数 0.9999995；validation 仅 1 条因 float16 量化跨阈值。没有缓存错位证据。
- L32 probe 的 257 个 test 错误中，238 个（92.6%）来自 WildGuard；固定四层的 150 个 oracle 修正中 138 个也来自 WildGuard。Chinese curated 与 JailBench 已接近 99.5%–100%，总体增益自然被天花板压缩。
- fusion 对 `repeat_last` 的逐行与 group-equal 差值方向一致；选层融合三 seed 的 group-equal 差值为 +0.169 / +0.313 / −0.440pp。JailBench wrapper 重复没有制造主要结论。
- 来源与类别切片的收益方向不稳定，category 3 在当前三 seed 中尤其没有获益。数据异质性和小切片方差是次要限制，仍值得做标签/边界样本复核。

## 建议的下一轮

1. 先修阈值：用按 `group_id` 分层的 train OOF 预测拟合 Platt/temperature calibration；validation 只作 gate。避免直接取单个 validation accuracy 的离散最优点，或在校准后预注册 0.5。
2. 把“独立 per-layer probe + 正则化 logit stack”作为主融合基线，并做参数匹配；再比较输入相关 gate。当前全局四权重不足以处理 rescue/harm 同时很大的层。
3. 用 group-OOF 的真实 stack 增益选层。固定集合 L8/L16/L24/L32 应继续保留，不能因 L8 standalone 较弱而排除。
4. 对融合分支做监督 warm-up，或取消使分支首步无梯度的零输出初始化；分别搜索 fusion-output、layer-value、classifier 学习率和训练轮数。
5. 至少跑 5 个 head seed，并用 3 个以上独立 LoRA seed 重复。当前三个 head seed 只覆盖读出层方差，结论仍条件于“为末层任务训练过的 E03 LoRA”。
6. 重点复核 WildGuard 与 category 3 的错误和标签边界，并用新的外部盲测集确认；当前 test 已被历史实验探索。

