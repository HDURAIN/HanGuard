# Input-conditioned dynamic gate 诊断

同一个冻结 E03_s42 LoRA、同一份最后有效 token 的 32 层缓存。三个动态门控头逐参数同构；dynamic_repeat_last 将第 32 层复制四份，因此是容量、初始化和优化过程匹配的主要对照。
主结果预注册使用固定阈值 0.5；validation 最大 accuracy 阈值仅保存在结果中作为敏感性分析。

## Post-barrier evaluator amendment

原 test barrier 未修改或重锁。锁定版 evaluator 错误地假定 5 个 dynamic seeds 都有原静态头预测；实际仅 seeds 42/43/44 存在，因此旧运行在写出 dynamic seeds 42/43/44/45 的预测后、生成 results.json/report.md 前失败。
锁定代码 SHA-256：`580130cad2c9527c2dd7a19b9031e0ece9be331c4e57d74a32431dc1ed4f356c`；首次交集修复 SHA-256：`50dc9f4ecd4e51b2df21655d72495cdd4749d155e22a1854d1fcdee1c3b75057`；带恢复门禁的代码 SHA-256：`a30b7abc15a34f84daa418a32c3ab4da4b8d01c3938e617fa5463b68c6d03796`。
不可变 amendment SHA-256：`01382c9b37b57988d0141fad4972f0d069b09d50a3f10128be70cdc30a45d0ff`；seed 42–45 精确重算审计 SHA-256：`f6c52ccee56c0f499af942d8fe5c7d7ff830faea8943285058c82e4e5ab88c2b`。逐行 schema、dtype、概率、决策和 gate 权重均逐值完全一致；已有预测未被覆盖。
主等容量比较仍使用全部 seeds=[42, 43, 44, 45, 46]；次要静态比较按 amendment 固定为真实共同 seeds=[42, 43, 44]。

## 测试集结果（5 个 head seeds）

| arm | Accuracy mean±sd | F1 mean±sd | AUROC | BCE | gate entropy/log(4) | effective layers |
|---|---:|---:|---:|---:|---:|---:|
| dynamic_repeat_last | 96.930±0.075 | 97.057±0.075 | 0.995878 | 0.081389 | 0.9392 | 3.699 |
| dynamic_fixed | 96.912±0.116 | 97.034±0.117 | 0.996003 | 0.081930 | 0.8962 | 3.493 |
| dynamic_selected | 97.020±0.108 | 97.145±0.100 | 0.996136 | 0.080098 | 0.8743 | 3.424 |

## 与动态 repeat baseline 的容量匹配比较

差值为候选减 dynamic_repeat_last。每个 seed 的区间按 group_id 整组 bootstrap，随机化检验也按 group_id 整组交换。seed 均值只描述这 5 个 head 初始化；主等容量比较完整使用 seeds=[42, 43, 44, 45, 46]。

| candidate | Accuracy Δ mean±sd (pp) | F1 Δ mean±sd (pp) | positive seeds |
|---|---:|---:|---:|
| dynamic_fixed | -0.018±0.131 | -0.023±0.135 | 2/5 |
| dynamic_selected | 0.090±0.114 | 0.088±0.112 | 4/5 |

## 与原全局权重头的次要比较

动态头多了 query/key/score 参数，所以下表不是容量匹配因果对照，只用于判断动态门控是否值得继续。两侧均用固定阈值 0.5。
静态头仅存在于共同 seeds=[42, 43, 44]（n=3）；dynamic-only seeds=[45, 46] 没有对应静态 test prediction，因此未纳入本附加比较，也没有复制或合成预测。主等容量比较仍为完整 5 seeds。

| dynamic | global baseline | n | seeds | Accuracy Δ mean±sd (pp) | F1 Δ mean±sd (pp) |
|---|---|---:|---|---:|---:|
| dynamic_repeat_last | repeat_last | 3 | [42, 43, 44] | 0.021±0.079 | 0.029±0.068 |
| dynamic_fixed | fusion_fixed | 3 | [42, 43, 44] | -0.051±0.112 | -0.058±0.104 |
| dynamic_selected | fusion_selected | 3 | [42, 43, 44] | -0.009±0.071 | -0.010±0.068 |

## 门控解释

normalized entropy 接近 1 表示接近均匀权重；effective layers 是 exp(entropy)。mean_l1_from_global_mean 与各 slot 的标准差衡量权重是否随输入变化。门控变化本身不证明有用，主要证据仍是同容量 dynamic_repeat_last 配对比较。

验证集选择层：[17, 18, 28, 32]；固定层：[8, 16, 24, 32]。

## 限制

- The frozen adapter was trained jointly with an E03 final-layer head.
- Head seeds are not independent LoRA seeds.
- The source test split has already been explored; this is a post-hoc diagnostic.
- The selected-layer arm reuses a validation-driven exploratory layer selection.
- The primary threshold is fixed at 0.5; no out-of-fold calibration is fitted in this diagnostic.
- Per-source and gate-weight analyses are descriptive and unadjusted for multiplicity.
