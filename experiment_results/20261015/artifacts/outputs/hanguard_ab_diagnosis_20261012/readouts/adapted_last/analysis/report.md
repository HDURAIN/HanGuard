# A/B 冻结表征诊断

B 的 20,048 条数据均由未训练过其所在 group 的分类头预测。
LoRA 只使用 A；该结果用于定位原因，不是正式测试成绩。

所有层选择、停止轮次和阈值都在对应外层训练部分的 inner dev 确定。
置信区间对 group 重采样，并以这一个固定 LoRA 为条件，不能替代多 LoRA 复现。

| 方法 | Accuracy | BCE | AUROC | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| last | 96.364% | 0.0929 | 0.9942 | 95.217% | 2.213% |
| best_single | 96.663% | 0.0929 | 0.9941 | 96.442% | 3.062% |
| repeat_best | 96.623% | 0.0947 | 0.9939 | 96.135% | 2.772% |
| fusion_fixed | 96.713% | 0.0908 | 0.9944 | 96.280% | 2.750% |
| fusion_selected | 96.169% | 0.0996 | 0.9934 | 95.622% | 3.152% |
| fusion_fixed_best_anchor | 96.483% | 0.0973 | 0.9936 | 96.262% | 3.241% |
| fusion_selected_best_anchor | 96.249% | 0.1001 | 0.9934 | 95.820% | 3.219% |
| joint_repeat_best | 96.424% | 0.0998 | 0.9933 | 96.406% | 3.554% |
| joint_fixed | 96.598% | 0.0954 | 0.9940 | 96.478% | 3.252% |
| joint_selected | 96.498% | 0.0936 | 0.9940 | 95.811% | 2.649% |

| 比较 | 修正 | 新增错误 | 净修正 | Accuracy Δ及 group 95% CI |
|---|---:|---:|---:|---:|
| best_single − last | 226 | 166 | 60 | 0.299 pp [0.069, 0.518] |
| repeat_best − best_single | 96 | 104 | -8 | -0.040 pp [-0.191, 0.106] |
| fusion_fixed − last | 175 | 105 | 70 | 0.349 pp [0.177, 0.510] |
| fusion_fixed − best_single | 154 | 144 | 10 | 0.050 pp [-0.149, 0.252] |
| fusion_fixed − repeat_best | 150 | 132 | 18 | 0.090 pp [-0.087, 0.272] |
| fusion_selected − best_single | 150 | 249 | -99 | -0.494 pp [-0.690, -0.302] |
| fusion_selected − repeat_best | 113 | 204 | -91 | -0.454 pp [-0.645, -0.277] |
| fusion_fixed_best_anchor − repeat_best | 122 | 150 | -28 | -0.140 pp [-0.316, 0.034] |
| fusion_selected_best_anchor − repeat_best | 109 | 184 | -75 | -0.374 pp [-0.562, -0.193] |
| fusion_fixed_best_anchor − fusion_fixed | 111 | 157 | -46 | -0.229 pp [-0.385, -0.065] |
| fusion_selected_best_anchor − fusion_selected | 31 | 15 | 16 | 0.080 pp [0.010, 0.151] |
| joint_fixed − joint_repeat_best | 190 | 155 | 35 | 0.175 pp [-0.039, 0.384] |
| joint_selected − joint_repeat_best | 194 | 179 | 15 | 0.075 pp [-0.192, 0.324] |

单层 probe 的 oracle 只是这些已训练 probe 的错误并集诊断，不能当作可部署融合收益或严格上界。
若任何诊断促使方法修改，需要将后续结果标明探索性；正式验证、测试仍独立保留。
