# A/B 冻结表征诊断

B 的 20,048 条数据均由未训练过其所在 group 的分类头预测。
LoRA 只使用 A；该结果用于定位原因，不是正式测试成绩。

所有层选择、停止轮次和阈值都在对应外层训练部分的 inner dev 确定。
置信区间对 group 重采样，并以这一个固定 LoRA 为条件，不能替代多 LoRA 复现。

| 方法 | Accuracy | BCE | AUROC | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| last | 95.396% | 0.1226 | 0.9903 | 95.496% | 4.728% |
| best_single | 96.289% | 0.1059 | 0.9928 | 96.027% | 3.387% |
| repeat_best | 96.134% | 0.1061 | 0.9929 | 96.045% | 3.755% |
| fusion_fixed | 96.104% | 0.1104 | 0.9919 | 96.226% | 4.046% |
| fusion_selected | 95.945% | 0.1118 | 0.9919 | 95.856% | 3.945% |
| fusion_fixed_best_anchor | 96.159% | 0.1037 | 0.9928 | 96.099% | 3.767% |
| fusion_selected_best_anchor | 96.224% | 0.1058 | 0.9926 | 96.181% | 3.722% |
| joint_repeat_best | 95.875% | 0.1119 | 0.9916 | 95.487% | 3.644% |
| joint_fixed | 95.965% | 0.1057 | 0.9925 | 96.892% | 5.186% |
| joint_selected | 96.069% | 0.1050 | 0.9925 | 96.298% | 4.214% |

| 比较 | 修正 | 新增错误 | 净修正 | Accuracy Δ及 group 95% CI |
|---|---:|---:|---:|---:|
| best_single − last | 398 | 219 | 179 | 0.893 pp [0.642, 1.154] |
| repeat_best − best_single | 126 | 157 | -31 | -0.155 pp [-0.316, 0.010] |
| fusion_fixed − last | 329 | 187 | 142 | 0.708 pp [0.491, 0.933] |
| fusion_fixed − best_single | 199 | 236 | -37 | -0.185 pp [-0.399, 0.015] |
| fusion_fixed − repeat_best | 229 | 235 | -6 | -0.030 pp [-0.243, 0.188] |
| fusion_selected − best_single | 184 | 253 | -69 | -0.344 pp [-0.555, -0.141] |
| fusion_selected − repeat_best | 218 | 256 | -38 | -0.190 pp [-0.413, 0.025] |
| fusion_fixed_best_anchor − repeat_best | 170 | 165 | 5 | 0.025 pp [-0.148, 0.204] |
| fusion_selected_best_anchor − repeat_best | 189 | 171 | 18 | 0.090 pp [-0.100, 0.273] |
| fusion_fixed_best_anchor − fusion_fixed | 152 | 141 | 11 | 0.055 pp [-0.117, 0.214] |
| fusion_selected_best_anchor − fusion_selected | 156 | 100 | 56 | 0.279 pp [0.122, 0.431] |
| joint_fixed − joint_repeat_best | 230 | 212 | 18 | 0.090 pp [-0.114, 0.298] |
| joint_selected − joint_repeat_best | 199 | 160 | 39 | 0.195 pp [0.000, 0.374] |

单层 probe 的 oracle 只是这些已训练 probe 的错误并集诊断，不能当作可部署融合收益或严格上界。
若任何诊断促使方法修改，需要将后续结果标明探索性；正式验证、测试仍独立保留。
