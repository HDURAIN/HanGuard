# A/B 冻结表征诊断

B 的 20,048 条数据均由未训练过其所在 group 的分类头预测。
LoRA 只使用 A；该结果用于定位原因，不是正式测试成绩。

所有层选择、停止轮次和阈值都在对应外层训练部分的 inner dev 确定。
置信区间对 group 重采样，并以这一个固定 LoRA 为条件，不能替代多 LoRA 复现。

| 方法 | Accuracy | BCE | AUROC | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| last | 92.508% | 0.1817 | 0.9805 | 92.280% | 7.209% |
| best_single | 93.446% | 0.1603 | 0.9851 | 93.469% | 6.583% |
| repeat_best | 93.141% | 0.1720 | 0.9827 | 91.911% | 5.331% |
| fusion_fixed | 93.501% | 0.1595 | 0.9850 | 93.271% | 6.214% |
| fusion_selected | 93.401% | 0.1682 | 0.9842 | 93.712% | 6.986% |
| fusion_fixed_best_anchor | 93.805% | 0.1576 | 0.9860 | 93.883% | 6.293% |
| fusion_selected_best_anchor | 93.880% | 0.1602 | 0.9857 | 94.235% | 6.561% |
| joint_repeat_best | 93.002% | 0.1743 | 0.9820 | 93.190% | 7.231% |
| joint_fixed | 93.670% | 0.1541 | 0.9859 | 94.406% | 7.243% |
| joint_selected | 93.585% | 0.1539 | 0.9856 | 93.199% | 5.935% |

| 比较 | 修正 | 新增错误 | 净修正 | Accuracy Δ及 group 95% CI |
|---|---:|---:|---:|---:|
| best_single − last | 764 | 576 | 188 | 0.938 pp [0.549, 1.352] |
| repeat_best − best_single | 258 | 319 | -61 | -0.304 pp [-0.541, -0.069] |
| fusion_fixed − last | 623 | 424 | 199 | 0.993 pp [0.647, 1.325] |
| fusion_fixed − best_single | 517 | 506 | 11 | 0.055 pp [-0.290, 0.390] |
| fusion_fixed − repeat_best | 529 | 457 | 72 | 0.359 pp [-0.000, 0.684] |
| fusion_selected − best_single | 419 | 428 | -9 | -0.045 pp [-0.360, 0.261] |
| fusion_selected − repeat_best | 420 | 368 | 52 | 0.259 pp [-0.051, 0.536] |
| fusion_fixed_best_anchor − repeat_best | 416 | 283 | 133 | 0.663 pp [0.397, 0.912] |
| fusion_selected_best_anchor − repeat_best | 383 | 235 | 148 | 0.738 pp [0.469, 0.985] |
| fusion_fixed_best_anchor − fusion_fixed | 436 | 375 | 61 | 0.304 pp [-0.000, 0.611] |
| fusion_selected_best_anchor − fusion_selected | 343 | 247 | 96 | 0.479 pp [0.218, 0.741] |
| joint_fixed − joint_repeat_best | 561 | 427 | 134 | 0.668 pp [0.354, 0.983] |
| joint_selected − joint_repeat_best | 490 | 373 | 117 | 0.584 pp [0.276, 0.891] |

单层 probe 的 oracle 只是这些已训练 probe 的错误并集诊断，不能当作可部署融合收益或严格上界。
若任何诊断促使方法修改，需要将后续结果标明探索性；正式验证、测试仍独立保留。
