本目录是额外监督适配实验，用户要求的原始 Qwen3Guard 能力对照见 [原版评测](../../experiment_results/20261025_qwen3guard_clean_test_v2/ORIGINAL_CAPABILITY_COMPARISON.md)。

分类头在本地数据上训练，因此 97.44%／83.33% 是适配后的二分类／五分类成绩，不能称为原版 Qwen3Guard 准确率。此前将此表当作原始能力对照的解释撤回；模型、协议、预测及原始数值保持不变。

原版在完整 8,350 条上的二分类准确率为 90.86%（仅 Unsafe）和 94.83%（Unsafe＋Controversial）；两种映射均推理前固定。原生九类与本地五类没有可靠直接映射，原生五分类为 N/A。
