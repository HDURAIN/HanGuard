# Qwen3Guard 同数据监督分类对照（20261025）

冻结 Qwen3Guard-Gen-4B，不使用 LoRA；二分类和原生两来源五分类各三个 seed。
末位置 MLP、均值池化 MLP、独立共享注意力 MLP 全部报告。
官方完整模板；池化仅原始文本 token，末位置读出含官方后缀。
v2 全部使用固定单条形状提取，同形状重复要求逐位一致；v1 无有效收据，不作为输入。
与主实验相同 train/validation/test；全部验证集选模锁定后才提取测试特征。
本项目测试集此前已查看，本轮属于后续对照。
原生生成二分类在 ../20261025_qwen3guard_clean_test_v2 单列，不能混称监督成绩。

实时状态：`outputs/qwen3guard_supervised_comparison_20261025_v2/queue_state.json`。
