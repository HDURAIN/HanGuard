# Qwen3Guard 首次尝试：因结果序列化错误停止

模型已加载并完成首批生成，但写结果时 categories 关键字重复，导致异常。完成分块为 0，没有有效测试成绩。旧协议、初始源码快照、调度修订和日志原样保留，不能当作已完成实验。

修复版本将从相同完整 8,350 条测试文本重新分词和推理，使用新输出目录 `outputs/qwen3guard_clean_test_20261025_v2`，证据包为 `experiment_results/20261025_qwen3guard_clean_test_v2`。未修改旧协议以匹配修复后的源码。
