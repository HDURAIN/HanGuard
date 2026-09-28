# 演示输入

`prompts.jsonl` 是通用中文演示输入，可直接交给 `infer.py --input`。示例不含预设预测；以实时输出为准。

`power_grid.md` 与 `power_grid.csv` 是仓库原有电网场景输入，已从根目录迁至此处；它们不是新增训练样本，不用于报告模型准确率。可复制单条文本传给 `demo_infer.py --text`。CSV 沿用原字段，批量推理需要按 `infer.py --help` 指定或提供 `prompt` 列。
