**用途纠正：本目录为额外监督适配，不代表原始 Qwen3Guard 能力，不作为原版外部基线。** 原版结果为二分类 90.86%／94.83%，本地五分类 N/A；详见 [用途说明](EVALUATION_SCOPE.md)。

# Qwen3Guard 冻结骨干监督分类对照

| 任务 | 普通分类头 | Accuracy 均值±SD / % |
|---|---|---:|
| binary | last_mlp | 97.39 ± 0.05 |
| binary | mean_pool_mlp | 97.18 ± 0.05 |
| binary | shared_attention_mlp | 97.44 ± 0.15 |
| category | last_mlp | 75.63 ± 3.45 |
| category | mean_pool_mlp | 82.74 ± 1.02 |
| category | shared_attention_mlp | 83.33 ± 0.64 |

二分类阈值只在验证集确定，同时保留固定 0.5 阈值成绩。五分类只使用同一原生已标注有害测试子集。
原生生成式二分类与本次监督分类头对照分开报告。共享注意力为独立全局 query，未加入类别查询或多层融合。
