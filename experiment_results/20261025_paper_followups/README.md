# 论文补充实验（20261025）

同一数据划分，复用20261022本轮新特征；不改变已完成的主消融结果。

| 项目 | 内容 | 当前进度 |
|---|---|---|
| pooling | 全文均值/共享注意力＋普通分类头，二分类与五分类各三seed | `{"total_jobs": 6, "selected_jobs": 6, "fit_complete_jobs": 6, "completed_jobs": 6, "test_complete_jobs": 6, "total_heads": 12, "selected_heads": 12, "test_complete_heads": 12, "pending_jobs": [], "test_barrier_locked": true}` |
| external | Chinese-MacBERT-base，原生两来源五分类三seed | `{"state": "complete", "pending_jobs": [], "completed_jobs": 6, "total_jobs": 6, "selected_seeds": [42, 43, 44], "evaluated_seeds": [42, 43, 44]}` |
| analysis | 分类别错误分析与空卡端到端效率 | `{"cpu_analysis_complete": true, "benchmark_complete": true, "benchmark_pending": false}` |

训练允许有显存余量的共享卡；效率只在无其他计算进程的卡上测量。
进度来自各子协议，未完成项不生成或替换为旧成绩。

运行状态：`outputs/hanguard_paper_followups_20261025/queue_state.json`。
