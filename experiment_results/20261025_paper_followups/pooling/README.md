# 通用全文池化补充对照

| 任务 | 方法 | Accuracy 均值±SD / % |
|---|---|---:|
| binary | mean_pool_mlp | 96.14 ± 0.16 |
| binary | shared_attention_mlp | 95.92 ± 0.23 |
| category | mean_pool_mlp | 82.42 ± 0.97 |
| category | shared_attention_mlp | 82.81 ± 0.59 |

两个读出均不使用显式类别查询或类别身份向量，使用普通 MLP 输出类别得分。
均值池化明确为：逐 token 无参数 LN → Linear/GELU/LN 值投影 → 有效 token 均值。
共享注意力为单个独立可学习 query；同一文本仅有一份共享摘要。
参数量按每组 selection.json 实际功能参数报告，训练延长和 OOM 拆批亦单独记录。
这是查看父实验结果后登记的补充对照；不声称整个研究的测试集从未被查看。
