# 同划分五分类外部基线：Chinese-MacBERT-base

三 seed 已完成；原生两来源已知有害测试样本共 2,180 条。

| 指标 | 三 seed 均值 ± 样本标准差 |
|---|---:|
| accuracy | 81.04 ± 0.63% |
| macro_f1 | 79.89 ± 0.71% |

使用官方 HFL Chinese-MacBERT-base 固定 revision，FP32 参数、CUDA BF16 autocast，端到端微调。
所有原文保留；长文采用 510 内容 token 窗口、128 token 重叠，窗口 logits 均值后按文档计算 CE。
训练/验证/测试与主研究完全相同；三 seed 最小验证 CE 检查点锁定后才处理测试文本。
该对比采用不同模型与训练预算，不是冻结骨干模块的容量匹配消融。

官方来源：https://huggingface.co/hfl/chinese-macbert-base；许可证 Apache-2.0。
