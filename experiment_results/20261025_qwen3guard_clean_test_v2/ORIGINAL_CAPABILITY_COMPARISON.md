本文件对应用户要求的 Qwen3Guard-Gen-4B 原始能力外部对照。使用官方模型、官方单 user 审核模板和模型原生生成结果；没有额外训练分类头、LoRA、类别查询或融合模块。

| 模型／原生输出判定规则 | 二分类 Accuracy / % | 本地五分类 Accuracy / % |
|---|---:|---:|
| Qwen3Guard 原版：仅 Unsafe 判有害 | 90.86 | N/A |
| Qwen3Guard 原版：Unsafe＋Controversial 判有害 | 94.83 | N/A |
| Hanguard 类别查询＋融合（本地监督，3 seed） | 96.12 ± 0.25 | 83.13 ± 0.87 |

二分类使用同一测试集全部 8,350 条，模型输出覆盖率 100%，无无效输出；两种映射均在推理前登记，不按测试结果挑选。Hanguard 使用本地训练监督，Qwen3Guard 原版直接零样本推理；这张表比较任务表现，不能隔离内部方法贡献。

Qwen3Guard 官方原生输出 Safe / Unsafe / Controversial 及九类风险；本地五分类是另一套主要风险领域标签。核对定义后，没有可直接采用的可靠一一映射，例如 Unethical Acts 横跨本地歧视、权益侵害等类别，Jailbreak 作为攻击方式横跨五类。原生五分类明确记 N/A，不把附加训练分类头的 83.33% 填入。

此前 97.44%／83.33% 属于额外监督适配结果，只保留为附加实验；它们不代表原始 Qwen3Guard 能力，也不替代本表。

[官方标签与使用方式](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B)；[本地五类标准核查](../../docs/primary_category_standard_review.md)。
