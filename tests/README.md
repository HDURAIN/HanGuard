# 测试导航

本目录保留当前方法、兼容部署实现和历史实验的测试。文件名对应被测实现；历史测试存在不表示需要重新启动相应实验。

| 范围 | 主要测试 |
|---|---|
| 当前主消融 | [分类组件](test_clean_ablation_heads.py)、[特征身份](test_clean_ablation_features.py)、[训练与选择](test_clean_ablation_training.py)、[实验流程](test_clean_ablation_study.py)、[运行配置](test_clean_ablation_runtime.py)、[I/O边界](test_clean_ablation_io.py) |
| 当前证据归档 | [主实验打包](test_package_clean_ablation_results.py)；论文派生表用打包工具的`--verify`核对 |
| 原生Qwen3Guard | [推理与评分](test_qwen3guard_clean_eval.py)、[原队列](test_qwen3guard_clean_queue.py)、[共享调度](test_qwen3guard_shared_runner.py) |
| 论文补充 | [通用聚合](test_paper_pooling_controls.py)、[MacBERT](test_paper_external_classifier.py)、[分析与效率](test_paper_error_efficiency.py)、[实验流程](test_paper_followup_study.py)、[恢复实现](test_paper_followup_recovery.py) |
| 部署与数据合约 | [运行时](test_hanguard_runtime.py)、[导出](test_export_model.py)、[兼容训练入口](test_current_training.py)、[未知类型评估](test_evaluate_unknown_types.py)、[正式数据合约](test_remediated_dataset_contract.py) |
| 共用类别查询 | [类别查询组件](test_multilabel_heads.py)；历史文件名不代表当前任务是多标签 |
| 历史研究 | `test_four_arm_*`、`test_frozen_*`、`test_ab_*`、`test_sparse_token_*`、`test_qwen3guard_supervised_*`等；用途见[脚本导航](../scripts/hanguard/README.md) |
| 翻译与来源审计 | `test_translation_*`、`test_assemble_translation_reviews.py`、`test_merge_translation_repair.py`等；不修改正式发布 |

在仓库根目录、安装训练依赖的环境中运行已有CPU测试：

```bash
CUDA_VISIBLE_DEVICES='' .venv-hanguard/bin/python -m pytest -q tests
```

测试使用合成输入和临时目录检查相应实现。通过CPU测试不能替代真实模型特征一致性、已登记结果的SHA核验或重新运行实验；完整数据和模型的重放条件见[数据导航](../data/README.md)与[实验结果目录](../experiment_results/README.md)。本次仓库导航整理仅检查文档链接和冻结证据，没有新增训练或GPU推理。
