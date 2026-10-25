# 固定权重融合与类别向量四组实验（20261019）

当前状态：**pending**。这是独立登记的无 LoRA 四臂实验，未知结果保留 `pending`。

| 组别 | 固定四层等权融合 | 类别向量 |
|---|---|---|
| `baseline` | × | × |
| `fusion_only` | ✓ | × |
| `category_only` | × | ✓ |
| `fusion_category` | ✓ | ✓ |

骨干冻结为原始 Qwen3.5-4B；层为 L8/L16/L24/L32，每层权重固定 0.25。类别向量聚合 L32 的全部有效 token。二分类与五分类均独立运行上述四组，seeds 为 42/43/44，检查点、阈值只由验证集选择。

三个指标范围分开报告：

- **直接二分类**：三来源，训练/验证/测试为 66,802/8,350/8,350；同时列出沿用既有 122 个排除 ID 的固定 8,228 条子集。两组范围内所有方法样本一致。
- **五类主要风险识别**：两来源中既有标签的有害样本，17,440/2,180/2,180；不把未知类型当负例，不重新标注。
- **完整系统六类输出**：在有已知最终标签的 3,704 条测试样本上，比较二分类 gate 与五类类型头组合。经典 2×2 为 gate `baseline/fusion_only` × type `baseline/category_only`；其余类型融合组合单列补充表。这里的六类准确率不能与前两项混用分母。

旧 **+2.1026 个百分点**来自另一个有可学习层权重和最佳单层锚点的旧模型在事后选定子集上的固定预测；它不属于这次固定等权四组结果，不填入新主表。旧结果及已完成的细组件/seed 消融保留在[旧证据包](../20261019_fusion_ablation/README.md)。

## 三种子主表

只有全部三个种子完成，才显示均值 ± 样本标准差；单一种子完成不会被写成三种子结果。


### 直接二分类

| 测试范围 | 方法 | 状态 | 准确率均值 ± SD / % | Macro-F1 均值 ± SD / % |
|---|---|---|---:|---:|
| binary_fixed_subset_8228 | `baseline` | pending | pending | — |
| binary_fixed_subset_8228 | `category_only` | pending | pending | — |
| binary_fixed_subset_8228 | `fusion_category` | pending | pending | — |
| binary_fixed_subset_8228 | `fusion_only` | pending | pending | — |
| binary_full_test_8350 | `baseline` | pending | pending | — |
| binary_full_test_8350 | `category_only` | pending | pending | — |
| binary_full_test_8350 | `fusion_category` | pending | pending | — |
| binary_full_test_8350 | `fusion_only` | pending | pending | — |

### 五类主要风险识别

| 测试范围 | 方法 | 状态 | 准确率均值 ± SD / % | Macro-F1 均值 ± SD / % |
|---|---|---|---:|---:|
| category_known_harmful_2180 | `baseline` | pending | pending | pending |
| category_known_harmful_2180 | `category_only` | pending | pending | pending |
| category_known_harmful_2180 | `fusion_category` | pending | pending | pending |
| category_known_harmful_2180 | `fusion_only` | pending | pending | pending |

### 完整系统经典 2×2

| 测试范围 | 方法 | 状态 | 准确率均值 ± SD / % | Macro-F1 均值 ± SD / % |
|---|---|---|---:|---:|
| pipeline_known_labels_3704 | `baseline` | pending | pending | pending |
| pipeline_known_labels_3704 | `category_only` | pending | pending | pending |
| pipeline_known_labels_3704 | `fusion_category` | pending | pending | pending |
| pipeline_known_labels_3704 | `fusion_only` | pending | pending | pending |

### 完整系统：类型阶段也加入融合

| 测试范围 | 方法 | 状态 | 准确率均值 ± SD / % | Macro-F1 均值 ± SD / % |
|---|---|---|---:|---:|
| pipeline_known_labels_3704 | `baseline` | pending | pending | pending |
| pipeline_known_labels_3704 | `category_only` | pending | pending | pending |
| pipeline_known_labels_3704 | `fusion_category` | pending | pending | pending |
| pipeline_known_labels_3704 | `fusion_only` | pending | pending | pending |

逐种子数值见 [results_summary.csv](results_summary.csv)，均值见 [seed_summary.csv](seed_summary.csv)，任务完成状态见 [coverage.csv](coverage.csv)。[index.json](index.json)记录快照与范围，[manifest.json](manifest.json)记录文件及源文件 SHA256。

`artifacts/` 按原仓库相对路径保存小协议、选择记录、指标及 Python 源码。逐样本预测只输出 ID、标签、来源、长度和数值预测，不含中文/英文原文；权重、特征、日志与进程临时文件不打包。该包不能替代完整基座、数据和特征缓存。

刷新与独立校验（不启动训练）：

```bash
python scripts/hanguard/package_four_arm_experiments.py
python scripts/hanguard/package_four_arm_experiments.py --verify
```
