# hanguard 当前训练入口

统一入口为 `scripts/hanguard/train.py`。它训练分类器，不再训练生成两行标签的旧 SFT 模型。

| 阶段 | 默认数据 | 默认方法 | 训练内容 |
|---|---|---|---|
| `binary` | `data/three_source_translation_repaired` | E03：LoRA＋末层 MLP；E04：LoRA＋多层融合 | 重新加载本地 Qwen3.5-4B，训练 LoRA 和有害判别头 |
| `category` | `data/hanguard_two_source_primary_20260928` | 末 token MLP、可学习类别查询、描述类别查询 | 冻结已选二分类骨干和 LoRA，只训练五分类头 |

二分类使用修复后三来源原始中文文本。类型训练只接受中文整理语料与 JailBench，保留原标签；类别 1–5 参加交叉熵，安全类 0 仅保留用于二分类门控及六类端到端评估。没有重新标注、增强、重划分或截断。

## 环境与冷启动

在仓库根目录执行，当前队列的子进程使用 `.venv-hanguard/bin/python`。依赖见 `requirements-train.txt`；准备本地 `models/Qwen3.5-4B` 的完整模型权重、配置和 tokenizer。入口及 worker 使用本地文件，不自动下载模型或数据。

```bash
python3 -m venv .venv-hanguard
.venv-hanguard/bin/python -m pip install -r requirements-train.txt
```

GPU 编号需按本机空闲设备设置，示例中的 `5 6 7` 不是自动探测结果。单卡可写 `--gpus 0`；多卡按任务并行，每张卡训练一个实验，不是把同一次训练拆成 DDP。

已有合适的二分类父模型时，可直接进行类型阶段。默认父模型为 `outputs/hanguard_repaired_core_20260928/runs/E04_s42`，必须同时保留其父目录的 `protocol.json`、`arms.json` 和该 run 的 `selection.json`、`adapter.pt`、`head.pt`。历史协议包含原路径和哈希；若冷启动环境无法满足这些路径，应先训练一个新的二分类父模型，不改写历史协议。

## 先检查，不启动 GPU

```bash
.venv-hanguard/bin/python scripts/hanguard/train.py binary dry-run \
  --output outputs/hanguard_binary_next --gpus 5 6 7

.venv-hanguard/bin/python scripts/hanguard/train.py category dry-run \
  --output outputs/hanguard_category_next --gpus 5 6 7
```

`dry-run` 检查数据内容、标签一致性、来源范围、跨集 ID/group/规范化全文重叠、长度元数据、模型配置及父检查点。它打印协议和审计，不创建输出目录，不加载模型，不启动 worker。长度元数据检查不能代替 tokenizer 核验：实际 worker 会重新编码完整文本，超预算时报错，不截断。

## 完整训练

重新训练二分类基线与融合方法，默认三个种子，共六组：

```bash
.venv-hanguard/bin/python scripts/hanguard/train.py binary run \
  --output outputs/hanguard_binary_next --gpus 5 6 7
```

如果只需要一个新的融合父模型，可明确缩小范围：

```bash
.venv-hanguard/bin/python scripts/hanguard/train.py binary run \
  --output outputs/hanguard_binary_parent_next \
  --binary-arms E04 --seeds 42 --gpus 5
```

然后用验证集选出的父检查点训练三种类型头，默认三个种子，共九组：

```bash
.venv-hanguard/bin/python scripts/hanguard/train.py category run \
  --output outputs/hanguard_category_next \
  --parent-run outputs/hanguard_binary_parent_next/runs/E04_s42 \
  --gpus 5 6 7
```

`run` 自动完成新目录初始化、数据审计、协议和代码注册、GPU 预检、训练、统一解锁测试及报告。类型阶段先抽取一次绑定全文与父模型的 BF16 特征缓存，再运行三个头；无需人工创建 `training_code_ready.json`。

所有注册方法、种子都选好验证集检查点后才能解锁测试。二分类阈值只在验证集上选择；类型阶段不调整该阈值。类型头按验证 CE 选择检查点，patience 只影响停止时间；即使 CE 的下降小于 `min_delta`，实际最低 CE 的检查点仍会保留。

## 分步执行与恢复

`init` 创建新草稿与数据审计，`register` 校验草稿并锁定协议、数据及代码快照；两者都不启动 GPU。草稿也不支持直接手改参数，需要修改参数时新建目录。

```bash
.venv-hanguard/bin/python scripts/hanguard/train.py category init \
  --output outputs/hanguard_category_next --gpus 5 6 7
.venv-hanguard/bin/python scripts/hanguard/train.py category register \
  --output outputs/hanguard_category_next
.venv-hanguard/bin/python scripts/hanguard/train.py category preflight \
  --output outputs/hanguard_category_next --resume
.venv-hanguard/bin/python scripts/hanguard/train.py category run \
  --output outputs/hanguard_category_next --resume
```

`preflight` **会使用 GPU**，检查真实长度的前向/反向和梯度；类型阶段若尚无缓存，会先完整抽取缓存。它不训练正式实验。已注册目录只有显式 `--resume` 才能继续；恢复命令只接受原阶段、动作、`--output` 与 `--resume`，其他参数保持注册值。代码、数据或父模型变化时拒绝恢复，需要新目录。若旧 supervisor 退出但 worker 仍活着，也会拒绝重复启动。

历史实验不是这个入口创建的，不能用此入口覆盖、重新注册或“恢复”。旧协议及其快照原样保留。

## 主要参数

| 参数 | 二分类默认 | 类型默认 |
|---|---:|---:|
| `--seeds` | `42 43 44` | `42 43 44` |
| `--gpus` | `0` | `0` |
| `--epochs` | 3 轮联合训练 | 最多 40 轮 |
| `--warm-head-epochs` | 1 轮分类头预热 | 不使用 |
| `--head-lr` | 0.0001；预热固定 0.001 | 0.001 |
| `--adapter-lr` | 0.00002 | 不使用 |
| `--effective-batch` | 128 | 128 |
| `--token-budget` | 24576 | 16384 |
| `--max-micro` / `--pad-multiple` | 64 / 32 | 64 / 32 |
| `--max-tokens` | 4096 | 4096 |
| `--head-width` | 128 | 查询头 128；MLP 按参数量自动近似匹配 |
| `--patience` / `--min-delta` | 不使用 | 8 / 0.0001 |
| `--cache-budget-gb` | 不使用 | 8 GiB |

`--data` 可指定另一个已整理且符合阶段来源约束的数据目录，仍需三集完整元数据和严格隔离；它不会自动筛选来源。`--descriptions` 指定五类描述 JSON，默认类型数据目录的 `descriptions.json`。`--extraction-gpu` 指定类型特征抽取设备，默认使用 `--gpus` 的第一张卡。默认 Qwen3.5-4B、查询宽度 128 时，MLP 自动宽度 275，对应 705,655 参数；两种查询头各 706,177 参数。

## 产物和维护范围

新实验目录保存 `protocol.json`、`registration.json`、`registered_code/`、`data_audit.json`、日志、各 arm/seed 的检查点和 `selection.json`、`test_barrier.json` 以及 `report.md`。二分类结构化报告为 `summary.json`，类型报告为 `report.json`。`training_status.json` 表示统一入口状态；类型阶段另有 `queue_status.json` 和每组 `status.json`。类型报告包含逐种子成绩、种子均值/样本标准差、同种子配对比较和分来源指标。

现有兼容库保留 `repaired_study/repaired_heads` 以加载二分类权重；`multilabel_study` 仅提供当前特征抽取、缓存与类别头构建，统一入口不会调用其旧多标签训练流程。旧 `repaired_queue`、手工 ready gate、生成式 SFT 入口与已归档实验不参与新训练。继承标签和已探索测试集的局限仍然存在，不能把不同数据范围的历史分数直接当作方法增益。
