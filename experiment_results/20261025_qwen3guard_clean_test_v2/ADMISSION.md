# 已授权的共享显卡调度修订

用户于 2026-10-25 明确允许共享其他用户正在使用但有足够剩余显存的 GPU。

原始 protocol.json、queue_registration.json、测试数据、官方提示模板、BF16 精度、解码设置、固定 batch 计划及评分规则保持原字节。新增 shared_admission_amendment.json 绑定原登记与三份执行源码。

入场要求至少 22 GiB 剩余显存；PyTorch caching allocator 限制为 14 GiB，另有 8 GiB 入场余量。该限制不是整个进程的绝对显存上限，不包含全部驱动与工作空间内存。原协议已允许的 OOM 拆批保留并记录。所有符合容量条件的卡均可并行，每卡最多一个本次评测 worker。

其他用户进程可以在我们入场前已存在。状态中的进程观察事件应结合 shared_worker_launch.admission_inventory 判断，不能据此声称其他进程后来进入。共享卡计时仅用于运行进度估算，不作为独占卡效率基准。

恢复已登记作业：

```bash
.venv-hanguard/bin/python scripts/hanguard/qwen3guard_shared_runner.py run
```

同一输出目录只允许一个协调器；检查 queue_state.json 和进程身份后恢复，不能重复启动。
