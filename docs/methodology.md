# 测量方法

## 正确性优先

测试使用确定性、可表示的输入模式与独立 CPU 参考。所有 rank 的通信前、计时输出、
通信后校验均通过，才接受性能结果。正确性、准备输入、分配缓冲区和写日志不属于
通信计时窗口。CPU 控制组使用 Gloo；计时工作使用 MCCL 与显式 MUSA stream。

CPU 单测覆盖参数、计划、oracle、统计、导出与安全状态机；GPU 测试检查实际通信结果。

## 计时

- 预热至少 20 次，然后用 pilot 估计每块迭代数。
- 至少 5 个计时块；每块的最大 rank 设备事件窗口达到 200 ms，必要时增加迭代数。
- 每块取各 rank 的最大每次操作时间，再对各块取中位数。
- 原始记录保留 device-event 和 host-completion 两种口径，CSV 主带宽使用前者。
- 块间波动由各块平均时间计算；逐操作 P95/P99 需要另外采集单次耗时。

与官方 perf 工具对照时，应对齐库、GPU 集合、dtype、reduce op、buffer mode、消息
字节、算法/协议请求、计时域和同步方式。单进程多卡与多进程一卡、固定次数与长窗口
统计应分别注明，方便解释结果差异。

## 字节和带宽

`GB/s = bytes / seconds / 1e9`。消息 `GiB = 2^30 bytes`。

| 原语 | CLI 大小 S 的含义 |
|---|---|
| AllReduce | 每 rank 输入/输出字节 |
| AllGather | 每 rank 收到的完整输出字节 |
| ReduceScatter | 每 rank 的完整输入字节 |
| Broadcast / Reduce | 单个输入或结果消息字节 |
| AllToAll | 每 rank 总发送量，含 self 块 |
| Gather / Scatter | root 总接收/总发送量 |
| P2P | 每个方向的消息字节 |
| AllToAllv | 最大 rank 的总发送/接收量 Q，另记非自身 payload |

`algbw=S/t` 是约定的逻辑带宽；`busbw` 为归一化指标，例如 AllReduce 使用
`2*(N-1)/N`，8 ranks 为 1.75。跨原语比较时应先对齐 S 的含义。
AllToAllv 使用路由对应的有效 payload 指标。

上述指标统计逻辑字节；物理 MTLink/NIC 流量需要使用对应的硬件计数器采集。
判断大包是否达到带宽平台，需要比较连续多个尺寸点。

## 运行审计

原始文件保留 manifest/source/library hashes、rank/物理 GPU 映射、实际实现、设置、
正确性和计时块。正常 CSV 必须来自完整 8-rank、同一 manifest、全部通过的结果。
外部竞争、监控失败、超时和 rank 错误会记录为失败；跳过 preflight 或允许忙机时，
记录为 `non_comparable`。这些记录与正常 CSV 分开保存。
