# 使用方法

## 运行环境

工具采用单机 8 rank / 8 GPU，一进程一卡。`--gpus` 接受 8 个不同的宿主物理
编号，默认 `0,1,2,3,4,5,6,7`。`--src`、`--dst` 和 `--roots` 使用 **rank 编号**，
不是任意物理 GPU 编号。不要在已有 torchrun 作业内再次嵌套启动这个 CLI。

安装匹配的 MUSA SDK、MCCL、PyTorch/torch_musa。默认 SDK 根为 `MUSA_HOME` 或
`/usr/local/musa`。原生适配器会核对库和头文件的 ABI 常量及公开入口，遇到不匹配会报错。

```bash
musacomm allreduce --musa-home /opt/musa --dry-run --output plan-custom-sdk
```

## 常用选项

| 选项 | 用途 |
|---|---|
| `--dtype bf16\|fp16\|fp32` | 数据类型，默认 BF16 |
| `--sizes 1MiB,16MiB,1GiB` | 明确的消息尺寸列表 |
| `--min-size` / `--max-size` | 倍增尺寸范围，默认 1 MiB 到 1 GiB |
| `--reduce-op sum\|min\|max\|prod\|avg\|all` | 适用归约操作 |
| `--buffer-mode out_of_place\|in_place` | 适用原语的缓冲区模式 |
| `--roots 0` / `--roots all` | rooted collective 的 root |
| `--backend grouped\|torch\|native` | AllToAllv 执行路径；固定 AllToAll 仍为 grouped |
| `--warmup` / `--blocks` / `--block-seconds` | 至少 20 / 5 / 0.2 |
| `--pilot-iterations` | 自适应迭代数估计，默认 5 |
| `--output` | 新的结果目录，拒绝覆盖已有目录 |
| `--dry-run` | CPU 计划生成，无 GPU 通信 |

尺寸单位显式区分：`MiB/GiB` 是二进制，`MB/GB` 是十进制。带宽统一为十进制 GB/s。
可选尺寸以至 1 GiB 的 2 的幂为主，另有部分合法小元素点。工具会检查 dtype/rank
对齐、尺寸目录和显存预算，不符合条件的请求会报错。
`--limit` 和 `--allow-truncate` 用于选择部分测试点，选中数量会写入审计。

## 算法和传输开关

默认 `--algo auto --proto auto --p2p auto`，清除工具管理的继承选择变量，由 MCCL
自行选择。`--algo ring` 等选项用于指定算法；实际选择结果查看 MCCL 日志。
其余继承的 `MCCL_*` 配置保留在 `config.json` 中，对比两次测试时应同时核对这些设置。

`--mccl-env-style modern`（默认）使用 `MCCL_ALGO`/`MCCL_PROTO` 字符串变量；
`--mccl-env-style legacy` 使用历史数值变量。历史适配中 Ring=1、Tree=0、CollNet=2，
Simple=2、LL=0。根据安装版 SDK 文档选择变量形式；不兼容的设置会保留错误日志。

```bash
musacomm allreduce --dry-run --algo ring --proto simple \
  --mccl-env-style legacy --output plan-ring-legacy
```

`--p2p mtl` 选择 MTL 距离/路径；`--p2p off` 禁用软件 GPU P2P 路径，保留物理链路状态。
`--debug info|trace` 会增加日志开销，对照测试应使用相同日志级别。

## 安全检查与诊断模式

1. 测试前申请整机独占。
2. 正常模式在宿主机核对 `mthreads-gmi` 状态、进程表和物理 GPU 映射，检查失败时停止。
3. 本地文件锁协调使用同一锁文件的 musacomm 进程。
4. 超时、中断或检测到外部 GPU 进程时，工具清理本次启动的进程组。

容器里的 PID/设备映射可能不同于宿主机，默认检查会拒绝这种运行环境。
容器诊断可使用 `--skip-preflight`；它跳过全部 gmi 检查、运行监测和结束检查。
`--allow-busy` 则保留检查并允许已有 GPU 占用。这两种模式都输出 `non_comparable`，
诊断完成返回 3，并保留原始记录而不导出正常性能 CSV。

### 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 帮助/计划完成，或正常测试及严格 CSV 导出通过；通过 audit 区分 |
| 2 | 参数、执行、监控、清理或导出校验失败 |
| 3 | 显式诊断运行结束，`non_comparable`，不产生正常 CSV |
| 124 | 启动器超时 |
| 130 | 用户中断 |

自动化流程应同时检查退出码和 `audit.json`，保留逐 rank 记录供复核。

## 实现说明

- AllToAll/Gather/Scatter 使用 grouped Send/Recv，表格会写明实际实现。
- AllToAllv native 路径需要满足安装版的内存注册等前置条件，执行失败时记录错误原因。
- AllToAllv 使用等量路由，计时对象是通信操作。

结果字段和计算公式见[测量方法](methodology.md)。
