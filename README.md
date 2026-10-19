# musacomm

[English](README.en.md)

**MUSA 通信基准测试工具**。`musa + communication`：用一条命令生成测试计划，
调用 MCCL 通信原语，检查结果正确性，并保留可复核的逐 rank 计时和带宽记录。

## 安装

运行环境：Linux、Python 3.10+，单机 8 张 MUSA GPU。在源码根目录执行：

```bash
python -m pip install --no-deps .
musacomm --help
musacomm --version
```

也可以使用 `python -m musacomm`。CLI 帮助、计划生成和 CPU 单测不依赖 GPU 或 PyTorch。
GPU 测试需要安装匹配的驱动、MUSA SDK/MCCL、PyTorch 和 `torch_musa`。
默认 SDK 根目录为 `MUSA_HOME`，未设置时用 `/usr/local/musa`，
也可用 `--musa-home`、`--mccl-library`、`--mccl-header` 指定。

## 先生成计划

以下命令不初始化 GPU、不执行通信，只写出计划：

```bash
musacomm allreduce --dry-run --dtype bf16 \
  --sizes 1MiB,16MiB,256MiB,1GiB --output plan-allreduce
```

## 运行测试

先在你们的调度系统中申请整机独占，再在具有全部物理 GPU 视图的 Linux 宿主环境运行：

```bash
musacomm allreduce --dtype bf16 --reduce-op sum \
  --min-size 1MiB --max-size 1GiB --output results/allreduce

musacomm alltoall --dtype bf16 \
  --sizes 1MiB,64MiB,1GiB --output results/alltoall

musacomm sendrecv --src 0 --dst 1 --direction bidirectional \
  --sizes 1MiB,64MiB,1GiB --output results/p2p
```

工具会检查 GPU 进程、显存和利用率；遇到占用或异常会停止本次测试。
容器运行与诊断选项见[使用方法](docs/usage.md)。

## 测什么

| CLI 原语 | 当前执行路径 / 范围 |
|---|---|
| `allreduce`、`allgather`、`reducescatter` | MCCL 原生 API |
| `broadcast`、`reduce` | MCCL 原生 API，可选择 root |
| `alltoall`、`gather`、`scatter` | grouped Send/Recv 组合路径 |
| `alltoallv` | 等量路由；可选 grouped、torch、native |
| `sendrecv` | 指定 rank 对，单向或双向 |

支持 BF16/FP16/FP32。归约可选 SUM/MIN/MAX/PROD/AVG，每次运行都会检查结果正确性。
默认一次运行对应一种 buffer mode；适用原语可选 OOP/IP。

## 如何读结果

默认至少 20 次预热、5 个计时块，每块至少 200 ms；完整检查各 rank 的输入/输出。
设备事件计时与主机完成时间分开记录，先取各 rank 的最大耗时，再对计时块取中位数。

- `algbw`：约定逻辑 payload / 操作耗时。
- `busbw`：按通信原语归一化的逻辑带宽。8-rank AllReduce 为 `algbw × 1.75`。
- 上述带宽按算子的逻辑字节量计算；物理 MTLink 流量属于另一种计量口径。

每次运行写出 `cases.jsonl`、`config.json`、`preflight.json`、`launcher.log`、
`results/rank-*.jsonl` 和 `audit.json`，正常模式另保存 `monitoring.jsonl`。
终端显示 JSON 格式的运行摘要；干净运行且全部 rank 验证通过后，逐点数据导出为
`results.csv`。输出目录不允许覆盖。原始结果含环境信息，默认不纳入版本控制。

## 文档与开发

- AI 使用：[musacomm-usage](skills/musacomm-usage/SKILL.md)。
- AI 开发：[musacomm-development](skills/musacomm-development/SKILL.md)。
- 仓库内 AI 任务入口：[AGENTS.md](AGENTS.md)。
- [使用方法与参数](docs/usage.md)
- [计时与带宽计算](docs/methodology.md)
- [开发、测试与发布检查](docs/development.md)
- [测试记录](docs/validation.md)
- [贡献指南](CONTRIBUTING.md) · [安全与数据处理](SECURITY.md)
- [运行依赖](docs/third-party.md)

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/check_public_tree.py
```

CI 运行 CPU 测试和打包检查。

## 许可证

项目代码采用 [MIT](LICENSE)。
