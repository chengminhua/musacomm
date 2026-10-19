---
name: musacomm-usage
description: 使用 musacomm 安装、生成通信测试计划、运行已获授权的单机测试、解读 CSV/JSON 结果或排查运行错误。修改源码与测试时使用 musacomm-development。
---

# 使用 musacomm

## 先确定任务

以包含 `pyproject.toml` 和 `src/musacomm/` 的目录为仓库根目录，以下路径相对于它。
先区分用户要的是命令说明、CPU 计划、GPU 测试，还是已有结果解读；按该范围操作。
修改实现时转到[开发 skill](../musacomm-development/SKILL.md)。

- 安装与基础参数：读[README](../../README.md)和[使用方法](../../docs/usage.md)。
- 解读或对比性能：读[测量方法](../../docs/methodology.md)。
- 文档与行为有差异时，核对 `src/musacomm/cli.py` 的 `build_parser`、`validate_args`、
  `make_manifest`，以及 `src/musacomm/exporter.py`；不要猜测不存在的命令。

## 入口与环境

Linux、Python 3.10+；GPU 执行使用单机 8 个 rank，一进程一卡。先检查目标运行时中
的 `musacomm --help` / `--version`。用户要求安装时，在选定 Python 环境执行：

```bash
python3 -m pip install --no-deps .
```

GPU 运行需要匹配的驱动、MUSA SDK/MCCL、PyTorch/torch_musa。通过 `MUSA_HOME`、
`--musa-home` 或库/头文件选项选择已安装运行时，不为排错自动升级驱动或框架。
源码方式运行须使用**绝对** `PYTHONPATH`：执行器会将工作目录切到输出目录，相对
`PYTHONPATH=src` 会导致子进程找不到包。CLI 自带 torchrun，不再外套一层 torchrun。

## 生成 CPU 计划

只问用法或计划时不启动 GPU。这个示例在仓库外创建计划，不需要 torch/MUSA：

```bash
plan_root=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -B -m musacomm allreduce \
  --dry-run --dtype bf16 --sizes 1MiB,16MiB,1GiB --output "$plan_root/allreduce"
```

确认生成的 `cases.jsonl` 数量、原语、dtype、root、缓冲区模式与请求相符。
dry-run 只生成 `cases.jsonl`、`config.json`、`preflight.json`，不生成 `audit.json` 或性能 CSV。
`--output` 必须是尚不存在的目录，所以不要直接把已创建的 `plan_root` 当输出目录。
交付生成文件的实际路径；不要把 dry-run 的退出码 0 解释成 GPU 测试通过。

## 配置测试

- `--gpus` 是 8 个不同的宿主物理编号；`--roots`、`--src`、`--dst` 是 rank 0–7。
- 默认尺寸为 1 MiB 到 1 GiB 倍增。字节解析器认识 MB/GB，但执行计划只接受目录中的
  精确尺寸；3 MiB、任意十进制大小或大于目录上限的点可能被拒绝，不能静默替换。
- AllToAll/Gather/Scatter 使用 grouped Send/Recv；AllToAllv 才可选择
  `--backend grouped|torch|native`，当前 CLI 使用等量路由。
- 多 root、多归约可能超过 `--max-cases`。核对期望覆盖后提高上限；只有用户接受
  部分覆盖时才用 `--limit` 或 `--allow-truncate`。
- 强制 Ring/协议前确认安装版变量形式：默认 modern，旧数值变量需显式 legacy。
  实际算法查看 MCCL 日志；`--p2p off` 禁用软件 P2P 路径，`--p2p mtl` 选择路径。
- 对比测试时同时核对 `config.json` 中保留的 MCCL 环境变量和日志级别。

## 执行与排错

用户授权 GPU 测试后，按调用环境的资源预约流程取得整机独占，再核对物理卡映射和
宿主状态。只读空闲快照与本地文件锁不是预约记录。资源、依赖或映射不明确时停止启动，
说明缺少的条件；不停止外部 GPU 作业。

正常模式在宿主机运行。遇到繁忙、遥测错误或容器命名空间问题，先查 `preflight.json`、
`monitoring.jsonl`、`launcher.log` 和 `audit.json`。不要为了得到成功状态自动加跳过检查
的选项。用户明确要求诊断时才用 `--skip-preflight` / `--allow-busy`，保留其诊断状态。

| 退出码 | 处理 |
|---|---|
| 0 | 区分计划/帮助与实测；正式结果还要确认 `audit.status=passed` |
| 2 | 参数、执行、监控、清理或导出失败；按日志定位 |
| 3 | `non_comparable` 诊断结束，不产生正常 CSV |
| 124 / 130 | 超时 / 用户中断；确认本任务进程结束后再释放资源 |

排错停留在当前授权范围。修改代码、切换硬件/运行时或扩大实验矩阵属于新的处理选择，
不要用修改审计文件或手工导出的 CSV 覆盖失败/诊断状态。

## 读取和展示结果

当前终端输出审计 JSON，正式通过后生成 `results.csv`，逐 rank 明细在 `results/`。
没有内置 HTML/PDF/Markdown 报告命令；整理已有数据成表格时说明数据来源和样本范围。

- 对照 `cases.jsonl`、配置、审计与全部 8 个 rank 记录，区分计划、完整通过、诊断和失败。
- CSV 的 `median_max_rank_device_us` 为设备时间；host-completion 为另一计时口径。
- `algbw` 为约定逻辑 payload/t，8-rank AllReduce 的 `busbw=algbw×1.75`。
  不同原语的 S 定义见测量方法；AllToAllv/P2P 使用对应有效 payload/方向列。
- `block_relative_range` 是比例值，展示百分比时乘 100；它是块间波动，不是逐操作尾延迟。
- 交付配置、逐尺寸结果、最高已测点与文件路径。历史记录不能改标为本次新实测。
