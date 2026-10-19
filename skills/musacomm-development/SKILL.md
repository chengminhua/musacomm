---
name: musacomm-development
description: 开发 musacomm 仓库，修改命令行、测试目录、MCCL 执行器、统计导出、安全检查、文档或打包，并执行对应 CPU 回归。仅运行或解读测试时使用 musacomm-usage。
---

# 开发 musacomm

## 定位改动

从包含 `pyproject.toml` 和 `src/musacomm/` 的仓库根目录工作。先读相关实现和已有测试，
明确用户要求改变的行为；只做诊断或评审时不实施修复。需要运行/解读通信测试时另读
[使用 skill](../musacomm-usage/SKILL.md)。

按改动选择入口，不必通读全部目录：

| 改动 | 实现位置 | 对应测试 |
|---|---|---|
| CLI、参数、计划、启动和审计 | `src/musacomm/cli.py` | `tests/test_cli.py` |
| case/schema/状态 | `src/musacomm/catalog/` | `tests/test_catalog.py`、runtime/exporter 测试 |
| MCCL 调用、校验、计时、oracle | `src/musacomm/runtime/` | `tests/test_runtime.py`、`tests/test_exporter.py` |
| CSV 与完整性检查 | `src/musacomm/exporter.py` | `tests/test_exporter.py` |
| GPU 状态、PID、进程组清理 | `src/musacomm/safety.py` | `tests/test_safety.py`、`tests/test_cli.py` |
| 公开内容扫描 / 源码包归一化 | `scripts/check_public_tree.py` / `scripts/normalize_sdist.py` | `tests/test_public_tree.py` / `tests/test_release.py` |
| 安装、发布、CI、仓库 skill | `pyproject.toml`、`MANIFEST.in`、`.github/`、`skills/` | 公开扫描、相关 CPU 测试与按需包安装检查 |

参数说明见[使用方法](../../docs/usage.md)，计时/字节定义见
[测量方法](../../docs/methodology.md)，构建流程见[开发文档](../../docs/development.md)。

## 保持一致的实现约定

修改触及对应部分时检查以下约定：

- `import musacomm`、帮助、dry-run 和 CPU 测试不导入 torch、不初始化 GPU。
  GPU 运行时依赖由目标环境提供，不把它们加成普通安装必拉取的依赖。
- CLI 生成确定性 manifest；显式参数必须传到执行器，尺寸、root、归约组合不能
  因一次性迭代器或过滤条件而漏项。保留 case identity、实际实现与选择数量。
- 公开 GPU 执行固定单机 8 rank。catalog 中 AB16/MoE 等规划结构不是已经开放的
  CLI 功能；新增支持需同时更新参数、绑定、执行、导出和测试。
- 控制通信使用 CPU/Gloo，计时使用 MUSA 事件并等设备完成。分配、填充、校验和
  日志不进入通信计时窗口；device 与 host-completion 分开统计。
- 正式计时至少 20 次预热、5 块、每块最大 rank 设备窗口至少 200 ms。
  每块先取最慢 rank，再取块中位数；校验覆盖全部 rank 的前、计时输出和后检查。
- 按原语定义逻辑 payload，AllReduce 的 1.75 系数只乘一次。
  P2P 方向与 AllToAllv 有效 payload 分别处理；修改统计需同时检查导出列。
- 明确指定 grouped 路径时不能被新增 native 符号覆盖。失败记录与正常结果分开；
  导出要核对完整 8-rank membership、manifest、配置、正确性与原始计时汇总。
- GMI 进程表是 GPU_ID、PID 两列；保留未知/失败遥测与容器 PID 映射检查。
  只清理本次创建的进程组；诊断模式保留 `non_comparable` 与退出码 3，导出失败非零。

先补充能复现旧问题的行为测试，再修改实现。使用 CPU fixture/mock 验证参数、ABI
调用形状、状态流和统计，不依赖本机 GPU 来跑普通回归。

## 验证

运行与改动相关的测试，再视范围执行全量 CPU 测试；以下命令不会在仓库生成字节码：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -B -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/check_public_tree.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -B -m musacomm --help
```

需要计划验证时使用仓库外临时目录，输出指向其**尚不存在的子目录**：

```bash
plan_root=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -B -m musacomm allreduce \
  --dry-run --sizes 1MiB,16MiB,1GiB --output "$plan_root/allreduce"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -B -m musacomm.runtime.collective \
  --validate-only --cases "$plan_root/allreduce/cases.jsonl"
```

构建/依赖/入口发生变化时，从仓库外临时副本构建 wheel/sdist，在另一个临时 Python
环境安装 wheel，并切离源码目录测试入口。源码包要包含 `AGENTS.md` 和 `skills/`；
新增顶层目录也要检查公开扫描白名单，skill 目录不能被结果目录的名称前缀跳过。
同时用 `git check-ignore` 核对新 skill 未被 `.gitignore` 排除。

GPU 复验只在用户授权并取得调用环境要求的整机预约后进行。交付时分别列出 CPU
回归、安装/计划验证、实际 GPU 执行，不能用 mock 或历史结果替代本次设备验证。

## 收尾

- 简要说明修改文件、行为变化、实际执行的检查与结果。
- 检查 Markdown 相对链接、两个 skill 的文件名/名称与 UI metadata 一致。
- 避免留下 `.venv`、`build`、`dist`、`*.egg-info`、`__pycache__` 或测试日志；
  这些产物应放在仓库外。清理只针对本次确认的生成物，不对未提交仓库执行 `git clean`。
- 文档保持功能、参数、结果导向，不加入重复的身份或成熟度声明。
- GitHub 推送、发布、全局 skill 安装等操作按用户当次要求处理。
