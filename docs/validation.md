# 0.1.0 本地发布准备验证

验证日期：2026-10-19。环境：Linux aarch64 / Python 3.12。

| 项目 | 结果与范围 |
|---|---|
| CPU 回归 | 87 项通过；包括 case/schema、oracle、计时统计、完整 rank 导出、安全状态、公开内容扫描和归档元数据 |
| CLI | `--version`、`--help` 和 CPU dry-run 通过 |
| 构建 | wheel / sdist 成功；wheel 从 sdist 构建 |
| 独立安装 | 在仓库外、没有 torch/torch_musa 的新虚拟环境中安装 wheel，通过包入口与模块入口检查 |
| 原语计划 | 10 类原语，各 1 MiB 和 1 GiB 两点，生成计划并经 runtime `--validate-only` 通过 |
| 发布扫描 | 候选源码与构建产物检查通过；源码包所有者元数据已归一化 |
| GPU 测试 | 本次未运行 |

GitHub CI 配置为 Python 3.10–3.13；本地测试使用 Python 3.12，远程 CI 尚未运行。
