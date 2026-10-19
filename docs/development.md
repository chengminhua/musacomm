# 开发与发布检查

## 结构

```text
src/musacomm/
  cli.py                 # 参数、计划、启动、审计
  safety.py              # 空闲/进程检查与本任务清理
  catalog/               # 确定性 case、schema、CPU 语义
  runtime/collective.py   # MCCL 执行器，运行时才导入 torch
  runtime/helpers.py     # oracle、内存预算、计时汇总
  exporter.py            # 完整结果门禁与 CSV
tests/                   # CPU-only 回归
scripts/check_public_tree.py
.github/workflows/ci.yml  # CPU 测试、打包、wheel 安装
```

开发安装与自检：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/check_public_tree.py
musacomm allreduce --dry-run --sizes 1MiB,16MiB,1GiB --output plan-check
```

`--help`、`--version` 和 `--dry-run` 不应导入 torch、不初始化设备、不连接服务器。
运行新测试前先确认它是否属于 CPU-only；不要给 CI 接入生产凭证。

## 构建

在构建环境安装 `build`，然后：

```bash
python -m build
python scripts/normalize_sdist.py dist/musacomm-*.tar.gz
```

第二条命令去除源码归档中的本地 UID/GID、用户名/组名和时间戳，保留源码内容。
wheel/sdist 生成在 `dist/`。

项目使用 `pyproject.toml` 和 SPDX 许可证元数据，结构参考
[PyPA 打包指南](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)。

将 wheel 安装到另一个空虚拟环境，并从仓库之外测试 `musacomm --help` 和 dry-run，
检查独立安装是否完整。源码包也应从解包后的独立目录重新构建。

## 提交或发布前

- 所有 CPU 测试、公开内容扫描和 wheel/sdist 检查通过。
- 检查 `git status` / `git diff --cached` 中的待提交文件。
- 不提交内部地址、账号、路径、容器名、原始日志、SDK/header/binary 或令牌。
- 分享测试结果前去除环境、人员和组织信息。
- 核对 LICENSE、依赖、版本号和 CHANGELOG。
- 测试记录分别列出 CPU 测试、安装检查和 GPU 实测结果。
