# 在 musacomm 仓库中工作

按任务选择并完整阅读对应 skill：

- 安装、生成命令或计划、运行测试、查看结果、排查运行错误：
  [musacomm-usage](skills/musacomm-usage/SKILL.md)。
- 修改源码、参数、统计、测试、文档或打包：
  [musacomm-development](skills/musacomm-development/SKILL.md)。
- 同时涉及使用与开发时，先用 usage 确认测量需求，再用 development 实现修改。

这些是仓库内的说明文件；从本仓库读取，不要求安装到个人全局目录。
下文及 skill 中的 shell 示例均从仓库根目录执行。

保持目录整洁：CPU 检查使用 `PYTHONDONTWRITEBYTECODE=1` / `python3 -B`，
临时计划、环境、构建包和测试输出放在仓库外。保留已有源码、测试、文档和 Git 元数据。
