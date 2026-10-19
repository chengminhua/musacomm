# 运行依赖

GPU 测试使用以下组件：

- MUSA 驱动、Runtime、MCCL 与安装版 `mccl.h`；
- PyTorch、torch_musa，以及其运行时依赖；
- `mthreads-gmi` 诊断程序。

ctypes 适配器读取本机头文件核对 ABI，然后加载本机 MCCL。库和头文件的路径可通过
`--musa-home`、`--mccl-library` 和 `--mccl-header` 配置。

各组件的版本和许可证以对应安装包为准。
