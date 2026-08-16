# BIRA Software

本目录保存通用 C 多层推理 Runtime、模型训练与量化代码、模型公共部署布局、程序员编写的 BFSRCNN 推理、仿真 fixture 工具和 Chipyard 裸机应用。

模块职责见[软件架构](../docs/software.md)，执行命令见[使用指南](../docs/getting-started.md)。

快速运行软件回归：

```bash
make test
```

生成并运行 32×32 BFSRCNN 独立 Verilator：

```bash
make bfsrcnn-verilator
```

所有可再生输出默认位于 `build/`。
