# BIRA Hardware

本目录保存 Rocket-independent Chisel 硬件、ChiselTest 和独立 Verilator 环境。硬件模块边界、默认参数与支持范围统一记录在[硬件架构](../docs/hardware.md)，构建和测试命令统一记录在[使用指南](../docs/getting-started.md)与[测试指南](../docs/testing.md)。

常用命令：

```bash
sbt test:compile
sbt test
cd sim/models/bfsrcnn && make run
```

生成 RTL 默认进入 `build/generated-rtl/`，测试与仿真产物进入 `build/`，不应提交到源码目录。
