# Standalone Verilator Simulations

本目录只作为独立 Verilator 仿真的分类入口，不包含模型专用 Makefile 或 harness。每个模型的仿真代码放在 `models/<model>/`，当前实现见 [BFSRCNN](models/bfsrcnn/README.md)。

通用被测顶层是 `hw/src/main/scala/bira/top/StandaloneTop.scala`，生成 RTL 进入 `hw/build/generated-rtl/`。模型目录负责选择 fixture、编译 harness 和解释模型阶段，不应把模型层名写入通用硬件。
