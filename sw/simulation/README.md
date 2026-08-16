# Simulation Utilities

本目录负责生成和检查稳定的模型 fixture，不包含 RTL simulator。`bfsrcnn_fixture.py` 编译程序员手写的推理 C，通过 Trace Driver 生成命令、外部内存与 golden；`bfsrcnn_layer_api_test.c` 逐一把 14 层单独传入推理引擎；`command_codec.py` 校验 Python/RTL 编码。BFSRCNN Verilator harness 位于 `hw/sim/models/bfsrcnn/`。

```bash
cd generators/bira/sw
make bfsrcnn-fixture
make bfsrcnn-verilator
```
