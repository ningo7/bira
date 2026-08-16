# BFSRCNN Standalone Verilator Simulation

本目录保存 BFSRCNN 专用的 C++ trace harness 和 Makefile。快速回归默认读取仓库冻结的 4×4 fixture：

```bash
cd generators/bira/hw/sim/models/bfsrcnn
make run
```

运行软件生成的默认 32×32 fixture并输出逐层周期：

```bash
make profile
```

`make profile` 会调用 `sw/Makefile` 的 `bfsrcnn-fixture`，生成内容留在 `sw/build/bfsrcnn/`，不会复制到 `hw/` 源码目录。Verilator 编译产物位于 `hw/build/verilator/bfsrcnn/`，Chisel 生成的 RTL 位于 `hw/build/generated-rtl/`。

使用外部 fixture：

```bash
make run FIXTURE=/absolute/path/to/fixture
```
