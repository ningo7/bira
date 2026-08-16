# BIRA C Runtime

应用包含 `bira.h`，在手写推理 C 中定义 Tensor 描述和若干层描述，然后一次调用 `bira_inference()`。传入一层就是单层执行，传入多层就是跨层优化执行。

推理引擎自动分析调用范围的输入输出边界、Tensor 生命周期、Full/Binary SPAD 地址、中间结果驻留、Bank 复用和常量加载，Runtime 再将结果转换为 CFG、LOAD、EXEC、STORE 和 FENCE 指令。

`bira_inference.h` 定义程序员填写的数据结构、工作区和统一推理入口，`bira_planner.c` 实现跨层优化，`bira_runtime.h/.c` 和 `bira_rocc.h` 负责 ISA，`bira_trace.h/.c` 提供 Host 仿真后端。Runtime 不生成也不保存模型专用 C 代码。

```bash
cd generators/bira/sw
make runtime-test
```
