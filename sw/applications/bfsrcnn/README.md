# BFSRCNN C Application

该目录是程序员使用通用 BIRA 推理引擎手写模型推理的示例，不是生成目录。

`bfsrcnn_inference.c` 直接定义 Tensor 和 14 层算子，并引用生成的 `bfsrcnn_bira_data.h`。`bfsrcnn_infer()` 一次将全部 14 层传入 `bira_inference()`；`bfsrcnn_infer_layer()` 将一层传入同一个接口。通用引擎自动决定边界 DMA、SPAD 地址、中间结果驻留和可执行 ISA 序列。

参数导出由 `models/bfsrcnn/deployment/export.py` 完成，它只生成权重和 Parameter Buffer 数组。完整网络的裸机调用示例位于 `applications/baremetal/bira_bfsrcnn.c`。
