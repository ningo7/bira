# Models

每个子目录是一个相对独立的模型工程，可以定义网络、数据集、训练、量化和参数导出，但不保存硬件推理流程，也不应直接依赖 SPAD 地址、DMA 或 RoCC 编码。程序员手写的 C 推理代码位于 `sw/applications/<model>/`。当前端到端模型是 [BFSRCNN](bfsrcnn/README.md)。
