# Applications

`bfsrcnn/` 是程序员手写多层推理 C 的完整示例。Tensor 描述、14 层描述、整网调用和单层调用都位于 `bfsrcnn_inference.c`，不存在单独的模型图文件。

`baremetal/` 是 Chipyard/Rocket 裸机入口。量化 checkpoint 导出的参数 C 数组位于 `sw/build/bfsrcnn/generated/`，应用只引用参数，不生成网络拓扑或推理控制代码。
