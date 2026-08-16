# Common Deployment Layouts

本目录保存模型参数生成函数内部复用的 BIRA 数据布局工具，不是独立的编译器或推理阶段。

- `parameter_layout.py` 将模型适配器给出的 bias、shift、threshold 等语义参数编码成 Parameter Buffer 原生行。
- `weight_layout.py` 提供 int16 byte plane 和二值通道等基础权重布局。
- `c_data.py` 提供 C 类型、标识符和数组文本生成函数。

模型专用的 checkpoint 字段映射、特殊层布局和最终文件组织仍由 `models/<model>/deployment/` 负责。应用和 Runtime 不导入本目录中的 Python 模块，只编译模型生成的最终 `.h/.c` 参数文件。
