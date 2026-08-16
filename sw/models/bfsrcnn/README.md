# BFSRCNN Model

本目录管理 BFSRCNN 的浮点网络、训练/QAT、整数化、参数导出、纯 C reference 与模型测试。程序员手写的推理代码位于 `sw/applications/bfsrcnn/bfsrcnn_inference.c`，模型目录不生成或保存推理拓扑。基础量化数学复用 `models/common/quantization`。

当前14层的 Tensor 连接、硬件计算循环和 PixelShuffle/final 特殊布局见[BFSRCNN 硬件数据流与计算循环](../../../docs/models/bfsrcnn.md)。

当前部署精度为：

| 层 | 权重精度 |
|---|---:|
| head | W16 |
| shrink1 | W4 |
| shrink2 | W16 |
| shrink3 | W8 |
| mapping0～mapping7 | W1 |
| expand | W8 |
| final | W16 |

从 QAT checkpoint 重新量化时，输出文件不会自动覆盖仓库内置的回归检查点。可以先生成到 `build/`，验证后再明确指定该检查点进行参数导出：

```bash
cd generators/bira/sw
make bfsrcnn-quantize \
  QUANTIZED_CHECKPOINT="$PWD/build/bfsrcnn/quantized/bfsrcnn_integer.pth"

make bfsrcnn-generate \
  INTEGER_CHECKPOINT="$PWD/build/bfsrcnn/quantized/bfsrcnn_integer.pth"
```

`bfsrcnn-quantize` 当前显式传入 `--shrink3-weight-bits 8` 和 `--expand-weight-bits 8`。如果量化策略发生变化，必须重新生成检查点、参数头文件和仿真 fixture，不能仅修改推理层描述中的 `weight_precision`。

部署导出：

```bash
cd generators/bira/sw
python3 -m models.bfsrcnn.deployment.export \
  --checkpoint models/bfsrcnn/tests/fixtures/bfsrcnn_integer_v4.pth \
  --output-dir build/bfsrcnn/generated \
  --height 32 \
  --width 32
```

推荐直接按[使用指南](../../../docs/getting-started.md)运行 `make bfsrcnn-generate`、`make bfsrcnn-fixture` 或 `make bfsrcnn-verilator`。
