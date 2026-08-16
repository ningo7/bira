# Hardware Source Layers

| 目录 | 内容 | 允许依赖 |
|---|---|---|
| `common/` | 参数、ISA、Bundle 与公共接口 | Chisel 基础库 |
| `control/` | 命令前端、Context、Reservation Station 与调度 | `common` |
| `dma/` | 二维 Load/Store、页和 beat 切分 | `common` |
| `memory/` | SPAD、Accumulator、Parameter Buffer | `common` |
| `compute/` | 阵列、卷积控制、取数、插值与后处理 | `common`、`memory` 接口 |
| `top/` | Core、DataPlane、StandaloneTop 和 RTL 生成入口 | 上述所有层 |

Rocket TLB/PTW、TileLink DMA 和 LazyRoCC 不放在这里，而位于 `chipyard/src/main/scala/bira/`。

模型专用验证放在 `hw/src/test/scala/bira/models/<model>/`，不混入通用硬件源码。

完整模块说明见[`docs/hardware.md`](../../../../../docs/hardware.md)。
