# Chipyard 裸机验证

本目录程序覆盖 `RV64 → CUSTOM_3 RoCC → BIRA 调度 → TLB/PTW → TileLink DMA → 本地存储与计算阵列` 的完整路径。

- `bira_status`：STATUS、TLB_FLUSH、FENCE 和 Context。
- `bira_dma_loopback`：DRAM 到 Full SPAD 再返回 DRAM。
- `bira_dense_1x1`：signed A8×W8 与 Parameter Buffer。
- `bira_binary_1x1`：XNOR-popcount、correction、残差和二值后处理。
- `bira_bfsrcnn`：32×32 输入、14 层网络和 16 KiB golden。

```bash
make run
make run-bfsrcnn
```

默认 RISC-V 工具链来自 `<chipyard>/.conda-env/riscv-tools`，默认 simulator 是 `sims/verilator/simulator-chipyard.harness-BiRaRocketConfig`；可分别用 `RISCV=...` 与 `SIMULATOR=...` 覆盖。CPU 提交 DMA 前和读取 STORE 结果前需要 `fence rw, rw`，BIRA `FENCE` 只保证加速器任务完成，不能替代 CPU 内存屏障。完整流程见[使用指南](../../../docs/getting-started.md)。
