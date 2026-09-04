# BIRA: A Unified Binary–Integer Reconfigurable Accelerator

**English** | [简体中文](README.zh-CN.md)

BIRA (Binary–Integer Reconfigurable Accelerator) is a Chisel-based accelerator for binary and low-bit neural networks. It targets edge image super-resolution and similar workloads constrained by compute capacity, storage, and memory bandwidth. Built around hardware–software co-design, BIRA supports binary XNOR-popcount and W2/W4/W8/W16 integer operations on a unified compute array. Configurable dataflows, on-chip memory planning, operator fusion, and implicit data reordering provide both general convolutional-network support and optimizations for super-resolution models.

BFSRCNN x4 is currently the most complete end-to-end example. It covers training and quantization, parameter export, a manually described network, runtime planning, standalone RTL simulation, and bare-metal execution in Chipyard. Neither the BIRA hardware nor its runtime is tied to BFSRCNN layer names or topology.

BIRA is an exploratory project under active development. It does not yet provide a comprehensive performance evaluation or a general-purpose compiler.

## Motivation

Edge image super-resolution is commonly built around convolutional neural networks. Deeper networks can improve reconstruction quality, but they also introduce substantial parameter storage, computation, and intermediate feature data, making it difficult for constrained devices to meet latency, power, and memory requirements simultaneously. Binary and low-bit quantization can significantly compress a model and replace multiply–accumulate operations with simpler bitwise and integer operations. However, general-purpose processors and conventional high-precision accelerators cannot naturally exploit these benefits: bit packing and format conversion add overhead, while the large difference in computational intensity between binary layers and the integer layers at the beginning and end of a network can leave hardware resources underutilized.

BIRA therefore follows a dedicated-accelerator approach without reducing the hardware to a fixed pipeline for one network. It addresses three related questions:

- How can binary layers and integer layers with different weight precisions efficiently share the same physical compute resources?
- How can data layout, address generation, and post-processing fusion reduce data movement in super-resolution networks?
- How can software describe a network and automatically perform cross-layer on-chip memory planning and ISA command submission?

## Design Principles

### General Mechanisms with Workload-Specific Optimizations

BIRA provides general Conv and FC computation modes and represents networks through tensor and layer descriptors. Super-resolution features such as PixelShuffle layouts, bilinear residuals, and fused binary post-processing are built on configurable address-generation options, post-processing modes, and deployment software.

### Configurable Precision and Dataflow

The default array supports A8 activations, binary weights, and W2/W4/W8/W16 integer weights. Precision determines not only the data width, but also operand grouping, adder-tree entry points, and data distribution inside the array, allowing one physical array to support both integer and binary convolution.

### Global Decisions in Software, Efficient Execution in Hardware

Model tools perform quantization and parameter layout. Developers describe tensors and operators with public data structures, after which the inference engine analyzes tensor lifetimes, allocates on-chip addresses, arranges bank ping-pong, and removes unnecessary inter-layer DMA transfers. Hardware receives the planned contexts and dynamic tasks, then performs dependency checking, scheduling, data movement, computation, and post-processing.

## Architecture Overview

![BIRA architecture overview](img/architecture-overview.png)

*BIRA combines control and DMA, a unified binary–integer compute array, scalar post-processing, accumulators, and Full/Binary scratchpads.*

BIRA draws on Gemmini's RoCC integration, Decoupled Access/Execute organization, explicitly managed scratchpad memory, and the verification workflow of combining standalone simulation with full-system SoC simulation. Its implementation includes the following features:

- Integration into Chipyard as a RoCC accelerator through Rocket `CUSTOM_3` instructions;
- Independent Load, Execute, and Store queues, with constrained out-of-order issue based on RAW, WAR, and WAW dependencies between on-chip banks;
- A core pipeline divided into Compute, Accumulator, and Post/writeback stages, with queues and backpressure allowing the stages to overlap;
- Current and prefetched weight buffers that hide weight-read latency at tile boundaries;
- Software-planned SPAD bank ping-pong between adjacent layers.

The standalone `StandaloneTop` exposes a simple memory interface and is intended for module verification, trace replay, and accelerator-internal cycle analysis. The Chipyard configuration adds Rocket, private TLB/PTW paths, and TileLink DMA for bare-metal software and full-system verification.

## Binary–Integer Compute Array

Binary neural networks still require integer layers for input processing, output generation, and feature transformation. Separate binary and integer arrays can alternate between active and idle periods because of workload imbalance, while forcing them into a fixed parallel pipeline complicates scheduling and load balancing. BIRA maps binary and integer operations onto the same configurable bit cells and reduction trees, allowing the physical compute resources to be reorganized for the precision and dataflow of the current layer.

The default array has 16 lanes, with 16 configurable bit cells in each column:

- In binary mode, each bit cell performs XNOR and the column feeds a popcount tree;
- In integer mode, each bit cell performs AND between an activation bit and a signed weight bit, with local results interpreted as two's-complement values;
- AND forms the common logic for both modes, while binary mode additionally counts `0 == 0` to produce XNOR;
- A8 activations enter the array as bit planes. W2/W4/W8/W16 weights form 8/4/2/1 operand groups within a column, and the controller shifts and combines their results according to activation-bit significance;
- The configurable adder tree accepts 1, 2, 4, 8, or 16 operands at the matching level, avoiding a separate full multiplier array for every supported precision.

![Binary compute mode](img/binary-compute-mode.png)

*Binary mode: bit cells perform XNOR and the full adder tree computes popcount.*

![A8×W2 compute mode](img/a8-w2-compute-mode.png)

*A8×W2 mode: adjacent bit cells form W2 operands and enter the adder tree at the matching level.*

![A8×W4 compute mode](img/a8-w4-compute-mode.png)

*A8×W4 mode: four bit cells form each W4 operand, reorganizing array parallelism for the selected precision.*

This design does not expand both activations and weights into a fully bit-serial computation. Weight signs and bit positions remain local to each column, while activation bit-plane results are reconstructed locally. The integer path therefore avoids physical multi-bit multipliers without incurring all the extra iterations of a fully serial design.

## Tensor Layout and Configurable Dataflows

![Tensor layout and configurable dataflows](img/tensor-layout-and-configurable-dataflows.png)

*HWC layout, configurable input distribution and reduction, and implicit data reordering for PixelShuffle.*

One Full SPAD row stores 16 consecutive channels, and logical tensors normally use an `HWC16`-like channel-blocked layout. The convolution controller computes local addresses directly from the output coordinate, kernel tap, and channel block. Taps outside the padding boundary do not generate a read, so the design does not require `im2col` or a window buffer.

The array follows a weight-stationary reuse strategy and adapts to different operators through activation distribution, weight organization, and output reduction:

| Mode | Input and weight organization | Meaning of an array column | Typical use |
|---|---|---|---|
| Dense | Broadcast scalar or grouped activations; organize weights by output channel | Output channel | Standard and 1×1 convolution |
| Depthwise | Map input lanes one-to-one onto compute columns | Input/output channel | Depthwise convolution |
| Column reduce | Distribute input channels across columns and reduce again at the output | Input channel | Final/reduction operators with many input channels and few output channels |
| Binary | Feed packed activations and weights into the XNOR-popcount path | Output channel or mapping group | Binary convolution |

For data rearrangements such as PixelShuffle, BIRA writes the preceding layer directly in the pixel-pair layout required by the following layer. The next address generator then reads it using high-resolution coordinates. The rearrangement is therefore expressed as write-address and channel-index transformations rather than as a standalone data-reordering pass.

## Algorithm–Hardware Co-Optimization

![Algorithm–hardware co-optimization](img/algorithm-hardware-co-optimization.png)

*Halo-aware cross-layer patch execution, fused post-processing and rebinarization, Scale–RPReLU fusion, and multiplier-free PoT/APoT scaling.*

### Halo-Aware Image Patches

Deployment software can divide a large image into patches with halos so that each invocation's input, intermediate features, and active parameters fit more readily in on-chip storage. The valid region is used during final stitching, while the halo supplies convolution context at patch boundaries at the cost of a small amount of redundant computation. Patch partitioning, halo selection, and output stitching are handled by software.

### Cross-Layer Residency and Fusion

`bira_inference()` can accept multiple layer descriptors in one call. The inference engine identifies the first and last use of every tensor, reuses storage between tensors with non-overlapping lifetimes, and keeps intermediate results from adjacent layers in the Full or Binary SPAD. For binary layers, post-processing can directly produce the packed binary output required by the next layer, avoiding storage of a high-precision intermediate tensor followed by a separate sign operation.

### Fused Binary Post-Processing

A bipolar binary dot product is:

```text
dot = 2 × popcount(XNOR(a, w)) - N
```

The array and Accumulator retain `2 × popcount`, while the Parameter Buffer supplies a correction equal to `-N` for the valid kernel and channel count. The post-processing stage applies this correction directly. Scaling, thresholding, RPReLU, residual addition, saturation, and optional rebinarization can continue along the same path, reducing standalone operators and intermediate writes.

### Fused Scale and PReLU/RPReLU

A binary convolution normally applies a per-channel scale and bias to its dot-product result before RPReLU. Implementing these operations separately would first materialize a scaled intermediate tensor and then execute a PReLU-like branch operation. During quantization, BIRA folds the binary-convolution scale, convolution bias, and RPReLU parameters into two affine branches that operate directly on the raw integer dot product `x`. Let `s` be the convolution scale, `b_c` the convolution bias, `b_0` and `b_1` the RPReLU offsets, and `α` the negative-branch slope. The fused form is:

```text
t = b_0 + s × b_c

x >= -t / s : y = s × x       + t     + b_1
x <  -t / s : y = α × s × x   + α × t + b_1
```

The quantization tools precompute the branch threshold and the scale and bias of each affine branch, then write them into the Parameter Buffer. Hardware selects a branch from the threshold and applies it directly to `2 × popcount - N`, followed by residual addition, saturation, and optional sign output. Convolution scaling, RPReLU, and state update therefore execute in one post-processing pipeline without a standalone scale/PReLU operator or a scaled intermediate tensor. PReLU for ordinary integer layers is likewise performed by `IntPostProc` after convolution accumulation, with positive and negative branch coefficients implemented using shifts and additions.

### Multiplier-Free Fixed-Point Scaling

The quantization tools approximate multiplicative scales with power-of-two (PoT) values and two-term additive powers of two (APoT). Hardware implements multi-bit requantization and binary RPReLU branches with shifts and additions/subtractions, so neither the compute path nor the post-processing path requires a general multi-bit multiplier.

## ISA and Decoupled Execution

BIRA uses Rocket's `CUSTOM_3` opcode and separates static layer configuration from dynamic execution tasks:

| Instruction category | Purpose |
|---|---|
| `CFG_SHAPE` | Configure input/output shapes, kernel, and padding |
| `CFG_ADDR` | Configure local row addresses in the SPAD, Accumulator, and Parameter Buffer |
| `CFG_MODE` | Configure array mode, weight precision, signedness, and post-processing options |
| `CFG_COMMIT` | Freeze the configuration into an immutable Context snapshot |
| `LOAD_2D` / `STORE_2D` | Submit two-dimensional transfers between external and on-chip memory |
| `EXEC` | Submit a convolution task using a selected Context |
| `FENCE` / `STATUS` / `TLB_FLUSH` | Provide completion synchronization, status reporting, and address-translation maintenance |

Load, Execute, and Store enter independent queues. The scheduler checks dependencies using the bank read/write sets declared by each Context and allows younger, independent tasks to bypass an older blocked task. Context snapshots ensure that later configuration commands cannot change the semantics of queued tasks, while `FENCE` establishes a software-visible completion boundary.

External DRAM addresses are byte addresses, while local SPAD, Accumulator, and Parameter Buffer addresses are expressed in rows. Each 64-byte Parameter Buffer record contains bias, requantization, threshold, correction, and other post-processing parameters.

## Software Stack

BIRA separates model-specific export from the general inference engine and runtime:

| Stage | Developer input | Tool/runtime output |
|---|---|---|
| Training and quantization | Model, data, and quantization configuration | Quantized checkpoint |
| Parameter export | Checkpoint and model adapter | C weights, biases, quantization parameters, and Parameter Buffer arrays |
| Network description | Manually written tensor and layer descriptors | Operator properties and inter-layer connections |
| Cross-layer planning | One or more layers passed to `bira_inference()` | Shape validation, lifetime analysis, SPAD allocation, DMA planning, and bank ping-pong |
| Instruction execution | External input/output bindings | Runtime-generated CFG, LOAD, EXEC, STORE, and FENCE commands |

The parameter exporter does not generate network topology or inference C code. Developers do not manually assign SPAD addresses, encode RoCC commands, or write individual ISA instructions. Passing one layer to the same API supports operator-level validation; passing the full layer array enables cross-layer optimization over the complete network.

The runtime offers two backends. The RoCC Driver executes real RV64 instructions in Chipyard, while the Trace Driver records the same commands and external-memory image on the host for replay by ChiselTest and standalone Verilator. A software-generated execution plan can therefore be reused across fast module-level verification and full SoC verification.

## Current Scope

- Default 16-lane array, A8 activations, and 32-bit Accumulator;
- Binary and W2/W4/W8/W16 weight paths;
- Dense, Depthwise, Column-reduce, and Binary dataflows;
- Full/Binary SPADs, Accumulator, and Parameter Buffer;
- Decoupled Load/Execute/Store, bank dependency checking, and constrained out-of-order issue;
- Weight prefetching, overlapped compute/accumulate/post-processing, and cross-layer SPAD bank ping-pong;
- Fused binary post-processing, PoT/APoT fixed-point scaling, implicit PixelShuffle, and bilinear residuals;
- Three levels of verification: ChiselTest, standalone Verilator, and Chipyard Verilator.

## Verification Results

BIRA uses layered verification across software unit tests, ChiselTest, standalone Verilator, and Chipyard Verilator. The recorded BFSRCNN x4 test uses a 32×32 grayscale input and produces a 128×128 output. The optimized standalone RTL passes the 16,384-byte golden-output check. A current-version Chipyard end-to-end result has not yet been measured.

| Result | Value |
|---|---:|
| Latest recorded optimized standalone BIRA RTL, 14-layer effective inference | 4,602,938 cycles |
| Standalone latency at an assumed 200 MHz | 23.015 ms |
| Standalone theoretical frame rate at an assumed 200 MHz | 43.45 FPS |
| Current-version Chipyard `bfsrcnn_infer()` | Pending remeasurement |

## Quick Start

The commands below assume they are run from the Chipyard root and that BIRA is located at `generators/bira`.

Run the software tests:

```bash
cd generators/bira/sw
make test
```

Run the hardware tests:

```bash
cd generators/bira/hw
sbt test
```

Generate BFSRCNN parameters and fixtures and run the standalone end-to-end Verilator simulation:

```bash
cd generators/bira/sw
make bfsrcnn-verilator
```

Generate only the standalone top-level RTL:

```bash
cd generators/bira/hw
sbt "runMain bira.GenStandaloneTop build/generated-rtl"
```

## Repository Layout

```text
bira/
├── hw/          Generic Chisel RTL, ChiselTest, and standalone Verilator
├── sw/          Quantization/deployment tools, C runtime, model applications, and simulation utilities
├── chipyard/    LazyRoCC, TLB/PTW, TileLink DMA, and SoC configuration
├── patches/     Build-system patches for specific Chipyard versions
└── scripts/     Chipyard installation, validation, and removal scripts
```

## Roadmap

- [ ] Optimize hardware details, critical paths, and resource utilization;
- [ ] Establish consistent performance, area, power, and model-quality evaluation records;
- [ ] Expand operator coverage, model adapters, and deployment automation to improve software-stack generality;
- [ ] Explore multi-array architectures, NoCs, and a supporting simulator for larger-scale computation and system expansion.
