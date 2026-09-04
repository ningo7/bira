package bira

// Multi-cycle compiler for the static execution footprint of one context.

import chisel3._
import chisel3.util._

/** Converts a committed context into the bank-level footprint used by the
  * reservation station.
  *
  * An execution footprint is invariant for the lifetime of a committed
  * context.  Computing it once here removes the dimension/precision multiply
  * chain from every EXEC enqueue.  The explicit stages also keep at most one
  * variable multiply between registers so FPGA DSP pipeline registers can be
  * used effectively.
  */
class ResourceCompiler(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val input = Flipped(Decoupled(new Context(p)))
    val output = Decoupled(new ResourceSet(p))
  })

  private val idle :: dimensions :: rowProducts :: weightProducts :: buildMasks :: result :: Nil = Enum(6)
  private val state = RegInit(idle)
  private val context = Reg(new Context(p))

  private val inputPixels = Reg(UInt(32.W))
  private val outputPixels = Reg(UInt(32.W))
  private val kernelElements = Reg(UInt(8.W))
  private val inputBlocks = Reg(UInt(8.W))
  private val outputBlocks = Reg(UInt(8.W))
  private val weightInputFactor = Reg(UInt(16.W))

  private val inputRows = Reg(UInt(32.W))
  private val outputRows = Reg(UInt(32.W))
  private val binaryOutputRows = Reg(UInt(32.W))
  private val weightKernelBlocks = Reg(UInt(24.W))
  private val binaryWeightPartial = Reg(UInt(24.W))
  private val lowResidualPixels = Reg(UInt(32.W))

  private val multiWeightRows = Reg(UInt(32.W))
  private val binaryWeightRows = Reg(UInt(32.W))
  private val residualRows = Reg(UInt(32.W))
  private val accumulatorRows = Reg(UInt(32.W))
  private val resultResources = RegInit(
    0.U.asTypeOf(new ResourceSet(p))
  )

  io.input.ready := state === idle
  io.output.valid := state === result
  io.output.bits := resultResources

  when(io.input.fire) {
    context := io.input.bits
    state := dimensions
  }

  when(state === dimensions) {
    val channels = context.inputChannels
    val operandsPerGroup = MuxLookup(context.weightPrecision, 1.U(4.W))(
      Seq(
        WgtPrecision.w2.U -> 8.U,
        WgtPrecision.w4.U -> 4.U,
        WgtPrecision.w8.U -> 2.U,
        WgtPrecision.w16.U -> 1.U
      )
    )
    // Division by the precision-dependent 1/2/4/8 is expressed as fixed
    // shifts.  Multiplying the rounded group count back by the group size is
    // equivalent to rounding the channel count up to that power of two.
    val roundedChannels = MuxLookup(
      context.weightPrecision,
      channels
    )(
      Seq(
        WgtPrecision.w2.U -> (((channels + 7.U) >> 3) << 3),
        WgtPrecision.w4.U -> (((channels + 3.U) >> 2) << 2),
        WgtPrecision.w8.U -> (((channels + 1.U) >> 1) << 1),
        WgtPrecision.w16.U -> channels
      )
    )
    val singleInputGroup =
      context.arrayMode === ArrayMode.depthwise.U ||
        context.arrayMode === ArrayMode.columnReduce.U

    inputPixels := context.inputHeight * context.inputWidth
    outputPixels := context.outputHeight * context.outputWidth
    kernelElements := context.kernelHeight * context.kernelWidth
    inputBlocks :=
      (context.inputChannels + (p.dim - 1).U) >> p.laneIndexBits
    outputBlocks :=
      (context.outputChannels + (p.dim - 1).U) >> p.laneIndexBits
    weightInputFactor := Mux(
      singleInputGroup,
      Mux(
        context.weightPrecision === WgtPrecision.w16.U,
        1.U,
        operandsPerGroup
      ),
      roundedChannels
    )
    state := rowProducts
  }

  when(state === rowProducts) {
    val columnReduce =
      context.arrayMode === ArrayMode.columnReduce.U
    val shuffleRowsPerPixel =
      (p.maxShuffleScale * p.maxShuffleScale / 2).U
    val shuffleShift = log2Ceil(p.maxShuffleScale)

    inputRows := Mux(
      columnReduce,
      (inputPixels + 1.U) >> 1,
      inputPixels * inputBlocks
    )
    outputRows := Mux(
      columnReduce,
      (outputPixels + (p.dim - 1).U) >> p.laneIndexBits,
      Mux(
        context.shufflePack2,
        outputPixels * shuffleRowsPerPixel,
        outputPixels * outputBlocks
      )
    )
    binaryOutputRows := outputPixels * outputBlocks
    weightKernelBlocks := outputBlocks * kernelElements
    binaryWeightPartial := context.outputChannels * kernelElements
    lowResidualPixels :=
      (context.outputHeight >> shuffleShift) *
        (context.outputWidth >> shuffleShift)
    state := weightProducts
  }

  when(state === weightProducts) {
    val columnReduce =
      context.arrayMode === ArrayMode.columnReduce.U
    multiWeightRows := weightKernelBlocks * weightInputFactor
    binaryWeightRows := binaryWeightPartial * inputBlocks
    residualRows := Mux(
      context.postMode === PostMode.finalBilinearResidual.U,
      (lowResidualPixels + (p.dim - 1).U) >> p.laneIndexBits,
      binaryOutputRows
    )
    accumulatorRows := Mux(
      columnReduce,
      (outputPixels + (p.dim - 1).U) >> p.laneIndexBits,
      outputPixels
    )
    state := buildMasks
  }

  private def contiguousBankMask(
    baseRow: UInt,
    rowCount: UInt,
    banks: Int
  ): UInt = {
    val start = baseRow.pad(32)
    val last = start + rowCount - 1.U
    VecInit((0 until banks).map { bank =>
      val bankStart = (bank * p.bankRows).U(32.W)
      val bankEnd = ((bank + 1) * p.bankRows - 1).U(32.W)
      rowCount =/= 0.U && start <= bankEnd && last >= bankStart
    }).asUInt
  }

  when(state === buildMasks) {
    val resources = WireDefault(
      0.U.asTypeOf(new ResourceSet(p))
    )
    val binaryMode = context.arrayMode === ArrayMode.binary.U
    val usesResidual =
      context.postMode === PostMode.binaryFused.U ||
        context.postMode === PostMode.finalBilinearResidual.U

    val inputBase = context.baseRows(AddrRole.input)
    val weightLowBase = context.baseRows(AddrRole.weightLow)
    val weightHighBase = context.baseRows(AddrRole.weightHigh)
    val residualBase = context.baseRows(AddrRole.residual)
    val accumulatorBase = context.baseRows(AddrRole.accumulator)
    val outputFullBase = context.baseRows(AddrRole.outputFull)
    val outputBinaryBase = context.baseRows(AddrRole.outputBinary)

    val inputLowWgtMask =
      contiguousBankMask(inputBase, inputRows, p.fullBanks) |
        contiguousBankMask(weightLowBase, multiWeightRows, p.fullBanks)
    val highWeightMask = Mux(
      context.weightPrecision === WgtPrecision.w16.U,
      contiguousBankMask(weightHighBase, multiWeightRows, p.fullBanks),
      0.U
    )
    val residualMask = Mux(
      usesResidual,
      contiguousBankMask(residualBase, residualRows, p.fullBanks),
      0.U
    )

    resources.fullRead :=
      Mux(binaryMode, 0.U, inputLowWgtMask | highWeightMask) |
        residualMask
    resources.binaryRead := Mux(
      binaryMode,
      contiguousBankMask(inputBase, inputRows, p.binaryBanks) |
        contiguousBankMask(
          weightLowBase,
          binaryWeightRows,
          p.binaryBanks
        ),
      0.U
    )
    val workingAccumulatorMask = contiguousBankMask(
      accumulatorBase,
      accumulatorRows,
      p.accumulatorBanks
    )
    resources.accumulatorRead := workingAccumulatorMask
    resources.accumulatorWrite := workingAccumulatorMask
    resources.parameterRead := true.B
    resources.fullWrite := Mux(
      context.writeFull,
      contiguousBankMask(outputFullBase, outputRows, p.fullBanks),
      0.U
    )
    resources.binaryWrite := Mux(
      context.writeBinary,
      contiguousBankMask(
        outputBinaryBase,
        binaryOutputRows,
        p.binaryBanks
      ),
      0.U
    )

    resultResources := resources
    state := result
  }

  when(io.output.fire) {
    state := idle
  }
}
