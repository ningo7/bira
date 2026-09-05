package bira

// Multi-bit convolution controller.

import chisel3._
import chisel3.util._

/** Completed convolution row waiting for post-processing and writeback. */
class ConvCompletion(p: AccelParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
  val outputX = UInt(p.imageDimensionBits.W)
  val outputY = UInt(p.imageDimensionBits.W)
  val accumulator =
    Vec(p.dim, SInt(p.accumulatorBits.W))
}

/** One decoupled Accumulator transaction produced by the array pipeline. */
class ConvAccumulatorTask(p: AccelParams) extends Bundle {
  val readOnly = Bool()
  val address = UInt(p.accumulatorAddressBits.W)
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
  val completesOutput = Bool()
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
  val outputX = UInt(p.imageDimensionBits.W)
  val outputY = UInt(p.imageDimensionBits.W)
}

/** Runtime-configured stride-one convolution controller.
  *
  * The controller is not tied to a named network layer. Kernel dimensions,
  * padding, input channels, output blocks, activation signedness, activation
  * precision, and weight precision all come from ConvolutionCommand.
  *
  * Compute loop order:
  *
  *   output block
  *     -> kernel tap
  *       -> input-channel group
  *         -> output pixel
  *           -> activation bit
  *
  * A weight tile remains stationary while every output pixel is visited. The
  * input-channel group width is derived from the 16 weight bit-cells in each
  * array column: W16/W8/W4/W2 process 1/2/4/8 channels at once.
  */
class ConvCtrl(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new ConvolutionCommand(p)))
    val status = Output(new AcceleratorStatus)

    val biases = Input(
      Vec(p.maxOutputBlocks, Vec(p.dim, SInt(p.accumulatorBits.W)))
    )
    val postParameters = Input(
      Vec(
        p.maxOutputBlocks,
        Vec(p.dim, new PostProcessParameters(p))
      )
    )
    val binaryThresholds = Input(
      Vec(p.maxOutputBlocks, Vec(p.dim, SInt(p.accumulatorBits.W)))
    )

    val actFetchReq =
      Decoupled(new ActivationFetchRequest(p))
    val actFetchResp =
      Flipped(Decoupled(new ActivationFetchResponse(p)))
    val wgtFetchReq =
      Decoupled(new WeightFetchRequest(p))
    val wgtFetchResp =
      Flipped(Decoupled(new WeightFetchResponse(p)))

    val interpolationReq =
      Decoupled(new InterpReq(p))
    val interpolationResp =
      Flipped(Decoupled(Vec(p.dim, UInt(p.activationBits.W))))

    val accReq =
      Decoupled(new AccumulatorRequest(p))
    val accResp =
      Flipped(Decoupled(new AccumulatorResponse(p)))

    val postIn = Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val postInValid = Output(Bool())
    val postParams =
      Output(Vec(p.dim, new PostProcessParameters(p)))
    val postOut = Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val postOutValid = Input(Bool())

    val fullOutWrite = Decoupled(
      new VectorWriteRequest(
        p.fullAddressBits,
        p.dim,
        p.activationBits
      )
    )
    val binOutWrite = Decoupled(
      new VectorWriteRequest(
        p.binaryAddressBits,
        p.dim,
        1
      )
    )

    val arrayActBits =
      Output(Vec(p.maxWeightOperands, Vec(p.dim, Bool())))
    val arrayWgts = Output(
      Vec(p.maxWeightOperands, Vec(p.dim, UInt(p.weightBits.W)))
    )
    val arrayWPrec =
      Output(UInt(p.weightPrecisionBits.W))
    val arrayColumnReduce = Output(Bool())
    val arrayInValid = Output(Bool())
    val arraySums =
      Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val arrayOutValid = Input(Bool())
    val arrayReducedSum = Input(SInt(p.accumulatorBits.W))
    val arrayReducedValid = Input(Bool())
  })

  private val Seq(
    idle,
    initializeAccumulator,
    waitForInitialize,
    issueWeightFetch,
    waitForWeightFetch,
    usePrefetchedWeights,
    processPixel,
    issueActivationFetch,
    waitForActivationFetch,
    issueArrayCompute,
    waitForArrayResult,
    issueBitAccumulate,
    issueAccumulatorRead,
    waitForOutputDrain
  ) = Enum(14)

  private val Seq(
    postIdle,
    postIssueResidual,
    postWaitResidual,
    postIssueProcess,
    postWaitForProcess,
    postEnqueueOutput
  ) = Enum(6)

  private val state = RegInit(idle)
  private val postState = RegInit(postIdle)
  private val commandRegister =
    RegInit(0.U.asTypeOf(new ConvolutionCommand(p)))
  private val block = RegInit(0.U(p.blockIndexBits.W))
  private val pixel = RegInit(0.U(p.pixelIndexBits.W))
  private val kernelTap = RegInit(0.U(p.kernelIndexBits.W))
  private val outputX = RegInit(0.U(p.imageDimensionBits.W))
  private val outputY = RegInit(0.U(p.imageDimensionBits.W))
  private val kernelX = RegInit(0.U(p.kernelDimensionBits.W))
  private val kernelY = RegInit(0.U(p.kernelDimensionBits.W))
  private val inputGroup = RegInit(0.U(p.channelCountBits.W))
  private val weightOperand =
    RegInit(0.U(p.weightOperandIndexBits.W))
  private val activationOperand =
    RegInit(0.U(p.weightOperandIndexBits.W))
  private val activationBitIndex =
    RegInit(0.U(p.activationBitIndexBits.W))
  private val arrayResultBitIndex =
    RegInit(0.U(p.activationBitIndexBits.W))
  private val arrayBurstDone = RegInit(false.B)
  private val fetchedActivations =
    Reg(Vec(p.maxWeightOperands, UInt(p.activationBits.W)))
  private val fetchedActivationRow =
    Reg(Vec(p.dim, UInt(p.activationBits.W)))
  private val spatialIndexBits = p.pixelIndexBits + 2
  private val denseLinearBits =
    p.fullAddressBits + p.laneIndexBits + 2
  private val spatialIndexRegister = Reg(SInt(spatialIndexBits.W))
  private val denseLinearBaseRegister = Reg(SInt(denseLinearBits.W))
  private val depthwiseAddressRegister =
    Reg(SInt((p.fullAddressBits + 2).W))
  private val activationAddressRegister = Reg(UInt(p.fullAddressBits.W))
  private val activationLaneRegister = Reg(UInt(p.laneIndexBits.W))
  private val activationFetchRequiredRegister = RegInit(false.B)
  // Every command and output-block transition initializes this register
  // before it is consumed. Avoid a global synchronous reset mux on its live
  // update path; that mux had acquired completion-queue control as a long
  // route to the register R pins after synthesis.
  private val inputGroupChannelBase = Reg(UInt(p.channelCountBits.W))
  // Kernel-tap origins advance incrementally.  This removes the two cascaded
  // multipliers previously used each time a tile began.
  private val kernelSpatialStart = Reg(SInt(spatialIndexBits.W))
  private val kernelDenseStart = Reg(SInt(denseLinearBits.W))
  private val kernelDepthwiseStart =
    Reg(SInt((p.fullAddressBits + 2).W))
  private val initialSpatialStart = Reg(SInt(spatialIndexBits.W))
  private val initialDenseStart = Reg(SInt(denseLinearBits.W))
  private val initialDepthwiseStart =
    Reg(SInt((p.fullAddressBits + 2).W))
  private val spatialTapRowAdvance = Reg(SInt(spatialIndexBits.W))
  private val denseTapRowAdvance = Reg(SInt(denseLinearBits.W))
  private val depthwiseTapRowAdvance =
    Reg(SInt((p.fullAddressBits + 2).W))
  private val scanConfigPending = RegInit(false.B)
  private val spatialRowAdvance = Reg(SInt(spatialIndexBits.W))
  private val denseRowAdvance = Reg(SInt(denseLinearBits.W))
  private val depthwiseRowAdvance =
    Reg(SInt((p.fullAddressBits + 2).W))
  private val stationaryWeights =
    Reg(Vec(p.maxWeightOperands, Vec(p.dim, UInt(p.weightBits.W))))
  private val prefetchedWeights =
    Reg(Vec(p.maxWeightOperands, Vec(p.dim, UInt(p.weightBits.W))))
  private val preloadActive = RegInit(false.B)
  private val preloadWaiting = RegInit(false.B)
  private val preloadValid = RegInit(false.B)
  private val preloadBlock = RegInit(0.U(p.blockIndexBits.W))
  private val preloadKernelTap = RegInit(0.U(p.kernelIndexBits.W))
  private val preloadInputGroup = RegInit(0.U(p.channelCountBits.W))
  private val preloadOperand =
    RegInit(0.U(p.weightOperandIndexBits.W))
  // Weight rows are consumed in flattened tile order.  Keep the current and
  // prefetched tile bases as running offsets instead of rebuilding the
  // block/kernel/group products on every request.
  private val weightLowOffsetRegister =
    RegInit(0.U(p.fullAddressBits.W))
  private val weightHighOffsetRegister =
    RegInit(0.U(p.fullAddressBits.W))
  private val preloadWeightLowOffsetRegister =
    RegInit(0.U(p.fullAddressBits.W))
  private val preloadWeightHighOffsetRegister =
    RegInit(0.U(p.fullAddressBits.W))
  private val postAccumulator =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val postBlock = RegInit(0.U(p.blockIndexBits.W))
  private val postPixel = RegInit(0.U(p.pixelIndexBits.W))
  private val postOutputX = RegInit(0.U(p.imageDimensionBits.W))
  private val postOutputY = RegInit(0.U(p.imageDimensionBits.W))
  private val nextCompletionOutputX = RegInit(0.U(p.imageDimensionBits.W))
  private val nextCompletionOutputY = RegInit(0.U(p.imageDimensionBits.W))
  private val postNormalOffsetRegister = Reg(UInt(p.fullAddressBits.W))
  private val nextPostNormalOffset = Reg(UInt(p.fullAddressBits.W))
  private val postBinaryOffsetRegister = Reg(UInt(p.binaryAddressBits.W))
  private val nextPostBinaryOffset = Reg(UInt(p.binaryAddressBits.W))
  private val groupPartialSum =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val completedPostOutput =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val residualRow =
    Reg(Vec(p.dim, UInt(p.activationBits.W)))
  // Two-slot, non-transparent timing buffers prevent downstream ready from
  // propagating combinationally across VecAccum. They preserve steady-state
  // II=1 but do not provide task scheduling or reordering.
  private val completionQueue = Module(
    new TwoEntryBuffer(new ConvCompletion(p))
  )
  private val accumulatorTaskQueue = Module(
    new TwoEntryBuffer(new ConvAccumulatorTask(p))
  )
  private val accumulatorOutstanding = RegInit(0.U(2.W))
  private val outputBufferValid = RegInit(false.B)
  private val outputFullPending = RegInit(false.B)
  private val outputBinaryPending = RegInit(false.B)
  private val outputFullAddress = Reg(UInt(p.fullAddressBits.W))
  private val outputBinaryAddress = Reg(UInt(p.binaryAddressBits.W))
  private val outputFullData =
    Reg(Vec(p.dim, UInt(p.activationBits.W)))
  private val outputBinaryData = Reg(Vec(p.dim, UInt(1.W)))
  private val weightFetchCount =
    RegInit(0.U(p.weightFetchCountBits.W))
  private val donePulse = RegInit(false.B)

  // Decode precision-dependent loop bounds once when the command is accepted.
  // Keeping these values registered prevents weightPrecision from feeding the
  // inner-loop address multipliers and shared-memory ready cones.
  private val operandsPerCycle =
    RegInit(1.U(p.weightOperandCountBits.W))
  private val inputGroupCount =
    RegInit(1.U(p.channelCountBits.W))
  private val fullPrecision = RegInit(false.B)
  // Command-shape products are invariant for the whole operation.  Register
  // them at command acceptance so command fields do not drive the inner-loop
  // control and address cones.
  private val runtimeKernelElements =
    RegInit(1.U((2 * p.kernelDimensionBits).W))
  private val outputPixelCount =
    // A count must represent maxPixels itself, while pixelIndexBits only
    // needs to represent indices through maxPixels - 1.
    RegInit(1.U((p.pixelIndexBits + 1).W))
  private val accumulatorRowCount = Mux(
    commandRegister.columnReduce,
    (outputPixelCount + (p.dim - 1).U) >> p.laneIndexBits,
    outputPixelCount
  )

  private val isLastPixel = pixel === outputPixelCount - 1.U
  private val accumulatorLane =
    pixel.pad(p.laneIndexBits)(p.laneIndexBits - 1, 0)
  private val isAccumulatorRowEnd =
    accumulatorLane === (p.dim - 1).U || isLastPixel
  private val isLastKernelTap =
    kernelTap === runtimeKernelElements - 1.U
  private val isLastInputGroup =
    inputGroup === inputGroupCount - 1.U
  private val isLastWeightOperand =
    weightOperand === operandsPerCycle - 1.U
  private val isLastActivationOperand =
    activationOperand === operandsPerCycle - 1.U

  private val signedInputY =
    outputY.zext + kernelY.zext - commandRegister.paddingY.zext
  private val signedInputX =
    outputX.zext + kernelX.zext - commandRegister.paddingX.zext
  private val inputIsPadding =
    signedInputY < 0.S ||
      signedInputY >= commandRegister.inputHeight.zext ||
      signedInputX < 0.S ||
      signedInputX >= commandRegister.inputWidth.zext

  private val depthwiseInputAddress =
    depthwiseAddressRegister.asUInt(p.fullAddressBits - 1, 0)
  private val reducedInputAddress =
    commandRegister.inputBase + (spatialIndexRegister.asUInt >> 1)

  private val accumulatorAddress =
    commandRegister.accumulatorBase +
      Mux(
        commandRegister.columnReduce,
        pixel >> p.laneIndexBits,
        pixel
      )
  private val shuffleScale =
    (1.U(p.shuffleScaleBits.W) <<
      commandRegister.shuffleLog2)(
      p.shuffleScaleBits - 1,
      0
    )
  // shuffleScale is constrained to a power of two. Express these products
  // as shifts so Vivado does not build a cascaded DSP address datapath.
  private val shuffledWidth =
    commandRegister.outputWidth << commandRegister.shuffleLog2
  private val postShuffleSubPixel0 = postBlock << 1
  private val postShuffleSubY =
    postShuffleSubPixel0 >> commandRegister.shuffleLog2
  private val postShuffleSubX =
    postShuffleSubPixel0 & (shuffleScale - 1.U)
  private val postShuffledY =
    (postOutputY << commandRegister.shuffleLog2) + postShuffleSubY
  private val postShuffledX =
    (postOutputX << commandRegister.shuffleLog2) + postShuffleSubX
  private val postNormalOutputOffset =
    postNormalOffsetRegister
  private val postNormalOrColumnOffset = Mux(
    commandRegister.columnReduce,
    postPixel >> p.laneIndexBits,
    postNormalOutputOffset
  )

  // Prepare the post-processing write addresses in parallel with the much
  // longer arithmetic pipeline. Keeping coordinate formation, the shuffled
  // row multiply, and output-base addition in separate cycles removes the
  // former postBlock -> DSP -> CARRY4 -> outputFullAddress path without adding
  // a post-processing cycle.
  private val postAddressCoordinateValid = RegInit(false.B)
  private val postAddressProductValid = RegInit(false.B)
  private val postAddressFinalizeValid = RegInit(false.B)
  private val postAddressPreparedValid = RegInit(false.B)
  private val postAddressY = Reg(chiselTypeOf(postShuffledY))
  private val postAddressX = Reg(chiselTypeOf(postShuffledX))
  private val postAddressWidth = Reg(chiselTypeOf(shuffledWidth))
  private val postAddressNormalOffset = Reg(UInt(p.fullAddressBits.W))
  private val postAddressBinaryOffset = Reg(UInt(p.binaryAddressBits.W))
  private val postAddressUsesShuffle = Reg(Bool())
  private val postAddressShuffleOffset = Reg(UInt(p.fullAddressBits.W))
  private val postAddressProductNormalOffset =
    Reg(UInt(p.fullAddressBits.W))
  private val postAddressProductBinaryOffset =
    Reg(UInt(p.binaryAddressBits.W))
  private val postAddressProductUsesShuffle = Reg(Bool())
  private val preparedFullOutputAddress = Reg(UInt(p.fullAddressBits.W))
  private val preparedBinaryOutputAddress =
    Reg(UInt(p.binaryAddressBits.W))

  /** Load the first activation address of the current kernel/input tile.
    * The multiplications occur only at the tile boundary; the live pixel path
    * below advances these registers with adders.
    */
  private def loadActivationScanStart(): Unit = {
    spatialIndexRegister := kernelSpatialStart
    denseLinearBaseRegister := kernelDenseStart
    depthwiseAddressRegister := kernelDepthwiseStart
  }

  /** Prepare an activation request before entering issueActivationFetch.
    * The request state then presents only registered address and lane signals
    * to ConvFetch and the shared SPAD arbiter.
    */
  private def prepareActivationFetch(operand: UInt): Unit = {
    val channel = inputGroupChannelBase + operand
    val linearIndex =
      (denseLinearBaseRegister + channel.zext).asUInt
    activationAddressRegister := Mux(
      commandRegister.columnReduce,
      reducedInputAddress,
      Mux(
        commandRegister.depthwise,
        depthwiseInputAddress,
        commandRegister.inputBase +
          (linearIndex >> p.laneIndexBits)
      )
    )
    activationLaneRegister := Mux(
      commandRegister.depthwise,
      0.U,
      linearIndex(p.laneIndexBits - 1, 0)
    )
    activationFetchRequiredRegister :=
      commandRegister.depthwise || channel < commandRegister.inputChannels
  }

  /** Start filling the inactive weight buffer with the next global tile. */
  private def armNextTilePreload(
    currentLowOffset: UInt,
    currentHighOffset: UInt
  ): Unit = {
    preloadOperand := 0.U
    preloadWaiting := false.B
    preloadValid := false.B
    preloadWeightLowOffsetRegister :=
      currentLowOffset + Mux(fullPrecision, 1.U, operandsPerCycle)
    preloadWeightHighOffsetRegister := currentHighOffset + 1.U
    when(!isLastInputGroup) {
      preloadBlock := block
      preloadKernelTap := kernelTap
      preloadInputGroup := inputGroup + 1.U
      preloadActive := true.B
    }.elsewhen(!isLastKernelTap) {
      preloadBlock := block
      preloadKernelTap := kernelTap + 1.U
      preloadInputGroup := 0.U
      preloadActive := true.B
    }.elsewhen(block =/= commandRegister.outputBlocks - 1.U) {
      preloadBlock := block + 1.U
      preloadKernelTap := 0.U
      preloadInputGroup := 0.U
      preloadActive := true.B
    }.otherwise {
      preloadActive := false.B
    }
  }

  private def advanceAfterPixel(): Unit = {
    when(isLastPixel) {
      pixel := 0.U
      outputX := 0.U
      outputY := 0.U
      weightOperand := 0.U
      when(isLastInputGroup) {
        inputGroup := 0.U
        inputGroupChannelBase := 0.U
        when(isLastKernelTap) {
          state := issueAccumulatorRead
        }.otherwise {
          kernelTap := kernelTap + 1.U
          when(kernelX === commandRegister.kernelWidth - 1.U) {
            kernelX := 0.U
            kernelY := kernelY + 1.U
            kernelSpatialStart :=
              kernelSpatialStart + spatialTapRowAdvance
            kernelDenseStart := kernelDenseStart + denseTapRowAdvance
            kernelDepthwiseStart :=
              kernelDepthwiseStart + depthwiseTapRowAdvance
          }.otherwise {
            kernelX := kernelX + 1.U
            kernelSpatialStart := kernelSpatialStart + 1.S
            kernelDenseStart :=
              kernelDenseStart + commandRegister.inputChannels.zext
            kernelDepthwiseStart :=
              kernelDepthwiseStart + commandRegister.outputBlocks.zext
          }
          state := usePrefetchedWeights
        }
      }.otherwise {
        inputGroup := inputGroup + 1.U
        inputGroupChannelBase :=
          inputGroupChannelBase + operandsPerCycle
        state := usePrefetchedWeights
      }
    }.otherwise {
      pixel := pixel + 1.U
      spatialIndexRegister := Mux(
        outputX === commandRegister.outputWidth - 1.U,
        spatialIndexRegister + spatialRowAdvance,
        spatialIndexRegister + 1.S
      )
      denseLinearBaseRegister := Mux(
        outputX === commandRegister.outputWidth - 1.U,
        denseLinearBaseRegister + denseRowAdvance,
        denseLinearBaseRegister + commandRegister.inputChannels.zext
      )
      depthwiseAddressRegister := Mux(
        outputX === commandRegister.outputWidth - 1.U,
        depthwiseAddressRegister + depthwiseRowAdvance,
        depthwiseAddressRegister + commandRegister.outputBlocks.zext
      )
      when(outputX === commandRegister.outputWidth - 1.U) {
        outputX := 0.U
        outputY := outputY + 1.U
      }.otherwise {
        outputX := outputX + 1.U
      }
      state := processPixel
    }
  }

  private def advanceAfterAcc(): Unit = {
    when(isLastPixel && isLastKernelTap && isLastInputGroup) {
      pixel := 0.U
      outputX := 0.U
      outputY := 0.U
      state := waitForOutputDrain
    }.otherwise {
      advanceAfterPixel()
    }
  }

  private def advanceAfterActivation(): Unit = {
    when(isLastActivationOperand) {
      activationBitIndex := 0.U
      arrayResultBitIndex := 0.U
      arrayBurstDone := false.B
      groupPartialSum := VecInit.fill(p.dim)(
        0.S(p.accumulatorBits.W)
      )
      state := issueArrayCompute
    }.otherwise {
      val nextOperand = activationOperand + 1.U
      activationOperand := nextOperand
      prepareActivationFetch(nextOperand)
      state := issueActivationFetch
    }
  }

  private def finishOutputBlock(): Unit = {
    when(block === commandRegister.outputBlocks - 1.U) {
      val expectedWeightFetches =
        commandRegister.outputBlocks *
          runtimeKernelElements *
          inputGroupCount *
          operandsPerCycle
      assert(
        weightFetchCount === expectedWeightFetches,
        "every weight tile operand must be fetched exactly once"
      )
      state := idle
      donePulse := true.B
    }.otherwise {
      block := block + 1.U
      pixel := 0.U
      kernelTap := 0.U
      outputX := 0.U
      outputY := 0.U
      kernelX := 0.U
      kernelY := 0.U
      postOutputX := 0.U
      postOutputY := 0.U
      nextCompletionOutputX := 0.U
      nextCompletionOutputY := 0.U
      postNormalOffsetRegister := block + 1.U
      nextPostNormalOffset := block + 1.U
      postBinaryOffsetRegister := block + 1.U
      nextPostBinaryOffset := block + 1.U
      inputGroup := 0.U
      inputGroupChannelBase := 0.U
      weightOperand := 0.U
      kernelSpatialStart := initialSpatialStart
      kernelDenseStart := initialDenseStart
      kernelDepthwiseStart :=
        initialDepthwiseStart + (block + 1.U).zext
      state := initializeAccumulator
    }
  }

  donePulse := false.B
  io.command.ready :=
    state === idle &&
      postState === postIdle &&
      !completionQueue.io.deq.valid &&
      !outputBufferValid
  io.status.busy :=
    state =/= idle ||
      accumulatorOutstanding =/= 0.U ||
      accumulatorTaskQueue.io.deq.valid ||
      postState =/= postIdle ||
      completionQueue.io.deq.valid ||
      outputBufferValid
  io.status.done := donePulse

  io.actFetchReq.valid := false.B
  io.actFetchReq.bits.address := activationAddressRegister
  io.actFetchReq.bits.lane := activationLaneRegister
  io.actFetchResp.ready := state === waitForActivationFetch

  io.wgtFetchReq.valid := false.B
  io.wgtFetchReq.bits :=
    0.U.asTypeOf(new WeightFetchRequest(p))
  io.wgtFetchResp.ready :=
    state === waitForWeightFetch || preloadWaiting

  io.interpolationReq.valid := false.B
  io.interpolationReq.bits :=
    0.U.asTypeOf(new InterpReq(p))
  io.interpolationResp.ready :=
    postState === postWaitResidual

  io.accReq.valid := false.B
  io.accReq.bits :=
    0.U.asTypeOf(new AccumulatorRequest(p))
  io.accResp.ready :=
    state === waitForInitialize

  io.postIn := postAccumulator
  io.postInValid := postState === postIssueProcess
  private val selectedPostParameters =
    if (p.maxOutputBlocks == 1) io.postParameters(0)
    else {
      val indexBits = log2Ceil(p.maxOutputBlocks)
      io.postParameters(postBlock(indexBits - 1, 0))
    }
  private val selectedBiases =
    if (p.maxOutputBlocks == 1) io.biases(0)
    else {
      val indexBits = log2Ceil(p.maxOutputBlocks)
      io.biases(block(indexBits - 1, 0))
    }
  private val selectedBinaryThresholds =
    if (p.maxOutputBlocks == 1) io.binaryThresholds(0)
    else {
      val indexBits = log2Ceil(p.maxOutputBlocks)
      io.binaryThresholds(postBlock(indexBits - 1, 0))
    }
  for (lane <- 0 until p.dim) {
    io.postParams(lane) := Mux(
      commandRegister.columnReduce,
      selectedPostParameters(0),
      selectedPostParameters(lane)
    )
  }

  io.fullOutWrite.valid :=
    outputBufferValid && outputFullPending
  io.fullOutWrite.bits.address := outputFullAddress
  io.fullOutWrite.bits.data := outputFullData
  io.binOutWrite.valid :=
    outputBufferValid && outputBinaryPending
  io.binOutWrite.bits.address := outputBinaryAddress
  io.binOutWrite.bits.data := outputBinaryData
  completionQueue.io.enq.valid := false.B
  completionQueue.io.enq.bits :=
    0.U.asTypeOf(new ConvCompletion(p))
  completionQueue.io.deq.ready := false.B
  accumulatorTaskQueue.io.enq.valid := false.B
  accumulatorTaskQueue.io.enq.bits :=
    0.U.asTypeOf(new ConvAccumulatorTask(p))
  accumulatorTaskQueue.io.deq.ready := false.B

  private val fullWriteCompletes =
    !outputFullPending || io.fullOutWrite.fire
  private val binaryWriteCompletes =
    !outputBinaryPending || io.binOutWrite.fire
  when(outputBufferValid) {
    when(io.fullOutWrite.fire) {
      outputFullPending := false.B
    }
    when(io.binOutWrite.fire) {
      outputBinaryPending := false.B
    }
    when(fullWriteCompletes && binaryWriteCompletes) {
      outputBufferValid := false.B
    }
  }

  // The command-start products are deliberately split across two cycles.
  // Accumulator initialization provides many cycles before the first tile
  // consumes these values, so this adds no execution bubble.
  when(scanConfigPending) {
    val denseOrigin =
      initialSpatialStart * commandRegister.inputChannels.zext
    val depthwiseOrigin =
      commandRegister.inputBase.zext +
        initialSpatialStart * commandRegister.outputBlocks.zext
    initialDenseStart := denseOrigin
    initialDepthwiseStart := depthwiseOrigin
    kernelDenseStart := denseOrigin
    kernelDepthwiseStart := depthwiseOrigin
    denseTapRowAdvance :=
      spatialTapRowAdvance * commandRegister.inputChannels.zext
    depthwiseTapRowAdvance :=
      spatialTapRowAdvance * commandRegister.outputBlocks.zext
    scanConfigPending := false.B
  }

  for (operand <- 0 until p.maxWeightOperands) {
    for (lane <- 0 until p.dim) {
      val reducedActivation =
        if (lane < p.dim / 2) {
          operand.U === 0.U &&
            Mux(
              spatialIndexRegister.asUInt(0),
              fetchedActivationRow(lane + p.dim / 2)(
                activationBitIndex
              ),
              fetchedActivationRow(lane)(activationBitIndex)
            )
        } else {
          false.B
        }
      io.arrayActBits(operand)(lane) := Mux(
        commandRegister.columnReduce,
        reducedActivation,
        Mux(
          commandRegister.depthwise,
          operand.U === 0.U &&
            fetchedActivationRow(lane)(activationBitIndex),
          fetchedActivations(operand)(activationBitIndex)
        )
      )
    }
  }
  io.arrayWgts := stationaryWeights
  io.arrayWPrec := commandRegister.weightPrecision
  io.arrayColumnReduce := commandRegister.columnReduce
  io.arrayInValid := state === issueArrayCompute

  // AddTree accepts one activation bit-plane per cycle. Collect its ordered
  // two-cycle-latency responses locally, reconstruct the signed A8 product,
  // and update the SRAM accumulator once per pixel/tile instead of once per
  // activation bit.
  private val arrayResultValid = Mux(
    commandRegister.columnReduce,
    io.arrayReducedValid,
    io.arrayOutValid
  )
  private val arrayResultVector =
    Wire(Vec(p.dim, SInt(p.accumulatorBits.W)))
  arrayResultVector := io.arraySums
  when(commandRegister.columnReduce) {
    arrayResultVector :=
      VecInit.fill(p.dim)(0.S(p.accumulatorBits.W))
    arrayResultVector(accumulatorLane) :=
      io.arrayReducedSum
  }
  private val accumulatedArrayResult =
    Wire(Vec(p.dim, SInt(p.accumulatorBits.W)))
  for (lane <- 0 until p.dim) {
    val signedPartial = Mux(
      commandRegister.inputSigned &&
        arrayResultBitIndex === commandRegister.activationPrecision - 1.U,
      -arrayResultVector(lane),
      arrayResultVector(lane)
    )
    val shiftedWide = signedPartial << arrayResultBitIndex
    val shifted = shiftedWide(p.accumulatorBits - 1, 0).asSInt
    accumulatedArrayResult(lane) := Mux(
      arrayResultBitIndex === 0.U,
      shifted,
      (groupPartialSum(lane) + shifted)(p.accumulatorBits - 1, 0).asSInt
    )
  }
  when(
    arrayResultValid &&
      (state === issueArrayCompute ||
        state === waitForArrayResult)
  ) {
    for (lane <- 0 until p.dim) {
      groupPartialSum(lane) := accumulatedArrayResult(lane)
    }
    when(
      arrayResultBitIndex ===
        commandRegister.activationPrecision - 1.U
    ) {
      arrayBurstDone := true.B
    }.otherwise {
      arrayResultBitIndex := arrayResultBitIndex + 1.U
    }
  }

  // The current tile owns stationaryWeights. While it traverses pixels, the
  // independent weight fetch path fills prefetchedWeights with the next tile.
  // Activation and weight reads use separate ConvFetch paths and SPAD ports.
  when(
    preloadActive &&
      !preloadWaiting &&
      state =/= issueWeightFetch &&
      state =/= waitForWeightFetch
  ) {
    io.wgtFetchReq.valid := true.B
    io.wgtFetchReq.bits.lowAddress :=
      commandRegister.weightLowBase +
        preloadWeightLowOffsetRegister + preloadOperand
    io.wgtFetchReq.bits.highAddress :=
      commandRegister.weightHighBase +
        preloadWeightHighOffsetRegister
    io.wgtFetchReq.bits.highBytePresent :=
      fullPrecision
    when(io.wgtFetchReq.fire) {
      preloadWaiting := true.B
      weightFetchCount := weightFetchCount + 1.U
    }
  }

  when(preloadWaiting && io.wgtFetchResp.fire) {
    prefetchedWeights(preloadOperand) :=
      io.wgtFetchResp.bits.weights
    preloadWaiting := false.B
    when(preloadOperand === operandsPerCycle - 1.U) {
      preloadActive := false.B
      preloadValid := true.B
    }.otherwise {
      preloadOperand := preloadOperand + 1.U
    }
  }

  // Carry completion metadata through VecAccum itself. This allows multiple
  // independent requests to be in flight without a controller-side tag FIFO.
  private val taskPortActive =
    state =/= initializeAccumulator && state =/= waitForInitialize
  when(taskPortActive) {
    io.accReq.valid := accumulatorTaskQueue.io.deq.valid
    io.accReq.bits.operation := Mux(
      accumulatorTaskQueue.io.deq.bits.readOnly,
      AccumulatorOperation.read,
      AccumulatorOperation.add
    )
    io.accReq.bits.address := accumulatorTaskQueue.io.deq.bits.address
    io.accReq.bits.data := accumulatorTaskQueue.io.deq.bits.data
    io.accReq.bits.shift := 0.U
    io.accReq.bits.negate := false.B
    io.accReq.bits.completesOutput :=
      accumulatorTaskQueue.io.deq.bits.completesOutput
    io.accReq.bits.block := accumulatorTaskQueue.io.deq.bits.block
    io.accReq.bits.pixel := accumulatorTaskQueue.io.deq.bits.pixel
    io.accReq.bits.outputX := accumulatorTaskQueue.io.deq.bits.outputX
    io.accReq.bits.outputY := accumulatorTaskQueue.io.deq.bits.outputY
    accumulatorTaskQueue.io.deq.ready := io.accReq.ready

    completionQueue.io.enq.valid :=
      io.accResp.valid && io.accResp.bits.completesOutput
    completionQueue.io.enq.bits.block := io.accResp.bits.block
    completionQueue.io.enq.bits.pixel := io.accResp.bits.pixel
    completionQueue.io.enq.bits.outputX := io.accResp.bits.outputX
    completionQueue.io.enq.bits.outputY := io.accResp.bits.outputY
    completionQueue.io.enq.bits.accumulator := io.accResp.bits.data
    io.accResp.ready := Mux(
      io.accResp.bits.completesOutput,
      completionQueue.io.enq.ready,
      true.B
    )
  }

  private val accumulatorTaskIssued =
    taskPortActive && accumulatorTaskQueue.io.deq.fire
  private val accumulatorTaskCompleted =
    taskPortActive && io.accResp.fire
  switch(Cat(accumulatorTaskIssued, accumulatorTaskCompleted)) {
    is("b10".U) { accumulatorOutstanding := accumulatorOutstanding + 1.U }
    is("b01".U) { accumulatorOutstanding := accumulatorOutstanding - 1.U }
  }

  // Output-address pipeline. A completion remains in post-processing for at
  // least IntPostProc.latency cycles, so these three address stages complete
  // before postEnqueueOutput without extending externally visible latency.
  postAddressCoordinateValid := false.B
  postAddressProductValid := postAddressCoordinateValid
  postAddressFinalizeValid := postAddressProductValid
  when(postAddressCoordinateValid) {
    postAddressY := postShuffledY
    postAddressX := postShuffledX
    postAddressWidth := shuffledWidth
    postAddressNormalOffset := postNormalOrColumnOffset
    postAddressBinaryOffset := postBinaryOffsetRegister
    postAddressUsesShuffle :=
      commandRegister.shufflePack2 && !commandRegister.columnReduce
  }
  when(postAddressProductValid) {
    val shuffledPixel = postAddressY * postAddressWidth + postAddressX
    postAddressShuffleOffset :=
      (shuffledPixel >> 1).pad(p.fullAddressBits)(
        p.fullAddressBits - 1,
        0
      )
    postAddressProductNormalOffset := postAddressNormalOffset
    postAddressProductBinaryOffset := postAddressBinaryOffset
    postAddressProductUsesShuffle := postAddressUsesShuffle
  }
  when(postAddressFinalizeValid) {
    val selectedOffset = Mux(
      postAddressProductUsesShuffle,
      postAddressShuffleOffset,
      postAddressProductNormalOffset
    )
    preparedFullOutputAddress :=
      commandRegister.outputBase + selectedOffset
    preparedBinaryOutputAddress :=
      commandRegister.binaryOutputBase + postAddressProductBinaryOffset
    postAddressPreparedValid := true.B
  }

  // Completion processing is independent of the convolution state machine.
  // The registered completion cut preserves in-order transfer while keeping
  // post/SPAD readiness out of the Accumulator response timing path.
  switch(postState) {
    is(postIdle) {
      when(completionQueue.io.deq.valid) {
        completionQueue.io.deq.ready := true.B
        postAccumulator :=
          completionQueue.io.deq.bits.accumulator
        postBlock := completionQueue.io.deq.bits.block
        postPixel := completionQueue.io.deq.bits.pixel
        postOutputX := completionQueue.io.deq.bits.outputX
        postOutputY := completionQueue.io.deq.bits.outputY
        postNormalOffsetRegister := nextPostNormalOffset
        nextPostNormalOffset :=
          nextPostNormalOffset + commandRegister.outputBlocks
        postBinaryOffsetRegister := nextPostBinaryOffset
        nextPostBinaryOffset :=
          nextPostBinaryOffset + commandRegister.outputBlocks
        postAddressCoordinateValid := true.B
        residualRow :=
          VecInit.fill(p.dim)(0.U(p.activationBits.W))
        postState := Mux(
          commandRegister.columnReduce &&
            commandRegister.bilinearResidual,
          postIssueResidual,
          postIssueProcess
        )
      }
    }

    is(postIssueResidual) {
      io.interpolationReq.valid := true.B
      io.interpolationReq.bits.inputBase :=
        commandRegister.residualBase
      io.interpolationReq.bits.inputHeight :=
        commandRegister.outputHeight >>
          commandRegister.shuffleLog2
      io.interpolationReq.bits.inputWidth :=
        commandRegister.outputWidth >>
          commandRegister.shuffleLog2
      io.interpolationReq.bits.outputWidth :=
        commandRegister.outputWidth
      io.interpolationReq.bits.outputStartPixel :=
        (postPixel >> p.laneIndexBits) << p.laneIndexBits
      io.interpolationReq.bits.outputStartX := postOutputX
      io.interpolationReq.bits.outputStartY := postOutputY
      io.interpolationReq.bits.outputPixelCount :=
        outputPixelCount
      io.interpolationReq.bits.scaleLog2 :=
        commandRegister.shuffleLog2
      when(io.interpolationReq.fire) {
        postState := postWaitResidual
      }
    }

    is(postWaitResidual) {
      when(io.interpolationResp.fire) {
        residualRow := io.interpolationResp.bits
        postState := postIssueProcess
      }
    }

    is(postIssueProcess) {
      postState := postWaitForProcess
    }

    is(postWaitForProcess) {
      when(io.postOutValid) {
        completedPostOutput := io.postOut
        postState := postEnqueueOutput
      }
    }

    is(postEnqueueOutput) {
      when(!outputBufferValid) {
        assert(
          postAddressPreparedValid,
          "post output address pipeline must complete before writeback"
        )
        outputBufferValid := true.B
        outputFullPending := commandRegister.writeFullOutput
        outputBinaryPending := commandRegister.writeBinaryOutput
        outputFullAddress := preparedFullOutputAddress
        outputBinaryAddress := preparedBinaryOutputAddress
        postAddressPreparedValid := false.B
        for (lane <- 0 until p.dim) {
          val finalSum =
            completedPostOutput(lane).pad(p.accumulatorBits + 1) +
              residualRow(lane).zext
          val finalClamped = Mux(
            finalSum < 0.S,
            0.U(p.activationBits.W),
            Mux(
              finalSum > ((1 << p.activationBits) - 1).S,
              ((1 << p.activationBits) - 1).U,
              finalSum(p.activationBits - 1, 0).asUInt
            )
          )
          val laneIsLive =
            (postPixel >> p.laneIndexBits) * p.dim.U + lane.U <
              outputPixelCount
          outputFullData(lane) := Mux(
            commandRegister.columnReduce,
            Mux(
              laneIsLive,
              Mux(
                commandRegister.bilinearResidual,
                finalClamped,
                completedPostOutput(lane)(
                  p.activationBits - 1,
                  0
                ).asUInt
              ),
              0.U
            ),
            completedPostOutput(lane)(
              p.activationBits - 1,
              0
            ).asUInt
          )
          outputBinaryData(lane) :=
            (completedPostOutput(lane) >=
              selectedBinaryThresholds(lane)).asUInt
        }
        postState := postIdle
      }
    }
  }

  when(io.command.fire) {
    val configuredOutputPixels =
      io.command.bits.outputHeight *
        io.command.bits.outputWidth
    val cfgAccRows = Mux(
      io.command.bits.columnReduce,
      (configuredOutputPixels + (p.dim - 1).U) >>
        p.laneIndexBits,
      configuredOutputPixels
    )

    assert(io.command.bits.inputHeight > 0.U)
    assert(io.command.bits.inputHeight <= p.maxImageHeight.U)
    assert(io.command.bits.inputWidth > 0.U)
    assert(io.command.bits.inputWidth <= p.maxImageWidth.U)
    assert(io.command.bits.outputHeight > 0.U)
    assert(io.command.bits.outputHeight <= p.maxImageHeight.U)
    assert(io.command.bits.outputWidth > 0.U)
    assert(io.command.bits.outputWidth <= p.maxImageWidth.U)
    assert(io.command.bits.inputChannels > 0.U)
    assert(io.command.bits.inputChannels <= p.maxInputChannels.U)
    assert(io.command.bits.outputBlocks > 0.U)
    assert(io.command.bits.outputBlocks <= p.maxOutputBlocks.U)
    assert(
      io.command.bits.accumulatorBase +&
        cfgAccRows <=
        p.accumulatorRows.U,
      "configured output does not fit in accumulator storage"
    )
    assert(io.command.bits.kernelHeight > 0.U)
    assert(io.command.bits.kernelHeight <= p.kernelSize.U)
    assert(io.command.bits.kernelWidth > 0.U)
    assert(io.command.bits.kernelWidth <= p.kernelSize.U)
    assert(
      io.command.bits.weightPrecision === 2.U ||
        io.command.bits.weightPrecision === 4.U ||
        io.command.bits.weightPrecision === 8.U ||
        io.command.bits.weightPrecision === 16.U
    )
    assert(io.command.bits.activationPrecision > 0.U)
    assert(io.command.bits.activationPrecision <= p.activationBits.U)
    assert(
      io.command.bits.writeFullOutput ||
        io.command.bits.writeBinaryOutput
    )
    when(io.command.bits.writeFullOutput) {
      assert(
        (io.command.bits.inputBase >> p.bankRowBits) =/=
          (io.command.bits.outputBase >> p.bankRowBits),
        "streaming full input and output must use different banks"
      )
    }
    when(io.command.bits.depthwise) {
      assert(io.command.bits.weightPrecision === 16.U)
      assert(
        io.command.bits.inputChannels ===
          io.command.bits.outputBlocks * p.dim.U
      )
    }
    when(io.command.bits.shufflePack2) {
      val scale =
        1.U(p.shuffleScaleBits.W) <<
          io.command.bits.shuffleLog2
      assert(io.command.bits.shuffleLog2 > 0.U)
      assert(
        io.command.bits.shuffleLog2 <=
          log2Ceil(p.maxShuffleScale).U
      )
      assert(
        io.command.bits.outputBlocks ===
          ((scale * scale) >> 1),
        "shufflePack2 requires scale^2/2 output blocks"
      )
      assert(io.command.bits.writeFullOutput)
      assert(!io.command.bits.writeBinaryOutput)
      assert(!io.command.bits.depthwise)
    }
    when(io.command.bits.columnReduce) {
      assert(
        io.command.bits.inputChannels === (p.dim / 2).U,
        "columnReduce uses the first half of the columns as input channels"
      )
      assert(io.command.bits.outputBlocks === 1.U)
      assert(io.command.bits.weightPrecision === 16.U)
      assert(io.command.bits.writeFullOutput)
      assert(!io.command.bits.writeBinaryOutput)
      assert(!io.command.bits.depthwise)
      assert(!io.command.bits.shufflePack2)
      when(io.command.bits.bilinearResidual) {
        assert(
          (io.command.bits.residualBase >> p.bankRowBits) =/=
            (io.command.bits.outputBase >> p.bankRowBits),
          "bilinear source and final output must use different banks"
        )
        assert(io.command.bits.shuffleLog2 > 0.U)
        assert(
          io.command.bits.shuffleLog2 <=
            log2Ceil(p.maxShuffleScale).U
        )
        val scale =
          1.U(p.shuffleScaleBits.W) <<
            io.command.bits.shuffleLog2
        assert(
          io.command.bits.outputHeight % scale === 0.U
        )
        assert(
          io.command.bits.outputWidth % scale === 0.U
        )
        assert(
          io.command.bits.inputHeight ===
            io.command.bits.outputHeight
        )
        assert(
          io.command.bits.inputWidth ===
            io.command.bits.outputWidth
        )
      }
    }.otherwise {
      assert(
        !io.command.bits.bilinearResidual,
        "bilinearResidual is only valid in columnReduce mode"
      )
    }

    commandRegister := io.command.bits
    val cfgInitialSpatial =
      -(io.command.bits.paddingY.zext *
        io.command.bits.inputWidth.zext +
        io.command.bits.paddingX.zext)
    val cfgTapRowAdvance =
      io.command.bits.inputWidth.zext -
        io.command.bits.kernelWidth.zext + 1.S
    initialSpatialStart := cfgInitialSpatial
    kernelSpatialStart := cfgInitialSpatial
    spatialTapRowAdvance := cfgTapRowAdvance
    scanConfigPending := true.B
    runtimeKernelElements :=
      io.command.bits.kernelHeight * io.command.bits.kernelWidth
    outputPixelCount := configuredOutputPixels
    operandsPerCycle := MuxLookup(
      io.command.bits.weightPrecision,
      1.U(p.weightOperandCountBits.W)
    )(
      Seq(
        16.U -> 1.U,
        8.U -> 2.U,
        4.U -> 4.U,
        2.U -> 8.U
      )
    )
    inputGroupCount := Mux(
      io.command.bits.depthwise || io.command.bits.columnReduce,
      1.U,
      MuxLookup(
        io.command.bits.weightPrecision,
        1.U(p.channelCountBits.W)
      )(
        Seq(
          16.U -> io.command.bits.inputChannels,
          8.U -> ((io.command.bits.inputChannels + 1.U) >> 1),
          4.U -> ((io.command.bits.inputChannels + 3.U) >> 2),
          2.U -> ((io.command.bits.inputChannels + 7.U) >> 3)
        )
      )
    )
    fullPrecision := io.command.bits.weightPrecision === 16.U
    val cfgRowAdvance =
      io.command.bits.inputWidth.zext -
        io.command.bits.outputWidth.zext + 1.S
    spatialRowAdvance := cfgRowAdvance
    denseRowAdvance :=
      cfgRowAdvance * io.command.bits.inputChannels.zext
    depthwiseRowAdvance :=
      cfgRowAdvance * io.command.bits.outputBlocks.zext
    block := 0.U
    pixel := 0.U
    kernelTap := 0.U
    outputX := 0.U
    outputY := 0.U
    kernelX := 0.U
    kernelY := 0.U
    postOutputX := 0.U
    postOutputY := 0.U
    nextCompletionOutputX := 0.U
    nextCompletionOutputY := 0.U
    postNormalOffsetRegister := 0.U
    nextPostNormalOffset := 0.U
    postBinaryOffsetRegister := 0.U
    nextPostBinaryOffset := 0.U
    inputGroup := 0.U
    inputGroupChannelBase := 0.U
    weightOperand := 0.U
    preloadActive := false.B
    preloadWaiting := false.B
    preloadValid := false.B
    preloadBlock := 0.U
    preloadKernelTap := 0.U
    preloadInputGroup := 0.U
    preloadOperand := 0.U
    weightLowOffsetRegister := 0.U
    weightHighOffsetRegister := 0.U
    preloadWeightLowOffsetRegister := 0.U
    preloadWeightHighOffsetRegister := 0.U
    activationOperand := 0.U
    activationBitIndex := 0.U
    arrayResultBitIndex := 0.U
    arrayBurstDone := false.B
    groupPartialSum := VecInit.fill(p.dim)(
      0.S(p.accumulatorBits.W)
    )
    weightFetchCount := 0.U
    outputBufferValid := false.B
    outputFullPending := false.B
    outputBinaryPending := false.B
    postState := postIdle
    state := initializeAccumulator
  }

  switch(state) {
    is(initializeAccumulator) {
      io.accReq.valid := true.B
      io.accReq.bits.operation := AccumulatorOperation.write
      io.accReq.bits.address := Mux(
        commandRegister.columnReduce,
        commandRegister.accumulatorBase + pixel,
        accumulatorAddress
      )
      for (lane <- 0 until p.dim) {
        io.accReq.bits.data(lane) := Mux(
          commandRegister.columnReduce,
          selectedBiases(0),
          selectedBiases(lane)
        )
      }
      io.accReq.bits.shift := 0.U
      io.accReq.bits.negate := false.B
      when(io.accReq.fire) {
        state := waitForInitialize
      }
    }

    is(waitForInitialize) {
      when(io.accResp.fire) {
        when(pixel === accumulatorRowCount - 1.U) {
          pixel := 0.U
          kernelTap := 0.U
          outputX := 0.U
          outputY := 0.U
          kernelX := 0.U
          kernelY := 0.U
          inputGroup := 0.U
          weightOperand := 0.U
          state := Mux(
            block === 0.U,
            issueWeightFetch,
            usePrefetchedWeights
          )
        }.otherwise {
          pixel := pixel + 1.U
          state := initializeAccumulator
        }
      }
    }

    is(issueWeightFetch) {
      io.wgtFetchReq.valid := true.B
      io.wgtFetchReq.bits.lowAddress :=
        commandRegister.weightLowBase +
          weightLowOffsetRegister + weightOperand
      io.wgtFetchReq.bits.highAddress :=
        commandRegister.weightHighBase + weightHighOffsetRegister
      io.wgtFetchReq.bits.highBytePresent :=
        fullPrecision
      when(io.wgtFetchReq.fire) {
        weightFetchCount := weightFetchCount + 1.U
        state := waitForWeightFetch
      }
    }

    is(waitForWeightFetch) {
      when(io.wgtFetchResp.fire) {
        stationaryWeights(weightOperand) :=
          io.wgtFetchResp.bits.weights
        when(isLastWeightOperand) {
          pixel := 0.U
          outputX := 0.U
          outputY := 0.U
          loadActivationScanStart()
          armNextTilePreload(
            weightLowOffsetRegister,
            weightHighOffsetRegister
          )
          state := processPixel
        }.otherwise {
          weightOperand := weightOperand + 1.U
          state := issueWeightFetch
        }
      }
    }

    is(usePrefetchedWeights) {
      when(preloadValid) {
        assert(preloadBlock === block)
        assert(preloadKernelTap === kernelTap)
        assert(preloadInputGroup === inputGroup)
        stationaryWeights := prefetchedWeights
        preloadValid := false.B
        weightLowOffsetRegister := preloadWeightLowOffsetRegister
        weightHighOffsetRegister := preloadWeightHighOffsetRegister
        armNextTilePreload(
          preloadWeightLowOffsetRegister,
          preloadWeightHighOffsetRegister
        )
        pixel := 0.U
        outputX := 0.U
        outputY := 0.U
        loadActivationScanStart()
        state := processPixel
      }
    }

    is(processPixel) {
      when(inputIsPadding) {
        when(isLastKernelTap && isLastInputGroup) {
          when(
            !commandRegister.columnReduce ||
              isAccumulatorRowEnd
          ) {
            state := issueAccumulatorRead
          }.otherwise {
            advanceAfterPixel()
          }
        }.otherwise {
          advanceAfterPixel()
        }
      }.otherwise {
        activationOperand := 0.U
        prepareActivationFetch(0.U)
        state := issueActivationFetch
      }
    }

    is(issueActivationFetch) {
      when(!activationFetchRequiredRegister) {
        fetchedActivations(activationOperand) := 0.U
        advanceAfterActivation()
      }.otherwise {
        io.actFetchReq.valid := true.B
        when(io.actFetchReq.fire) {
          state := waitForActivationFetch
        }
      }
    }

    is(waitForActivationFetch) {
      when(io.actFetchResp.fire) {
        when(
          commandRegister.depthwise ||
            commandRegister.columnReduce
        ) {
          fetchedActivationRow := io.actFetchResp.bits.row
          activationBitIndex := 0.U
          arrayResultBitIndex := 0.U
          arrayBurstDone := false.B
          groupPartialSum := VecInit.fill(p.dim)(
            0.S(p.accumulatorBits.W)
          )
          state := issueArrayCompute
        }.otherwise {
          fetchedActivations(activationOperand) :=
            io.actFetchResp.bits.activation
          advanceAfterActivation()
        }
      }
    }

    is(issueArrayCompute) {
      when(
        activationBitIndex ===
          commandRegister.activationPrecision - 1.U
      ) {
        state := waitForArrayResult
      }.otherwise {
        activationBitIndex := activationBitIndex + 1.U
      }
    }

    is(waitForArrayResult) {
      val finalArrayResult =
        arrayResultValid &&
          arrayResultBitIndex === commandRegister.activationPrecision - 1.U
      when(finalArrayResult) {
        accumulatorTaskQueue.io.enq.valid := true.B
        accumulatorTaskQueue.io.enq.bits.readOnly := false.B
        accumulatorTaskQueue.io.enq.bits.address := accumulatorAddress
        accumulatorTaskQueue.io.enq.bits.data := accumulatedArrayResult
        accumulatorTaskQueue.io.enq.bits.completesOutput :=
          isLastKernelTap &&
            isLastInputGroup &&
            (!commandRegister.columnReduce || isAccumulatorRowEnd)
        accumulatorTaskQueue.io.enq.bits.block := block
        accumulatorTaskQueue.io.enq.bits.pixel := pixel
        accumulatorTaskQueue.io.enq.bits.outputX :=
          nextCompletionOutputX
        accumulatorTaskQueue.io.enq.bits.outputY :=
          nextCompletionOutputY
        assert(
          accumulatorTaskQueue.io.enq.ready,
          "multi-bit array completion must enter Accumulator without a bubble"
        )
        when(accumulatorTaskQueue.io.enq.fire) {
          advanceAfterAcc()
        }
      }.elsewhen(arrayBurstDone) {
        // Defensive fallback for a delayed control observation. The normal
        // fixed-latency path consumes the last AddTree result above.
        state := issueBitAccumulate
      }
    }

    is(issueBitAccumulate) {
      accumulatorTaskQueue.io.enq.valid := true.B
      accumulatorTaskQueue.io.enq.bits.readOnly := false.B
      accumulatorTaskQueue.io.enq.bits.address :=
        accumulatorAddress
      accumulatorTaskQueue.io.enq.bits.data :=
        groupPartialSum
      accumulatorTaskQueue.io.enq.bits.completesOutput :=
        isLastKernelTap &&
          isLastInputGroup &&
          (!commandRegister.columnReduce ||
            isAccumulatorRowEnd)
      accumulatorTaskQueue.io.enq.bits.block := block
      accumulatorTaskQueue.io.enq.bits.pixel := pixel
      accumulatorTaskQueue.io.enq.bits.outputX := nextCompletionOutputX
      accumulatorTaskQueue.io.enq.bits.outputY := nextCompletionOutputY
      when(accumulatorTaskQueue.io.enq.fire) {
        advanceAfterAcc()
      }
    }

    is(issueAccumulatorRead) {
      accumulatorTaskQueue.io.enq.valid := true.B
      accumulatorTaskQueue.io.enq.bits.readOnly := true.B
      accumulatorTaskQueue.io.enq.bits.address :=
        accumulatorAddress
      accumulatorTaskQueue.io.enq.bits.data :=
        VecInit.fill(p.dim)(0.S(p.accumulatorBits.W))
      accumulatorTaskQueue.io.enq.bits.completesOutput := true.B
      accumulatorTaskQueue.io.enq.bits.block := block
      accumulatorTaskQueue.io.enq.bits.pixel := pixel
      accumulatorTaskQueue.io.enq.bits.outputX := nextCompletionOutputX
      accumulatorTaskQueue.io.enq.bits.outputY := nextCompletionOutputY
      when(accumulatorTaskQueue.io.enq.fire) {
        advanceAfterAcc()
      }
    }

    is(waitForOutputDrain) {
      when(
        !accumulatorTaskQueue.io.deq.valid &&
          accumulatorOutstanding === 0.U &&
          !completionQueue.io.deq.valid &&
          postState === postIdle &&
          !outputBufferValid
      ) {
        finishOutputBlock()
      }
    }
  }

  // Completion vectors are produced in raster order. Tag each one with its
  // first output coordinate and derive the next tag from the current final
  // pixel. This stays one compare/add stage regardless of p.dim and avoids a
  // p.dim-deep coordinate-update chain in the post-processing path.
  when(
    accumulatorTaskQueue.io.enq.fire &&
      accumulatorTaskQueue.io.enq.bits.completesOutput
  ) {
    when(outputX === commandRegister.outputWidth - 1.U) {
      nextCompletionOutputX := 0.U
      nextCompletionOutputY := outputY + 1.U
    }.otherwise {
      nextCompletionOutputX := outputX + 1.U
      nextCompletionOutputY := outputY
    }
  }
}
