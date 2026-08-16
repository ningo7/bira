package bira

// Multi-bit convolution controller.

import chisel3._
import chisel3.util._

/** Completed convolution row waiting for post-processing and writeback. */
class ConvCompletion(p: BiRaParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
  val accumulator =
    Vec(p.dim, SInt(p.accumulatorBits.W))
}

/** One decoupled Accumulator transaction produced by the array pipeline. */
class ConvAccumulatorTask(p: BiRaParams) extends Bundle {
  val readOnly = Bool()
  val address = UInt(p.accumulatorAddressBits.W)
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
  val completesOutput = Bool()
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
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
class ConvCtrl(p: BiRaParams) extends Module {
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
      Decoupled(new BilinearInterpolationRequest(p))
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
    activatePrefetchedWeights,
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
    postIssueResidualInterpolation,
    postWaitForResidualInterpolation,
    postIssueProcess,
    postWaitForProcess,
    postEnqueueOutput
  ) = Enum(6)

  private val state = RegInit(idle)
  private val postState = RegInit(postIdle)
  private val accIdle :: accIssue :: accWait :: Nil = Enum(3)
  private val accState = RegInit(accIdle)
  private val commandRegister =
    RegInit(0.U.asTypeOf(new ConvolutionCommand(p)))
  private val block = RegInit(0.U(p.blockIndexBits.W))
  private val pixel = RegInit(0.U(p.pixelIndexBits.W))
  private val kernelTap = RegInit(0.U(p.kernelIndexBits.W))
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
  private val postAccumulator =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val postBlock = RegInit(0.U(p.blockIndexBits.W))
  private val postPixel = RegInit(0.U(p.pixelIndexBits.W))
  private val groupPartialSum =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val completedPostOutput =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val residualRow =
    Reg(Vec(p.dim, UInt(p.activationBits.W)))
  private val completionQueue =
    Module(new Queue(new ConvCompletion(p), entries = 4))
  private val accumulatorTaskQueue =
    Module(new Queue(new ConvAccumulatorTask(p), entries = 8))
  private val activeAccumulatorTask =
    Reg(new ConvAccumulatorTask(p))
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

  private val operandsPerCycle = MuxLookup(
    commandRegister.weightPrecision,
    0.U(p.weightOperandCountBits.W)
  )(
    Seq(
      16.U -> 1.U,
      8.U -> 2.U,
      4.U -> 4.U,
      2.U -> 8.U
    )
  )
  private val denseInputGroupCount = MuxLookup(
      commandRegister.weightPrecision,
      0.U(p.channelCountBits.W)
    )(
      Seq(
        16.U -> commandRegister.inputChannels,
        8.U -> ((commandRegister.inputChannels + 1.U) >> 1),
        4.U -> ((commandRegister.inputChannels + 3.U) >> 2),
        2.U -> ((commandRegister.inputChannels + 7.U) >> 3)
      )
    )
  private val inputGroupCount = Mux(
    commandRegister.depthwise || commandRegister.columnReduce,
    1.U,
    denseInputGroupCount
  )
  private val runtimeKernelElements =
    commandRegister.kernelHeight * commandRegister.kernelWidth
  private val outputPixelCount =
    commandRegister.outputHeight * commandRegister.outputWidth
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

  private val outputY = pixel / commandRegister.outputWidth
  private val outputX = pixel % commandRegister.outputWidth
  private val kernelY = kernelTap / commandRegister.kernelWidth
  private val kernelX = kernelTap % commandRegister.kernelWidth
  private val signedInputY =
    outputY.zext + kernelY.zext - commandRegister.paddingY.zext
  private val signedInputX =
    outputX.zext + kernelX.zext - commandRegister.paddingX.zext
  private val inputIsPadding =
    signedInputY < 0.S ||
      signedInputY >= commandRegister.inputHeight.zext ||
      signedInputX < 0.S ||
      signedInputX >= commandRegister.inputWidth.zext

  private val activeInputChannel =
    inputGroup * operandsPerCycle + activationOperand
  private val inputSpatialIndex =
    signedInputY.asUInt * commandRegister.inputWidth +
      signedInputX.asUInt
  private val inputLinearIndex =
    inputSpatialIndex * commandRegister.inputChannels +
      activeInputChannel
  private val inputVectorOffset =
    inputLinearIndex >> p.laneIndexBits
  private val inputLane =
    inputLinearIndex(p.laneIndexBits - 1, 0)
  private val depthwiseInputAddress =
    commandRegister.inputBase +
      inputSpatialIndex * commandRegister.outputBlocks + block
  private val reducedInputAddress =
    commandRegister.inputBase + (inputSpatialIndex >> 1)

  private val weightTileIndex =
    (block * runtimeKernelElements + kernelTap) *
      inputGroupCount + inputGroup
  private val lowPrecisionWeightOffset =
    weightTileIndex * operandsPerCycle + weightOperand
  private val weightLowOffset = Mux(
    commandRegister.weightPrecision === 16.U,
    weightTileIndex,
    lowPrecisionWeightOffset
  )
  private val preloadTileIndex =
    (preloadBlock * runtimeKernelElements + preloadKernelTap) *
      inputGroupCount + preloadInputGroup
  private val preloadLowPrecisionWeightOffset =
    preloadTileIndex * operandsPerCycle + preloadOperand
  private val preloadWeightLowOffset = Mux(
    commandRegister.weightPrecision === 16.U,
    preloadTileIndex,
    preloadLowPrecisionWeightOffset
  )
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
  private val shuffledWidth =
    commandRegister.outputWidth * shuffleScale
  private val postOutputY =
    postPixel / commandRegister.outputWidth
  private val postOutputX =
    postPixel % commandRegister.outputWidth
  private val postShuffleSubPixel0 = postBlock << 1
  private val postShuffleSubY =
    postShuffleSubPixel0 >> commandRegister.shuffleLog2
  private val postShuffleSubX =
    postShuffleSubPixel0 & (shuffleScale - 1.U)
  private val postShuffledY =
    postOutputY * shuffleScale + postShuffleSubY
  private val postShuffledX =
    postOutputX * shuffleScale + postShuffleSubX
  private val postShuffledPixel0 =
    postShuffledY * shuffledWidth + postShuffledX
  private val postNormalOutputOffset =
    postPixel * commandRegister.outputBlocks + postBlock
  private val postFullOutputOffset = Mux(
    commandRegister.columnReduce,
    postPixel >> p.laneIndexBits,
    Mux(
      commandRegister.shufflePack2,
      postShuffledPixel0 >> 1,
      postNormalOutputOffset
    )
  )

  /** Start filling the inactive weight buffer with the next global tile. */
  private def armNextTilePreload(): Unit = {
    preloadOperand := 0.U
    preloadWaiting := false.B
    preloadValid := false.B
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
      weightOperand := 0.U
      when(isLastInputGroup) {
        inputGroup := 0.U
        when(isLastKernelTap) {
          state := issueAccumulatorRead
        }.otherwise {
          kernelTap := kernelTap + 1.U
          state := activatePrefetchedWeights
        }
      }.otherwise {
        inputGroup := inputGroup + 1.U
        state := activatePrefetchedWeights
      }
    }.otherwise {
      pixel := pixel + 1.U
      state := processPixel
    }
  }

  private def advanceAfterQueuedAccumulator(): Unit = {
    when(isLastPixel && isLastKernelTap && isLastInputGroup) {
      pixel := 0.U
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
      activationOperand := activationOperand + 1.U
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
      inputGroup := 0.U
      weightOperand := 0.U
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
      accState =/= accIdle ||
      accumulatorTaskQueue.io.deq.valid ||
      postState =/= postIdle ||
      completionQueue.io.deq.valid ||
      outputBufferValid
  io.status.done := donePulse

  io.actFetchReq.valid := false.B
  io.actFetchReq.bits :=
    0.U.asTypeOf(new ActivationFetchRequest(p))
  io.actFetchResp.ready := state === waitForActivationFetch

  io.wgtFetchReq.valid := false.B
  io.wgtFetchReq.bits :=
    0.U.asTypeOf(new WeightFetchRequest(p))
  io.wgtFetchResp.ready :=
    state === waitForWeightFetch || preloadWaiting

  io.interpolationReq.valid := false.B
  io.interpolationReq.bits :=
    0.U.asTypeOf(new BilinearInterpolationRequest(p))
  io.interpolationResp.ready :=
    postState === postWaitForResidualInterpolation

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

  for (operand <- 0 until p.maxWeightOperands) {
    for (lane <- 0 until p.dim) {
      val reducedActivation =
        if (lane < p.dim / 2) {
          operand.U === 0.U &&
            Mux(
              inputSpatialIndex(0),
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
  when(
    arrayResultValid &&
      (state === issueArrayCompute ||
        state === waitForArrayResult)
  ) {
    for (lane <- 0 until p.dim) {
      val signedPartial = Mux(
        commandRegister.inputSigned &&
          arrayResultBitIndex ===
            commandRegister.activationPrecision - 1.U,
        -arrayResultVector(lane),
        arrayResultVector(lane)
      )
      val shiftedWide =
        signedPartial << arrayResultBitIndex
      val shifted =
        shiftedWide(p.accumulatorBits - 1, 0).asSInt
      groupPartialSum(lane) := Mux(
        arrayResultBitIndex === 0.U,
        shifted,
        (groupPartialSum(lane) + shifted)(
          p.accumulatorBits - 1,
          0
        ).asSInt
      )
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
      commandRegister.weightLowBase + preloadWeightLowOffset
    io.wgtFetchReq.bits.highAddress :=
      commandRegister.weightHighBase + preloadTileIndex
    io.wgtFetchReq.bits.highBytePresent :=
      commandRegister.weightPrecision === 16.U
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

  // Array work is queued independently from the single ordered Accumulator
  // port. The main controller can fetch and reduce the next pixel while this
  // engine commits the preceding pixel. Final responses feed the post queue
  // directly without returning to the compute state machine.
  switch(accState) {
    is(accIdle) {
      when(accumulatorTaskQueue.io.deq.valid) {
        accumulatorTaskQueue.io.deq.ready := true.B
        activeAccumulatorTask :=
          accumulatorTaskQueue.io.deq.bits
        accState := accIssue
      }
    }

    is(accIssue) {
      io.accReq.valid := true.B
      io.accReq.bits.operation := Mux(
        activeAccumulatorTask.readOnly,
        AccumulatorOperation.read,
        AccumulatorOperation.add
      )
      io.accReq.bits.address :=
        activeAccumulatorTask.address
      io.accReq.bits.data :=
        activeAccumulatorTask.data
      io.accReq.bits.shift := 0.U
      io.accReq.bits.negate := false.B
      when(io.accReq.fire) {
        accState := accWait
      }
    }

    is(accWait) {
      completionQueue.io.enq.valid :=
        io.accResp.valid &&
          activeAccumulatorTask.completesOutput
      completionQueue.io.enq.bits.block :=
        activeAccumulatorTask.block
      completionQueue.io.enq.bits.pixel :=
        activeAccumulatorTask.pixel
      completionQueue.io.enq.bits.accumulator :=
        io.accResp.bits.data
      io.accResp.ready := Mux(
        activeAccumulatorTask.completesOutput,
        completionQueue.io.enq.ready,
        true.B
      )
      when(io.accResp.fire) {
        accState := accIdle
      }
    }
  }

  // Completion processing is independent of the convolution state machine.
  // The compute side may continue producing later pixels until this queue
  // fills, while this side performs interpolation, post-processing, and SPAD
  // writeback in program order.
  switch(postState) {
    is(postIdle) {
      when(completionQueue.io.deq.valid) {
        completionQueue.io.deq.ready := true.B
        postAccumulator :=
          completionQueue.io.deq.bits.accumulator
        postBlock := completionQueue.io.deq.bits.block
        postPixel := completionQueue.io.deq.bits.pixel
        residualRow :=
          VecInit.fill(p.dim)(0.U(p.activationBits.W))
        postState := Mux(
          commandRegister.columnReduce &&
            commandRegister.bilinearResidual,
          postIssueResidualInterpolation,
          postIssueProcess
        )
      }
    }

    is(postIssueResidualInterpolation) {
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
      io.interpolationReq.bits.outputPixelCount :=
        outputPixelCount
      io.interpolationReq.bits.scaleLog2 :=
        commandRegister.shuffleLog2
      when(io.interpolationReq.fire) {
        postState := postWaitForResidualInterpolation
      }
    }

    is(postWaitForResidualInterpolation) {
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
        outputBufferValid := true.B
        outputFullPending := commandRegister.writeFullOutput
        outputBinaryPending := commandRegister.writeBinaryOutput
        outputFullAddress :=
          commandRegister.outputBase + postFullOutputOffset
        outputBinaryAddress :=
          commandRegister.binaryOutputBase +
            postPixel * commandRegister.outputBlocks + postBlock
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
    val configuredAccumulatorRows = Mux(
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
        configuredAccumulatorRows <=
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
    block := 0.U
    pixel := 0.U
    kernelTap := 0.U
    inputGroup := 0.U
    weightOperand := 0.U
    preloadActive := false.B
    preloadWaiting := false.B
    preloadValid := false.B
    preloadBlock := 0.U
    preloadKernelTap := 0.U
    preloadInputGroup := 0.U
    preloadOperand := 0.U
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
          inputGroup := 0.U
          weightOperand := 0.U
          state := Mux(
            block === 0.U,
            issueWeightFetch,
            activatePrefetchedWeights
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
        commandRegister.weightLowBase + weightLowOffset
      io.wgtFetchReq.bits.highAddress :=
        commandRegister.weightHighBase + weightTileIndex
      io.wgtFetchReq.bits.highBytePresent :=
        commandRegister.weightPrecision === 16.U
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
          armNextTilePreload()
          state := processPixel
        }.otherwise {
          weightOperand := weightOperand + 1.U
          state := issueWeightFetch
        }
      }
    }

    is(activatePrefetchedWeights) {
      when(preloadValid) {
        assert(preloadBlock === block)
        assert(preloadKernelTap === kernelTap)
        assert(preloadInputGroup === inputGroup)
        stationaryWeights := prefetchedWeights
        preloadValid := false.B
        armNextTilePreload()
        pixel := 0.U
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
        state := issueActivationFetch
      }
    }

    is(issueActivationFetch) {
      when(
        !commandRegister.depthwise &&
          activeInputChannel >= commandRegister.inputChannels
      ) {
        fetchedActivations(activationOperand) := 0.U
        advanceAfterActivation()
      }.otherwise {
        io.actFetchReq.valid := true.B
        io.actFetchReq.bits.address := Mux(
          commandRegister.columnReduce,
          reducedInputAddress,
          Mux(
            commandRegister.depthwise,
            depthwiseInputAddress,
            commandRegister.inputBase + inputVectorOffset
          )
        )
        io.actFetchReq.bits.lane :=
          Mux(commandRegister.depthwise, 0.U, inputLane)
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
      when(arrayBurstDone) {
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
      when(accumulatorTaskQueue.io.enq.fire) {
        advanceAfterQueuedAccumulator()
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
      when(accumulatorTaskQueue.io.enq.fire) {
        advanceAfterQueuedAccumulator()
      }
    }

    is(waitForOutputDrain) {
      when(
        !accumulatorTaskQueue.io.deq.valid &&
          accState === accIdle &&
          !completionQueue.io.deq.valid &&
          postState === postIdle &&
          !outputBufferValid
      ) {
        finishOutputBlock()
      }
    }
  }
}
