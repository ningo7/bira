package bira

// Binary convolution controller.

import chisel3._
import chisel3.util._

class BinaryAccumulatorTask(p: AccelParams) extends Bundle {
  val readOnly = Bool()
  val address = UInt(p.accumulatorAddressBits.W)
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
  val completesOutput = Bool()
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
}

class BinaryCompletion(p: AccelParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
  val accumulator = Vec(p.dim, SInt(p.accumulatorBits.W))
}

/** Runtime-configured packed binary convolution controller.
  *
  * Loop order:
  *
  *   output block
  *     -> kernel tap
  *       -> input-channel block
  *         -> output pixel
  *
  * A binary weight tile contains `dim` packed input-channel bits for each of
  * `dim` output lanes. The tile is loaded once and remains stationary while
  * all output pixels are visited.
  *
  * Accumulators start from zero. Each valid array result is an equality
  * popcount accumulated with shift=1, so the completed value is
  * `2 * popcount`. Before fused scale/RPReLU, the controller reads the
  * compiler-programmed per-pixel `-N` from the Parameter Buffer. The
  * post-processing input is therefore exactly `2 * popcount - N`.
  */
class BinConvCtrl(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val command =
      Flipped(Decoupled(new BinaryConvolutionCommand(p)))
    val status = Output(new AcceleratorStatus)

    val postParameters = Input(
      Vec(
        p.maxOutputBlocks,
        Vec(p.dim, new BinPostParams(p))
      )
    )
    val outputSignThresholds = Input(
      Vec(
        p.maxOutputBlocks,
        Vec(p.dim, SInt(p.accumulatorBits.W))
      )
    )

    val actReadReq =
      Decoupled(UInt(p.binaryAddressBits.W))
    val actReadResp =
      Flipped(Valid(Vec(p.dim, UInt(1.W))))
    val wgtReadReq =
      Decoupled(UInt(p.binaryAddressBits.W))
    val wgtReadResp =
      Flipped(Valid(Vec(p.dim, UInt(1.W))))
    val resReadReq =
      Decoupled(UInt(p.fullAddressBits.W))
    val resReadResp =
      Flipped(Valid(Vec(p.dim, UInt(p.activationBits.W))))

    val accReq =
      Decoupled(new AccumulatorRequest(p))
    val accResp =
      Flipped(Decoupled(new AccumulatorResponse(p)))
    val correctionReadReq =
      Decoupled(new CorrectionReq(p))
    val correctionReadResp =
      Flipped(Decoupled(UInt(512.W)))

    val arrayAct = Output(UInt(p.dim.W))
    val arrayWgts = Output(Vec(p.dim, UInt(p.dim.W)))
    val arrayInValid = Output(Bool())
    val arraySums =
      Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val arrayOutValid = Input(Bool())

    val postAcc =
      Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val postCorrection = Output(SInt(p.accumulatorBits.W))
    val postRes =
      Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val postInValid = Output(Bool())
    val postParams =
      Output(Vec(p.dim, new BinPostParams(p)))
    val postOut =
      Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
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
  })

  private val Seq(
    idle,
    issueAccInit,
    waitAccInit,
    issueWeightRead,
    waitForWeightRead,
    processPixel,
    issueActivationRead,
    waitForActivationRead,
    issueArrayCompute,
    waitForArrayResult,
    issuePopcountAccumulate,
    issueFinalCorrectionRead,
    waitFinalCorrectionRead,
    streamPixels,
    drainArrayStream,
    issueAccumulatorRead,
    waitForOutputDrain
  ) = Enum(17)

  private val state = RegInit(idle)
  private val Seq(
    postIdle,
    postIssueResidualRead,
    postWaitResidualRead,
    postIssueCorrectionRead,
    postWaitCorrectionRead,
    postIssueProcess,
    postWaitProcess,
    postEnqueueOutput
  ) = Enum(8)
  private val postState = RegInit(postIdle)
  private val commandRegister =
    RegInit(0.U.asTypeOf(new BinaryConvolutionCommand(p)))
  private val block = RegInit(0.U(p.blockIndexBits.W))
  private val pixel = RegInit(0.U(p.pixelIndexBits.W))
  private val kernelTap = RegInit(0.U(p.kernelIndexBits.W))
  private val outputX = RegInit(0.U(p.imageDimensionBits.W))
  private val outputY = RegInit(0.U(p.imageDimensionBits.W))
  private val kernelX = RegInit(0.U(p.kernelDimensionBits.W))
  private val kernelY = RegInit(0.U(p.kernelDimensionBits.W))
  private val inputBlock = RegInit(0.U(p.channelCountBits.W))
  private val weightLane = RegInit(0.U(p.laneIndexBits.W))
  private val stationaryWeights =
    Reg(Vec(p.dim, UInt(p.dim.W)))
  private val prefetchedWeights =
    Reg(Vec(p.dim, UInt(p.dim.W)))
  private val preloadActive = RegInit(false.B)
  private val preloadWaiting = RegInit(false.B)
  private val preloadValid = RegInit(false.B)
  private val preloadBlock = RegInit(0.U(p.blockIndexBits.W))
  private val preloadKernelTap = RegInit(0.U(p.kernelIndexBits.W))
  private val preloadInputBlock = RegInit(0.U(p.channelCountBits.W))
  private val preloadWeightLane = RegInit(0.U(p.laneIndexBits.W))
  private val weightLaneStride = Reg(UInt(p.binaryAddressBits.W))
  private val blockWeightAdvance = Reg(UInt(p.binaryAddressBits.W))
  private val weightTileBase = Reg(UInt(p.binaryAddressBits.W))
  private val weightReadAddress = Reg(UInt(p.binaryAddressBits.W))
  private val preloadWeightTileBase = Reg(UInt(p.binaryAddressBits.W))
  private val preloadWeightReadAddress = Reg(UInt(p.binaryAddressBits.W))
  private val activationRegister = Reg(UInt(p.dim.W))
  private val actAddrReg =
    Reg(UInt(p.binaryAddressBits.W))
  private val initialActivationStart =
    Reg(SInt((p.binaryAddressBits + 2).W))
  private val kernelActivationStart =
    Reg(SInt((p.binaryAddressBits + 2).W))
  private val initialSpatialStart =
    Reg(SInt((p.pixelIndexBits + 2).W))
  private val pixelSpatialRowAdvance =
    Reg(SInt((p.pixelIndexBits + 2).W))
  private val tapSpatialRowAdvance =
    Reg(SInt((p.pixelIndexBits + 2).W))
  private val activationTapRowAdvance =
    Reg(SInt((p.binaryAddressBits + 2).W))
  private val scanConfigPending = RegInit(false.B)
  private val activationRowAdvance =
    Reg(SInt((p.binaryAddressBits + 2).W))
  private val streamInputPixel = Reg(UInt(p.pixelIndexBits.W))
  private val streamInputPadding = RegInit(false.B)
  private val streamInputValid = RegInit(false.B)
  private val streamTagStage1 = Reg(UInt(p.pixelIndexBits.W))
  private val streamTagStage2 = Reg(UInt(p.pixelIndexBits.W))
  private val streamTagStage3 = Reg(UInt(p.pixelIndexBits.W))
  private val streamPaddingStage1 = RegInit(false.B)
  private val streamPaddingStage2 = RegInit(false.B)
  private val streamPaddingStage3 = RegInit(false.B)
  private val streamTagStage1Valid = RegInit(false.B)
  private val streamTagStage2Valid = RegInit(false.B)
  private val streamTagStage3Valid = RegInit(false.B)
  private val streamOutstanding = RegInit(0.U((p.pixelIndexBits + 1).W))
  private val finalStreamActive = RegInit(false.B)
  private val finalOutputOutstanding =
    RegInit(0.U((p.pixelIndexBits + 1).W))
  private val finalAccumulator =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val finalAccumulatorPixel = Reg(UInt(p.pixelIndexBits.W))
  private val finalResidualValid = RegInit(false.B)
  private val finalPostTagValid =
    RegInit(VecInit(Seq.fill(BinPostProc.latency)(false.B)))
  private val finalResidualAddress = Reg(UInt(p.fullAddressBits.W))
  private val finalFullOutputAddress = Reg(UInt(p.fullAddressBits.W))
  private val finalBinaryOutputAddress = Reg(UInt(p.binaryAddressBits.W))
  private val correctionNextRegister = Reg(UInt(512.W))
  private val correctionNextValid = RegInit(false.B)
  private val correctionPrefetchValid = RegInit(false.B)
  private val corrPrefetchPending = RegInit(false.B)
  private val corrPrefetchAddr = Reg(UInt(p.parameterAddressBits.W))
  private val correctionRegister =
    Reg(UInt(512.W))
  private val postAccumulator =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val postBlock = RegInit(0.U(p.blockIndexBits.W))
  private val postPixel = RegInit(0.U(p.pixelIndexBits.W))
  private val postResidualAddressRegister = Reg(UInt(p.fullAddressBits.W))
  private val postFullAddressRegister = Reg(UInt(p.fullAddressBits.W))
  private val postBinaryAddressRegister = Reg(UInt(p.binaryAddressBits.W))
  private val nextPostResidualAddress = Reg(UInt(p.fullAddressBits.W))
  private val nextPostFullAddress = Reg(UInt(p.fullAddressBits.W))
  private val nextPostBinaryAddress = Reg(UInt(p.binaryAddressBits.W))
  private val arrayColumnSumRegister =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val residualRegister =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val completedPostOutput =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
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
  // Two entries keep the steady-state request stream at II=1 while making
  // enq.ready depend only on registered occupancy.  A one-entry elastic
  // register propagates Accumulator/SPAD backpressure into the coordinate
  // state update and creates a long cross-controller combinational path.
  private val accumulatorTaskQueue = Module(
    new Queue(
      new BinaryAccumulatorTask(p),
      entries = 2,
      pipe = false,
      flow = false
    )
  )
  private val accumulatorOutstanding = RegInit(0.U(2.W))
  private val completionQueue = Module(
    new Queue(
      new BinaryCompletion(p),
      entries = 2,
      pipe = false,
      flow = false
    )
  )

  private val runtimeKernelElements =
    RegInit(1.U((2 * p.kernelDimensionBits).W))
  private val outputPixelCount =
    RegInit(1.U((p.pixelIndexBits + 1).W))
  private val inputBlockCount =
    RegInit(1.U(p.channelCountBits.W))

  private val isLastPixel = pixel === outputPixelCount - 1.U
  private val isLastKernelTap =
    kernelTap === runtimeKernelElements - 1.U
  private val isLastInputBlock =
    inputBlock === inputBlockCount - 1.U
  private val isLastWeightLane = weightLane === (p.dim - 1).U

  private val signedInputY =
    outputY.zext + kernelY.zext - commandRegister.paddingY.zext
  private val signedInputX =
    outputX.zext + kernelX.zext - commandRegister.paddingX.zext
  private val inputIsPadding =
    signedInputY < 0.S ||
      signedInputY >= commandRegister.inputHeight.zext ||
      signedInputX < 0.S ||
      signedInputX >= commandRegister.inputWidth.zext
  private val accumulatorAddress =
    commandRegister.accumulatorBase + pixel
  private val postCorrectionAddress =
    commandRegister.correctionBase +
      (postPixel >> p.correctionEntryIndexBits)
  private val postCorrectionEntry =
    postPixel.pad(p.correctionEntryIndexBits)(
      p.correctionEntryIndexBits - 1,
      0
    )
  private val postResidualAddress = postResidualAddressRegister
  private val postFullStateAddress = postFullAddressRegister
  private val postBinaryOutputAddress = postBinaryAddressRegister

  private val selectedPostParameters =
    if (p.maxOutputBlocks == 1) io.postParameters(0)
    else {
      val bits = log2Ceil(p.maxOutputBlocks)
      io.postParameters(postBlock(bits - 1, 0))
    }
  private val selectedSignThresholds =
    if (p.maxOutputBlocks == 1) io.outputSignThresholds(0)
    else {
      val bits = log2Ceil(p.maxOutputBlocks)
      io.outputSignThresholds(postBlock(bits - 1, 0))
    }

  private def armNextTilePreload(currentTileBase: UInt): Unit = {
    val nextTileBase = Mux(
      isLastInputBlock && isLastKernelTap,
      currentTileBase + blockWeightAdvance,
      currentTileBase + 1.U
    )
    preloadWeightLane := 0.U
    preloadWaiting := false.B
    preloadValid := false.B
    preloadWeightTileBase := nextTileBase
    preloadWeightReadAddress := nextTileBase
    when(!isLastInputBlock) {
      preloadBlock := block
      preloadKernelTap := kernelTap
      preloadInputBlock := inputBlock + 1.U
      preloadActive := true.B
    }.elsewhen(!isLastKernelTap) {
      preloadBlock := block
      preloadKernelTap := kernelTap + 1.U
      preloadInputBlock := 0.U
      preloadActive := true.B
    }.elsewhen(block =/= commandRegister.outputBlocks - 1.U) {
      preloadBlock := block + 1.U
      preloadKernelTap := 0.U
      preloadInputBlock := 0.U
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
      when(isLastInputBlock) {
        inputBlock := 0.U
        when(isLastKernelTap) {
          kernelTap := 0.U
          kernelX := 0.U
          kernelY := 0.U
          state := issueAccumulatorRead
        }.otherwise {
          kernelTap := kernelTap + 1.U
          when(kernelX === commandRegister.kernelWidth - 1.U) {
            kernelX := 0.U
            kernelY := kernelY + 1.U
            kernelActivationStart :=
              kernelActivationStart + activationTapRowAdvance
          }.otherwise {
            kernelX := kernelX + 1.U
            kernelActivationStart :=
              kernelActivationStart + inputBlockCount.zext
          }
          state := issueWeightRead
        }
      }.otherwise {
        inputBlock := inputBlock + 1.U
        state := issueWeightRead
      }
    }.otherwise {
      pixel := pixel + 1.U
      actAddrReg :=
        Mux(
          outputX === commandRegister.outputWidth - 1.U,
          (actAddrReg.zext + activationRowAdvance).asUInt,
          actAddrReg + inputBlockCount
        )(p.binaryAddressBits - 1, 0)
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
    when(isLastPixel && isLastKernelTap && isLastInputBlock) {
      pixel := 0.U
      outputX := 0.U
      outputY := 0.U
      state := waitForOutputDrain
    }.otherwise {
      advanceAfterPixel()
    }
  }

  private def finishOutputBlock(): Unit = {
    when(block === commandRegister.outputBlocks - 1.U) {
      val expectedWeightFetches =
        commandRegister.outputBlocks *
          runtimeKernelElements *
          inputBlockCount *
          p.dim.U
      assert(
        weightFetchCount === expectedWeightFetches,
        "every packed binary weight vector must be fetched once"
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
      inputBlock := 0.U
      weightLane := 0.U
      kernelActivationStart := initialActivationStart
      nextPostResidualAddress :=
        commandRegister.residualBase + block + 1.U
      nextPostFullAddress :=
        commandRegister.fullOutputBase + block + 1.U
      nextPostBinaryAddress :=
        commandRegister.binaryOutputBase + block + 1.U
      state := issueAccInit
    }
  }

  donePulse := false.B
  io.command.ready :=
    state === idle &&
      accumulatorOutstanding === 0.U &&
      !accumulatorTaskQueue.io.deq.valid &&
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

  io.actReadReq.valid := false.B
  io.actReadReq.bits := actAddrReg
  io.wgtReadReq.valid := false.B
  io.wgtReadReq.bits := weightReadAddress
  io.resReadReq.valid := false.B
  io.resReadReq.bits := postResidualAddress
  io.correctionReadReq.valid := false.B
  io.correctionReadReq.bits.address := postCorrectionAddress
  io.correctionReadResp.ready :=
    postState === postWaitCorrectionRead ||
      state === waitFinalCorrectionRead ||
      corrPrefetchPending

  io.accReq.valid := false.B
  io.accReq.bits :=
    0.U.asTypeOf(new AccumulatorRequest(p))
  io.accResp.ready :=
    state === waitAccInit

  private val streamPathActive =
    state === streamPixels || state === drainArrayStream
  private val streamActivationIssued =
    state === streamPixels &&
      (inputIsPadding || io.actReadReq.fire)
  private val streamArrayInputValid =
    streamPathActive && streamInputValid
  io.arrayAct := Mux(
    streamArrayInputValid && !streamInputPadding,
    io.actReadResp.bits.asUInt,
    activationRegister
  )
  io.arrayWgts := stationaryWeights
  io.arrayInValid :=
    state === issueArrayCompute || streamArrayInputValid

  // Binary dispatch adds one registered canonical-input stage before the
  // two-stage ComputeArray. Pixel tags follow the resulting three-cycle
  // the same register cadence so array results can enter Accumulator without
  // returning to a per-pixel wait state.
  streamInputValid := streamActivationIssued
  streamTagStage1Valid := streamArrayInputValid
  streamTagStage2Valid := streamTagStage1Valid
  streamTagStage3Valid := streamTagStage2Valid
  when(streamArrayInputValid) {
    streamTagStage1 := streamInputPixel
    streamPaddingStage1 := streamInputPadding
    when(!streamInputPadding) {
      assert(io.actReadResp.valid, "activation response must align with stream token")
    }
  }
  when(streamTagStage1Valid) {
    streamTagStage2 := streamTagStage1
    streamPaddingStage2 := streamPaddingStage1
  }
  when(streamTagStage2Valid) {
    streamTagStage3 := streamTagStage2
    streamPaddingStage3 := streamPaddingStage2
  }

  io.postAcc := Mux(finalStreamActive, finalAccumulator, postAccumulator)
  private val correctionEntries = correctionRegister.asTypeOf(
    Vec(
      p.correctionEntriesPerRow,
      SInt(p.accumulatorBits.W)
    )
  )
  private val finalCorrectionEntry =
    finalAccumulatorPixel.pad(p.correctionEntryIndexBits)(
      p.correctionEntryIndexBits - 1,
      0
    )
  io.postCorrection := Mux(
    finalStreamActive,
    correctionEntries(finalCorrectionEntry),
    correctionEntries(postCorrectionEntry)
  )
  io.postRes := Mux(
    finalStreamActive,
    VecInit(io.resReadResp.bits.map(_.asSInt.pad(p.accumulatorBits))),
    residualRegister
  )
  io.postInValid := Mux(
    finalStreamActive,
    finalResidualValid,
    postState === postIssueProcess
  )
  io.postParams := Mux(
    finalStreamActive,
    (if (p.maxOutputBlocks == 1) io.postParameters(0)
     else io.postParameters(block(log2Ceil(p.maxOutputBlocks) - 1, 0))),
    selectedPostParameters
  )

  when(
    preloadActive &&
      !preloadWaiting &&
      state =/= waitForWeightRead
  ) {
    io.wgtReadReq.valid := true.B
    io.wgtReadReq.bits := preloadWeightReadAddress
    when(io.wgtReadReq.fire) {
      preloadWaiting := true.B
      weightFetchCount := weightFetchCount + 1.U
    }
  }

  when(preloadWaiting && io.wgtReadResp.valid) {
    prefetchedWeights(preloadWeightLane) :=
      io.wgtReadResp.bits.asUInt
    preloadWaiting := false.B
    when(preloadWeightLane === (p.dim - 1).U) {
      preloadActive := false.B
      preloadValid := true.B
    }.otherwise {
      preloadWeightLane := preloadWeightLane + 1.U
      preloadWeightReadAddress :=
        preloadWeightReadAddress + weightLaneStride
    }
  }

  // The final tile consumes one packed correction row for every 16 pixels.
  // Fetch row zero before starting, then keep the following row in a second
  // register so Parameter Buffer latency never appears in the pixel stream.
  when(state === issueFinalCorrectionRead) {
    io.correctionReadReq.valid := true.B
    io.correctionReadReq.bits.address := commandRegister.correctionBase
  }.elsewhen(correctionPrefetchValid) {
    io.correctionReadReq.valid := true.B
    io.correctionReadReq.bits.address := corrPrefetchAddr
    when(io.correctionReadReq.fire) {
      correctionPrefetchValid := false.B
      corrPrefetchPending := true.B
    }
  }

  when(corrPrefetchPending && io.correctionReadResp.fire) {
    correctionNextRegister := io.correctionReadResp.bits
    correctionNextValid := true.B
    corrPrefetchPending := false.B
  }

  io.fullOutWrite.valid :=
    outputBufferValid && outputFullPending
  io.fullOutWrite.bits.address := outputFullAddress
  io.fullOutWrite.bits.data := outputFullData
  io.binOutWrite.valid :=
    outputBufferValid && outputBinaryPending
  io.binOutWrite.bits.address := outputBinaryAddress
  io.binOutWrite.bits.data := outputBinaryData
  accumulatorTaskQueue.io.enq.valid := false.B
  accumulatorTaskQueue.io.enq.bits :=
    0.U.asTypeOf(new BinaryAccumulatorTask(p))
  accumulatorTaskQueue.io.deq.ready := false.B
  completionQueue.io.enq.valid := false.B
  completionQueue.io.enq.bits :=
    0.U.asTypeOf(new BinaryCompletion(p))
  completionQueue.io.deq.ready := false.B

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

  private val finalBinaryReady =
    !commandRegister.writeBinaryOutput || io.binOutWrite.ready
  private val finalPostOutputFire =
    finalStreamActive && io.postOutValid && io.fullOutWrite.ready &&
      finalBinaryReady
  when(finalStreamActive) {
    io.fullOutWrite.valid := io.postOutValid
    io.fullOutWrite.bits.address := finalFullOutputAddress
    for (lane <- 0 until p.dim) {
      io.fullOutWrite.bits.data(lane) :=
        io.postOut(lane)(p.activationBits - 1, 0).asUInt
    }
    io.binOutWrite.valid :=
      io.postOutValid && commandRegister.writeBinaryOutput
    io.binOutWrite.bits.address := finalBinaryOutputAddress
    for (lane <- 0 until p.dim) {
      val finalThresholds =
        if (p.maxOutputBlocks == 1) io.outputSignThresholds(0)
        else io.outputSignThresholds(block(log2Ceil(p.maxOutputBlocks) - 1, 0))
      io.binOutWrite.bits.data(lane) :=
        (io.postOut(lane) >= finalThresholds(lane)).asUInt
    }
    when(io.postOutValid) {
      assert(finalPostTagValid.last, "final post tag must align with post output")
      assert(
        io.fullOutWrite.ready && finalBinaryReady,
        "final post writeback must sustain one vector per cycle"
      )
    }
    when(finalPostOutputFire) {
      finalFullOutputAddress :=
        finalFullOutputAddress + commandRegister.outputBlocks
      finalBinaryOutputAddress :=
        finalBinaryOutputAddress + commandRegister.outputBlocks
    }
  }

  // VecAccum returns the controller tag with each ordered response, allowing
  // independent rows to remain in flight without a task/tag FIFO pair.
  private val taskPortActive =
    state =/= issueAccInit &&
      state =/= waitAccInit
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
    io.accReq.bits.outputX := 0.U
    io.accReq.bits.outputY := 0.U
    accumulatorTaskQueue.io.deq.ready := io.accReq.ready

    // Register every completed output, including the final streaming tile.
    // This terminates the combinational Accumulator -> residual-SPAD ready
    // path.  The two-entry non-flow-through queue still sustains one result
    // per cycle in steady state.
    completionQueue.io.enq.valid :=
      io.accResp.valid && io.accResp.bits.completesOutput
    completionQueue.io.enq.bits.block := io.accResp.bits.block
    completionQueue.io.enq.bits.pixel := io.accResp.bits.pixel
    completionQueue.io.enq.bits.accumulator := io.accResp.bits.data
    io.accResp.ready := Mux(
      io.accResp.bits.completesOutput,
      completionQueue.io.enq.ready,
      true.B
    )
  }

  // Count a final result from the moment it enters the registered completion
  // buffer until its post-processed output is written.  Including buffered
  // results prevents drainArrayStream from completing while an item is queued.
  private val finalAccumulatorAccepted =
    taskPortActive && finalStreamActive && io.accResp.fire &&
      io.accResp.bits.completesOutput
  switch(Cat(finalAccumulatorAccepted, finalPostOutputFire)) {
    is("b10".U) {
      finalOutputOutstanding := finalOutputOutstanding + 1.U
    }
    is("b01".U) {
      finalOutputOutstanding := finalOutputOutstanding - 1.U
    }
  }
  // During the final tile, drain the registered completion stream directly
  // into the synchronous residual read port.  The queue absorbs any short
  // bank-arbitration delay without exposing it to VecAccum.
  when(finalStreamActive && completionQueue.io.deq.valid) {
    io.resReadReq.valid := true.B
    io.resReadReq.bits := finalResidualAddress
    completionQueue.io.deq.ready := io.resReadReq.ready
  }
  private val finalAccumulatorFire =
    finalStreamActive && completionQueue.io.deq.fire
  finalResidualValid := finalAccumulatorFire
  when(finalAccumulatorFire) {
    finalResidualAddress :=
      finalResidualAddress + commandRegister.outputBlocks
    finalAccumulator := completionQueue.io.deq.bits.accumulator
    finalAccumulatorPixel := completionQueue.io.deq.bits.pixel
    when(
      completionQueue.io.deq.bits.pixel.pad(p.correctionEntryIndexBits)(
        p.correctionEntryIndexBits - 1,
        0
      ) === 0.U &&
        completionQueue.io.deq.bits.pixel +
          p.correctionEntriesPerRow.U < outputPixelCount
    ) {
      corrPrefetchAddr := commandRegister.correctionBase +
        (completionQueue.io.deq.bits.pixel >>
          p.correctionEntryIndexBits) + 1.U
      correctionPrefetchValid := true.B
    }
  }
  when(finalResidualValid) {
    assert(io.resReadResp.valid, "final residual tag must align with SPAD response")
    when(
      finalCorrectionEntry === (p.correctionEntriesPerRow - 1).U &&
        finalAccumulatorPixel =/= outputPixelCount - 1.U
    ) {
      assert(
        correctionNextValid ||
          (corrPrefetchPending && io.correctionReadResp.fire),
        "next correction row must be prefetched before the pixel boundary"
      )
      correctionRegister := Mux(
        correctionNextValid,
        correctionNextRegister,
        io.correctionReadResp.bits
      )
      correctionNextValid := false.B
    }
  }

  finalPostTagValid(0) := finalResidualValid
  for (stage <- 1 until BinPostProc.latency) {
    finalPostTagValid(stage) := finalPostTagValid(stage - 1)
  }

  private val accumulatorTaskIssued =
    taskPortActive && accumulatorTaskQueue.io.deq.fire
  private val accumulatorTaskCompleted =
    taskPortActive && io.accResp.fire
  switch(Cat(accumulatorTaskIssued, accumulatorTaskCompleted)) {
    is("b10".U) { accumulatorOutstanding := accumulatorOutstanding + 1.U }
    is("b01".U) { accumulatorOutstanding := accumulatorOutstanding - 1.U }
  }

  when(streamTagStage3Valid) {
    assert(io.arrayOutValid, "binary stream tag must align with array output")
    accumulatorTaskQueue.io.enq.valid := true.B
    accumulatorTaskQueue.io.enq.bits.readOnly := streamPaddingStage3
    accumulatorTaskQueue.io.enq.bits.address :=
      commandRegister.accumulatorBase + streamTagStage3
    for (lane <- 0 until p.dim) {
      val shifted = io.arraySums(lane) << 1
      accumulatorTaskQueue.io.enq.bits.data(lane) :=
        shifted(p.accumulatorBits - 1, 0).asSInt
    }
    accumulatorTaskQueue.io.enq.bits.completesOutput := finalStreamActive
    accumulatorTaskQueue.io.enq.bits.block := block
    accumulatorTaskQueue.io.enq.bits.pixel := streamTagStage3
    assert(
      accumulatorTaskQueue.io.enq.ready,
      "non-final binary stream must sustain one accumulator task per cycle"
    )
  }

  private val streamArrayCompleted = streamTagStage3Valid
  switch(Cat(streamActivationIssued, streamArrayCompleted)) {
    is("b10".U) { streamOutstanding := streamOutstanding + 1.U }
    is("b01".U) { streamOutstanding := streamOutstanding - 1.U }
  }

  switch(postState) {
    is(postIdle) {
      when(completionQueue.io.deq.valid && !finalStreamActive) {
        completionQueue.io.deq.ready := true.B
        postAccumulator :=
          completionQueue.io.deq.bits.accumulator
        postBlock := completionQueue.io.deq.bits.block
        postPixel := completionQueue.io.deq.bits.pixel
        postResidualAddressRegister := nextPostResidualAddress
        postFullAddressRegister := nextPostFullAddress
        postBinaryAddressRegister := nextPostBinaryAddress
        nextPostResidualAddress :=
          nextPostResidualAddress + commandRegister.outputBlocks
        nextPostFullAddress :=
          nextPostFullAddress + commandRegister.outputBlocks
        nextPostBinaryAddress :=
          nextPostBinaryAddress + commandRegister.outputBlocks
        postState := postIssueResidualRead
      }
    }
    is(postIssueResidualRead) {
      io.resReadReq.valid := true.B
      when(io.resReadReq.fire) {
        postState := postWaitResidualRead
      }
    }
    is(postWaitResidualRead) {
      when(io.resReadResp.valid) {
        for (lane <- 0 until p.dim) {
          residualRegister(lane) :=
            io.resReadResp.bits(lane).asSInt.pad(
              p.accumulatorBits
            )
        }
        postState := Mux(
          postCorrectionEntry === 0.U,
          postIssueCorrectionRead,
          postIssueProcess
        )
      }
    }
    is(postIssueCorrectionRead) {
      io.correctionReadReq.valid := true.B
      when(io.correctionReadReq.fire) {
        postState := postWaitCorrectionRead
      }
    }
    is(postWaitCorrectionRead) {
      when(io.correctionReadResp.fire) {
        correctionRegister := io.correctionReadResp.bits
        postState := postIssueProcess
      }
    }
    is(postIssueProcess) {
      postState := postWaitProcess
    }
    is(postWaitProcess) {
      when(io.postOutValid) {
        completedPostOutput := io.postOut
        postState := postEnqueueOutput
      }
    }
    is(postEnqueueOutput) {
      when(!outputBufferValid) {
        outputBufferValid := true.B
        outputFullPending := true.B
        outputBinaryPending :=
          commandRegister.writeBinaryOutput
        outputFullAddress := postFullStateAddress
        outputBinaryAddress := postBinaryOutputAddress
        for (lane <- 0 until p.dim) {
          outputFullData(lane) :=
            completedPostOutput(lane)(
              p.activationBits - 1,
              0
            ).asUInt
          outputBinaryData(lane) :=
            (completedPostOutput(lane) >=
              selectedSignThresholds(lane)).asUInt
        }
        postState := postIdle
      }
    }
  }

  // Split command-derived scan products across the accumulator-clear window.
  // Each product now ends in a register instead of feeding another multiplier
  // or the shared SPAD arbitration cone in the same cycle.
  when(scanConfigPending) {
    val configuredInputBlocks = inputBlockCount.zext
    val configuredWeightStride =
      runtimeKernelElements * inputBlockCount
    initialActivationStart :=
      commandRegister.inputBase.zext +
        initialSpatialStart * configuredInputBlocks
    kernelActivationStart :=
      commandRegister.inputBase.zext +
        initialSpatialStart * configuredInputBlocks
    activationRowAdvance :=
      pixelSpatialRowAdvance * configuredInputBlocks
    activationTapRowAdvance :=
      tapSpatialRowAdvance * configuredInputBlocks
    weightLaneStride := configuredWeightStride
    blockWeightAdvance :=
      configuredWeightStride * (p.dim - 1).U + 1.U
    scanConfigPending := false.B
  }

  when(io.command.fire) {
    assert(io.command.bits.inputHeight > 0.U)
    assert(io.command.bits.inputHeight <= p.maxImageHeight.U)
    assert(io.command.bits.inputWidth > 0.U)
    assert(io.command.bits.inputWidth <= p.maxImageWidth.U)
    assert(io.command.bits.outputHeight > 0.U)
    assert(io.command.bits.outputHeight <= p.maxImageHeight.U)
    assert(io.command.bits.outputWidth > 0.U)
    assert(io.command.bits.outputWidth <= p.maxImageWidth.U)
    assert(io.command.bits.inputChannels >= p.dim.U)
    assert(io.command.bits.inputChannels <= p.maxInputChannels.U)
    assert(
      io.command.bits.inputChannels(p.laneIndexBits - 1, 0) === 0.U,
      "binary input channels must be a whole packed dim-wide block"
    )
    assert(io.command.bits.outputBlocks > 0.U)
    assert(io.command.bits.outputBlocks <= p.maxOutputBlocks.U)
    assert(io.command.bits.kernelHeight > 0.U)
    assert(io.command.bits.kernelHeight <= p.kernelSize.U)
    assert(io.command.bits.kernelWidth > 0.U)
    assert(io.command.bits.kernelWidth <= p.kernelSize.U)
    val correctionRows =
      (io.command.bits.outputHeight * io.command.bits.outputWidth +
        (p.correctionEntriesPerRow - 1).U) >>
        p.correctionEntryIndexBits
    assert(
      io.command.bits.correctionBase +& correctionRows <=
        p.parameterRows.U,
      "packed binary correction table exceeds the Parameter Buffer"
    )
    assert(
      (io.command.bits.residualBase >> p.bankRowBits) =/=
        (io.command.bits.fullOutputBase >> p.bankRowBits),
      "streaming residual input and full output must use different banks"
    )
    when(io.command.bits.writeBinaryOutput) {
      assert(
        (io.command.bits.inputBase >> p.bankRowBits) =/=
          (io.command.bits.binaryOutputBase >> p.bankRowBits),
        "streaming binary input and output must use different banks"
      )
    }

    val configuredInputBlocks =
      io.command.bits.inputChannels >> p.laneIndexBits
    val configuredPixels =
      io.command.bits.outputHeight * io.command.bits.outputWidth
    val configuredKernelElements =
      io.command.bits.kernelHeight * io.command.bits.kernelWidth
    val configuredInitialSpatial =
      -(io.command.bits.paddingY.zext * io.command.bits.inputWidth.zext +
        io.command.bits.paddingX.zext)
    commandRegister := io.command.bits
    inputBlockCount := configuredInputBlocks
    outputPixelCount := configuredPixels
    runtimeKernelElements := configuredKernelElements
    initialSpatialStart := configuredInitialSpatial
    pixelSpatialRowAdvance :=
      io.command.bits.inputWidth.zext -
        io.command.bits.outputWidth.zext + 1.S
    tapSpatialRowAdvance :=
      io.command.bits.inputWidth.zext -
        io.command.bits.kernelWidth.zext + 1.S
    scanConfigPending := true.B
    block := 0.U
    pixel := 0.U
    kernelTap := 0.U
    outputX := 0.U
    outputY := 0.U
    kernelX := 0.U
    kernelY := 0.U
    inputBlock := 0.U
    weightLane := 0.U
    preloadActive := false.B
    preloadWaiting := false.B
    preloadValid := false.B
    preloadBlock := 0.U
    preloadKernelTap := 0.U
    preloadInputBlock := 0.U
    preloadWeightLane := 0.U
    weightTileBase := io.command.bits.weightBase
    weightReadAddress := io.command.bits.weightBase
    preloadWeightTileBase := io.command.bits.weightBase
    preloadWeightReadAddress := io.command.bits.weightBase
    weightFetchCount := 0.U
    outputBufferValid := false.B
    outputFullPending := false.B
    outputBinaryPending := false.B
    postState := postIdle
    state := issueAccInit
    streamTagStage1Valid := false.B
    streamTagStage2Valid := false.B
    streamTagStage3Valid := false.B
    streamInputValid := false.B
    streamInputPadding := false.B
    streamOutstanding := 0.U
    finalStreamActive := false.B
    finalOutputOutstanding := 0.U
    finalResidualValid := false.B
    finalPostTagValid.foreach(_ := false.B)
    correctionNextValid := false.B
    correctionPrefetchValid := false.B
    corrPrefetchPending := false.B
    nextPostResidualAddress := io.command.bits.residualBase
    nextPostFullAddress := io.command.bits.fullOutputBase
    nextPostBinaryAddress := io.command.bits.binaryOutputBase
  }

  switch(state) {
    is(issueAccInit) {
      io.accReq.valid := true.B
      io.accReq.bits.operation :=
        AccumulatorOperation.write
      io.accReq.bits.address := accumulatorAddress
      io.accReq.bits.data := VecInit.fill(p.dim)(
        0.S(p.accumulatorBits.W)
      )
      when(io.accReq.fire) {
        state := waitAccInit
      }
    }

    is(waitAccInit) {
      when(io.accResp.fire) {
        when(isLastPixel) {
          pixel := 0.U
          state := Mux(
            block === 0.U,
            issueWeightRead,
            issueWeightRead
          )
        }.otherwise {
          pixel := pixel + 1.U
          state := issueAccInit
        }
      }
    }

    is(issueWeightRead) {
      when(preloadValid) {
        assert(preloadBlock === block)
        assert(preloadKernelTap === kernelTap)
        assert(preloadInputBlock === inputBlock)
        stationaryWeights := prefetchedWeights
        weightTileBase := preloadWeightTileBase
        weightReadAddress := preloadWeightTileBase
        preloadValid := false.B
        armNextTilePreload(preloadWeightTileBase)
        pixel := 0.U
        outputX := 0.U
        outputY := 0.U
        actAddrReg := kernelActivationStart.asUInt
        state := Mux(
          isLastKernelTap && isLastInputBlock,
          issueFinalCorrectionRead,
          streamPixels
        )
      }.elsewhen(!preloadActive && !preloadWaiting) {
        io.wgtReadReq.valid := true.B
        when(io.wgtReadReq.fire) {
          weightFetchCount := weightFetchCount + 1.U
          state := waitForWeightRead
        }
      }
    }

    is(waitForWeightRead) {
      when(io.wgtReadResp.valid) {
        stationaryWeights(weightLane) :=
          io.wgtReadResp.bits.asUInt
        when(isLastWeightLane) {
          pixel := 0.U
          outputX := 0.U
          outputY := 0.U
          actAddrReg := kernelActivationStart.asUInt
          armNextTilePreload(weightTileBase)
          state := Mux(
            isLastKernelTap && isLastInputBlock,
            issueFinalCorrectionRead,
            streamPixels
          )
        }.otherwise {
          weightLane := weightLane + 1.U
          weightReadAddress := weightReadAddress + weightLaneStride
          state := issueWeightRead
        }
      }
    }

    is(processPixel) {
      when(inputIsPadding) {
        when(isLastKernelTap && isLastInputBlock) {
          state := issueAccumulatorRead
        }.otherwise {
          advanceAfterPixel()
        }
      }.otherwise {
        state := issueActivationRead
      }
    }

    is(issueActivationRead) {
      io.actReadReq.valid := true.B
      when(io.actReadReq.fire) {
        state := waitForActivationRead
      }
    }

    is(waitForActivationRead) {
      when(io.actReadResp.valid) {
        activationRegister :=
          io.actReadResp.bits.asUInt
        state := issueArrayCompute
      }
    }

    is(issueArrayCompute) {
      state := waitForArrayResult
    }

    is(waitForArrayResult) {
      when(io.arrayOutValid) {
        arrayColumnSumRegister := io.arraySums
        state := issuePopcountAccumulate
      }
    }

    is(issuePopcountAccumulate) {
      accumulatorTaskQueue.io.enq.valid := true.B
      accumulatorTaskQueue.io.enq.bits.readOnly := false.B
      accumulatorTaskQueue.io.enq.bits.address :=
        accumulatorAddress
      for (lane <- 0 until p.dim) {
        val shifted = arrayColumnSumRegister(lane) << 1
        accumulatorTaskQueue.io.enq.bits.data(lane) :=
          shifted(p.accumulatorBits - 1, 0).asSInt
      }
      accumulatorTaskQueue.io.enq.bits.completesOutput :=
        isLastKernelTap && isLastInputBlock
      accumulatorTaskQueue.io.enq.bits.block := block
      accumulatorTaskQueue.io.enq.bits.pixel := pixel
      when(accumulatorTaskQueue.io.enq.fire) {
        advanceAfterAcc()
      }
    }

    is(issueFinalCorrectionRead) {
      when(io.correctionReadReq.fire) {
        state := waitFinalCorrectionRead
      }
    }

    is(waitFinalCorrectionRead) {
      when(io.correctionReadResp.fire) {
        correctionRegister := io.correctionReadResp.bits
        correctionNextValid := false.B
        finalStreamActive := true.B
        finalResidualAddress := commandRegister.residualBase + block
        finalFullOutputAddress := commandRegister.fullOutputBase + block
        finalBinaryOutputAddress := commandRegister.binaryOutputBase + block
        pixel := 0.U
        outputX := 0.U
        outputY := 0.U
        state := streamPixels
      }
    }

    is(streamPixels) {
      io.actReadReq.valid := !inputIsPadding
      when(inputIsPadding || io.actReadReq.fire) {
        streamInputPixel := pixel
        streamInputPadding := inputIsPadding
        when(isLastPixel) {
          pixel := 0.U
          outputX := 0.U
          outputY := 0.U
          state := drainArrayStream
        }.otherwise {
          pixel := pixel + 1.U
          actAddrReg :=
            Mux(
              outputX === commandRegister.outputWidth - 1.U,
              (actAddrReg.zext + activationRowAdvance).asUInt,
              actAddrReg + inputBlockCount
            )(p.binaryAddressBits - 1, 0)
          when(outputX === commandRegister.outputWidth - 1.U) {
            outputX := 0.U
            outputY := outputY + 1.U
          }.otherwise {
            outputX := outputX + 1.U
          }
        }
      }
    }

    is(drainArrayStream) {
      when(
        streamOutstanding === 0.U &&
          !streamInputValid &&
          !streamTagStage1Valid &&
          !streamTagStage2Valid &&
          !streamTagStage3Valid &&
          (!finalStreamActive ||
            (finalOutputOutstanding === 0.U &&
              accumulatorOutstanding === 0.U &&
              !accumulatorTaskQueue.io.deq.valid &&
              !finalResidualValid &&
              !finalPostTagValid.asUInt.orR &&
              !io.postOutValid))
      ) {
        pixel := 0.U
        when(finalStreamActive) {
          finalStreamActive := false.B
          correctionPrefetchValid := false.B
          corrPrefetchPending := false.B
          state := waitForOutputDrain
        }.otherwise {
          when(isLastInputBlock) {
            inputBlock := 0.U
            kernelTap := kernelTap + 1.U
            when(kernelX === commandRegister.kernelWidth - 1.U) {
              kernelX := 0.U
              kernelY := kernelY + 1.U
              kernelActivationStart :=
                kernelActivationStart + activationTapRowAdvance
            }.otherwise {
              kernelX := kernelX + 1.U
              kernelActivationStart :=
                kernelActivationStart + inputBlockCount.zext
            }
          }.otherwise {
            inputBlock := inputBlock + 1.U
          }
          state := issueWeightRead
        }
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
}
