package bira

// Binary convolution controller.

import chisel3._
import chisel3.util._

class BinaryAccumulatorTask(p: BiRaParams) extends Bundle {
  val readOnly = Bool()
  val address = UInt(p.accumulatorAddressBits.W)
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
  val completesOutput = Bool()
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
}

class BinaryCompletion(p: BiRaParams) extends Bundle {
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
class BinConvCtrl(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val command =
      Flipped(Decoupled(new BinaryConvolutionCommand(p)))
    val status = Output(new AcceleratorStatus)

    val postParameters = Input(
      Vec(
        p.maxOutputBlocks,
        Vec(p.dim, new BinaryPostProcessParameters(p))
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
      Decoupled(new BiRaCorrectionReadRequest(p))
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
      Output(Vec(p.dim, new BinaryPostProcessParameters(p)))
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
    issueAccumulatorInitialize,
    waitForAccumulatorInitialize,
    issueWeightRead,
    waitForWeightRead,
    processPixel,
    issueActivationRead,
    waitForActivationRead,
    issueArrayCompute,
    waitForArrayResult,
    issuePopcountAccumulate,
    issueAccumulatorRead,
    waitForOutputDrain
  ) = Enum(13)

  private val state = RegInit(idle)
  private val accIdle :: accIssue :: accWait :: Nil = Enum(3)
  private val accState = RegInit(accIdle)
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
  private val activationRegister = Reg(UInt(p.dim.W))
  private val correctionRegister =
    Reg(UInt(512.W))
  private val postAccumulator =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val postBlock = RegInit(0.U(p.blockIndexBits.W))
  private val postPixel = RegInit(0.U(p.pixelIndexBits.W))
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
  private val accumulatorTaskQueue =
    Module(new Queue(new BinaryAccumulatorTask(p), entries = 8))
  private val activeAccumulatorTask =
    Reg(new BinaryAccumulatorTask(p))
  private val completionQueue =
    Module(new Queue(new BinaryCompletion(p), entries = 4))

  private val runtimeKernelElements =
    commandRegister.kernelHeight * commandRegister.kernelWidth
  private val outputPixelCount =
    commandRegister.outputHeight * commandRegister.outputWidth
  private val inputBlockCount =
    commandRegister.inputChannels >> p.laneIndexBits

  private val isLastPixel = pixel === outputPixelCount - 1.U
  private val isLastKernelTap =
    kernelTap === runtimeKernelElements - 1.U
  private val isLastInputBlock =
    inputBlock === inputBlockCount - 1.U
  private val isLastWeightLane = weightLane === (p.dim - 1).U

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
  private val inputSpatialIndex =
    signedInputY.asUInt * commandRegister.inputWidth +
      signedInputX.asUInt

  private val accumulatorAddress =
    commandRegister.accumulatorBase + pixel
  private val activationAddress =
    commandRegister.inputBase +
      inputSpatialIndex * inputBlockCount + inputBlock
  private val outputChannel = block * p.dim.U + weightLane
  private val weightAddress =
    commandRegister.weightBase +
      (outputChannel * runtimeKernelElements + kernelTap) *
        inputBlockCount +
      inputBlock
  private val preloadOutputChannel =
    preloadBlock * p.dim.U + preloadWeightLane
  private val preloadWeightAddress =
    commandRegister.weightBase +
      (preloadOutputChannel * runtimeKernelElements +
        preloadKernelTap) * inputBlockCount +
      preloadInputBlock
  private val postCorrectionAddress =
    commandRegister.correctionBase +
      (postPixel >> p.correctionEntryIndexBits)
  private val postCorrectionEntry =
    postPixel.pad(p.correctionEntryIndexBits)(
      p.correctionEntryIndexBits - 1,
      0
    )
  private val postResidualAddress =
    commandRegister.residualBase +
      postPixel * commandRegister.outputBlocks + postBlock
  private val postFullStateAddress =
    commandRegister.fullOutputBase +
      postPixel * commandRegister.outputBlocks + postBlock
  private val postBinaryOutputAddress =
    commandRegister.binaryOutputBase +
      postPixel * commandRegister.outputBlocks + postBlock

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

  private def armNextTilePreload(): Unit = {
    preloadWeightLane := 0.U
    preloadWaiting := false.B
    preloadValid := false.B
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
      when(isLastInputBlock) {
        inputBlock := 0.U
        when(isLastKernelTap) {
          kernelTap := 0.U
          state := issueAccumulatorRead
        }.otherwise {
          kernelTap := kernelTap + 1.U
          state := issueWeightRead
        }
      }.otherwise {
        inputBlock := inputBlock + 1.U
        state := issueWeightRead
      }
    }.otherwise {
      pixel := pixel + 1.U
      state := processPixel
    }
  }

  private def advanceAfterQueuedAccumulator(): Unit = {
    when(isLastPixel && isLastKernelTap && isLastInputBlock) {
      pixel := 0.U
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
      inputBlock := 0.U
      weightLane := 0.U
      state := issueAccumulatorInitialize
    }
  }

  donePulse := false.B
  io.command.ready :=
    state === idle &&
      accState === accIdle &&
      !accumulatorTaskQueue.io.deq.valid &&
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

  io.actReadReq.valid := false.B
  io.actReadReq.bits := activationAddress
  io.wgtReadReq.valid := false.B
  io.wgtReadReq.bits := weightAddress
  io.resReadReq.valid := false.B
  io.resReadReq.bits := postResidualAddress
  io.correctionReadReq.valid := false.B
  io.correctionReadReq.bits.address := postCorrectionAddress
  io.correctionReadResp.ready :=
    postState === postWaitCorrectionRead

  io.accReq.valid := false.B
  io.accReq.bits :=
    0.U.asTypeOf(new AccumulatorRequest(p))
  io.accResp.ready :=
    state === waitForAccumulatorInitialize

  io.arrayAct := activationRegister
  io.arrayWgts := stationaryWeights
  io.arrayInValid := state === issueArrayCompute

  io.postAcc := postAccumulator
  private val correctionEntries = correctionRegister.asTypeOf(
    Vec(
      p.correctionEntriesPerRow,
      SInt(p.accumulatorBits.W)
    )
  )
  io.postCorrection := correctionEntries(postCorrectionEntry)
  io.postRes := residualRegister
  io.postInValid := postState === postIssueProcess
  io.postParams := selectedPostParameters

  when(
    preloadActive &&
      !preloadWaiting &&
      state =/= waitForWeightRead
  ) {
    io.wgtReadReq.valid := true.B
    io.wgtReadReq.bits := preloadWeightAddress
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
    }
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
      io.accReq.bits.address := activeAccumulatorTask.address
      io.accReq.bits.data := activeAccumulatorTask.data
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

  switch(postState) {
    is(postIdle) {
      when(completionQueue.io.deq.valid) {
        completionQueue.io.deq.ready := true.B
        postAccumulator :=
          completionQueue.io.deq.bits.accumulator
        postBlock := completionQueue.io.deq.bits.block
        postPixel := completionQueue.io.deq.bits.pixel
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

    commandRegister := io.command.bits
    block := 0.U
    pixel := 0.U
    kernelTap := 0.U
    inputBlock := 0.U
    weightLane := 0.U
    preloadActive := false.B
    preloadWaiting := false.B
    preloadValid := false.B
    preloadBlock := 0.U
    preloadKernelTap := 0.U
    preloadInputBlock := 0.U
    preloadWeightLane := 0.U
    weightFetchCount := 0.U
    outputBufferValid := false.B
    outputFullPending := false.B
    outputBinaryPending := false.B
    accState := accIdle
    postState := postIdle
    state := issueAccumulatorInitialize
  }

  switch(state) {
    is(issueAccumulatorInitialize) {
      io.accReq.valid := true.B
      io.accReq.bits.operation :=
        AccumulatorOperation.write
      io.accReq.bits.address := accumulatorAddress
      io.accReq.bits.data := VecInit.fill(p.dim)(
        0.S(p.accumulatorBits.W)
      )
      when(io.accReq.fire) {
        state := waitForAccumulatorInitialize
      }
    }

    is(waitForAccumulatorInitialize) {
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
          state := issueAccumulatorInitialize
        }
      }
    }

    is(issueWeightRead) {
      when(preloadValid) {
        assert(preloadBlock === block)
        assert(preloadKernelTap === kernelTap)
        assert(preloadInputBlock === inputBlock)
        stationaryWeights := prefetchedWeights
        preloadValid := false.B
        armNextTilePreload()
        pixel := 0.U
        state := processPixel
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
          armNextTilePreload()
          state := processPixel
        }.otherwise {
          weightLane := weightLane + 1.U
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
