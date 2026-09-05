package bira

// Complete Rocket-independent data plane.

import chisel3._
import chisel3.util._

/** Converts one scheduled EXEC task and its Context into the existing core
  * command, streaming every output block's two-lane parameter rows first.
  */
class ExecBridge(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new Context(p)))
    val task = Flipped(Decoupled(new ExecTask(p)))
    val parameterRequest =
      Decoupled(new ParamRead(p))
    val parameterResponse =
      Flipped(Decoupled(new ParamRowResponse(p)))
    val multiParameter =
      Decoupled(new ConvParamPairWrite(p))
    val binaryParameter =
      Decoupled(new BinParamPairWrite(p))
    val multiCommand =
      Decoupled(new ConvolutionCommand(p))
    val binaryCommand =
      Decoupled(new BinaryConvolutionCommand(p))
    val coreStatus = Input(new AcceleratorStatus)
    val completion = Decoupled(new Completion(p))
  })

  private val Seq(
    idle,
    requestParameter,
    streamParameter,
    issueCommand,
    run,
    complete
  ) = Enum(6)
  private val state = RegInit(idle)
  private val taskReg = Reg(new ExecTask(p))
  private val contextReg = Reg(new Context(p))
  private val block = RegInit(0.U(p.blockIndexBits.W))
  private val blocks = Reg(UInt(p.blockIndexBits.W))
  private val completionError = RegInit(ErrorCode.none.U(8.W))

  private val selectedContext = MuxLookup(
    io.task.bits.contextId,
    0.U.asTypeOf(new Context(p))
  )(
    (0 until p.nContexts).map(index =>
      index.U -> io.contexts(index)
    )
  )
  private val selectedBlocks =
    ((selectedContext.outputChannels + (p.dim - 1).U) >>
      p.laneIndexBits)(p.blockIndexBits - 1, 0)
  private val taskError = MuxCase(
    ErrorCode.none.U,
    Seq(
      (io.task.bits.contextId >= p.nContexts.U) ->
        ErrorCode.invalidContext.U,
      (!selectedContext.ready || !selectedContext.committed) ->
        ErrorCode.contextNotReady.U,
      (selectedContext.errorCode =/= ErrorCode.none.U) ->
        ErrorCode.contextFailed.U,
      (selectedBlocks === 0.U ||
        selectedBlocks > p.maxOutputBlocks.U) ->
        ErrorCode.badShape.U
    )
  )

  io.task.ready := state === idle
  when(io.task.fire) {
    taskReg := io.task.bits
    contextReg := selectedContext
    blocks := selectedBlocks
    block := 0.U
    completionError := taskError
    state := Mux(
      taskError =/= ErrorCode.none.U,
      complete,
      requestParameter
    )
  }

  io.parameterRequest.valid := state === requestParameter
  io.parameterRequest.bits.baseRow :=
    contextReg.baseRows(AddrRole.parameter)(
      p.parameterAddressBits - 1,
      0
    )
  io.parameterRequest.bits.block := block
  when(io.parameterRequest.fire) {
    state := streamParameter
  }

  private val binaryMode =
    contextReg.arrayMode === ArrayMode.binary.U
  // Decode only the two lanes carried by the registered Parameter Buffer
  // response. No complete-block or dual-interpretation register remains.
  private val parameterRow = io.parameterResponse.bits
  private val records = VecInit(
    parameterRow.data(255, 0),
    parameterRow.data(511, 256)
  )

  io.multiParameter.bits :=
    0.U.asTypeOf(new ConvParamPairWrite(p))
  io.multiParameter.bits.block := parameterRow.block
  io.multiParameter.bits.lanePair := parameterRow.lanePair
  io.binaryParameter.bits :=
    0.U.asTypeOf(new BinParamPairWrite(p))
  io.binaryParameter.bits.block := parameterRow.block
  io.binaryParameter.bits.lanePair := parameterRow.lanePair
  for (lane <- 0 until 2) {
    val record = records(lane)

    io.multiParameter.bits.bias(lane) := record(31, 0).asSInt
    io.multiParameter.bits.post(lane).positiveShift :=
      record(39, 32).asSInt
    io.multiParameter.bits.post(lane).negativeCoeff1 :=
      record(41, 40).asSInt
    io.multiParameter.bits.post(lane).negativeCoeff2 :=
      record(43, 42).asSInt
    io.multiParameter.bits.post(lane).negativeLeftShift1 :=
      record(48, 44)
    io.multiParameter.bits.post(lane).negativeLeftShift2 :=
      record(53, 49)
    io.multiParameter.bits.post(lane).negativeCommonShift :=
      record(61, 54).asSInt
    io.multiParameter.bits.post(lane).qMin :=
      record(93, 62).asSInt
    io.multiParameter.bits.post(lane).qMax :=
      record(125, 94).asSInt
    io.multiParameter.bits.binaryThreshold(lane) :=
      record(157, 126).asSInt

    io.binaryParameter.bits.post(lane).threshold :=
      record(31, 0).asSInt
    io.binaryParameter.bits.post(lane).positiveCoeff2 :=
      record(33, 32).asSInt
    io.binaryParameter.bits.post(lane).positiveLeftShift1 :=
      record(38, 34)
    io.binaryParameter.bits.post(lane).positiveLeftShift2 :=
      record(43, 39)
    io.binaryParameter.bits.post(lane).positiveCommonShift :=
      record(51, 44).asSInt
    io.binaryParameter.bits.post(lane).positiveBias :=
      record(83, 52).asSInt
    io.binaryParameter.bits.post(lane).negativeCoeff1 :=
      record(85, 84).asSInt
    io.binaryParameter.bits.post(lane).negativeCoeff2 :=
      record(87, 86).asSInt
    io.binaryParameter.bits.post(lane).negativeLeftShift1 :=
      record(92, 88)
    io.binaryParameter.bits.post(lane).negativeLeftShift2 :=
      record(97, 93)
    io.binaryParameter.bits.post(lane).negativeCommonShift :=
      record(105, 98).asSInt
    io.binaryParameter.bits.post(lane).negativeBias :=
      record(137, 106).asSInt
    io.binaryParameter.bits.post(lane).qMin :=
      record(169, 138).asSInt
    io.binaryParameter.bits.post(lane).qMax :=
      record(201, 170).asSInt
    io.binaryParameter.bits.outputSignThreshold(lane) :=
      record(233, 202).asSInt
  }

  io.multiParameter.valid :=
    state === streamParameter &&
      io.parameterResponse.valid && !binaryMode
  io.binaryParameter.valid :=
    state === streamParameter &&
      io.parameterResponse.valid && binaryMode
  io.parameterResponse.ready :=
    state === streamParameter &&
      Mux(binaryMode, io.binaryParameter.ready, io.multiParameter.ready)
  when(io.parameterResponse.fire) {
    when(parameterRow.last) {
      when(block === blocks - 1.U) {
        state := issueCommand
      }.otherwise {
        block := block + 1.U
        state := requestParameter
      }
    }
  }

  private val multi = WireDefault(
    0.U.asTypeOf(new ConvolutionCommand(p))
  )
  multi.inputBase :=
    contextReg.baseRows(AddrRole.input)
  multi.weightLowBase :=
    contextReg.baseRows(AddrRole.weightLow)
  multi.weightHighBase :=
    contextReg.baseRows(AddrRole.weightHigh)
  multi.outputBase :=
    contextReg.baseRows(AddrRole.outputFull)
  multi.binaryOutputBase :=
    contextReg.baseRows(AddrRole.outputBinary)
  multi.accumulatorBase :=
    contextReg.baseRows(AddrRole.accumulator)
  multi.residualBase :=
    contextReg.baseRows(AddrRole.residual)
  multi.inputHeight := contextReg.inputHeight
  multi.inputWidth := contextReg.inputWidth
  multi.outputHeight := contextReg.outputHeight
  multi.outputWidth := contextReg.outputWidth
  multi.inputChannels := contextReg.inputChannels
  multi.outputBlocks := blocks
  multi.kernelHeight := contextReg.kernelHeight
  multi.kernelWidth := contextReg.kernelWidth
  multi.paddingY := contextReg.paddingHeight
  multi.paddingX := contextReg.paddingWidth
  multi.weightPrecision := MuxLookup(
    contextReg.weightPrecision,
    16.U
  )(
    Seq(
      WgtPrecision.w2.U -> 2.U,
      WgtPrecision.w4.U -> 4.U,
      WgtPrecision.w8.U -> 8.U,
      WgtPrecision.w16.U -> 16.U
    )
  )
  multi.activationPrecision := p.activationBits.U
  multi.inputSigned := contextReg.inputSigned
  multi.depthwise :=
    contextReg.arrayMode === ArrayMode.depthwise.U
  multi.columnReduce :=
    contextReg.arrayMode === ArrayMode.columnReduce.U
  multi.bilinearResidual :=
    contextReg.postMode === PostMode.finalBilinearResidual.U
  multi.shufflePack2 := contextReg.shufflePack2
  multi.shuffleLog2 := log2Ceil(p.maxShuffleScale).U
  multi.writeFullOutput := contextReg.writeFull
  multi.writeBinaryOutput := contextReg.writeBinary

  private val binary = WireDefault(
    0.U.asTypeOf(new BinaryConvolutionCommand(p))
  )
  binary.inputBase :=
    contextReg.baseRows(AddrRole.input)
  binary.weightBase :=
    contextReg.baseRows(AddrRole.weightLow)
  binary.binaryOutputBase :=
    contextReg.baseRows(AddrRole.outputBinary)
  binary.residualBase :=
    contextReg.baseRows(AddrRole.residual)
  binary.fullOutputBase :=
    contextReg.baseRows(AddrRole.outputFull)
  binary.correctionBase :=
    contextReg.baseRows(AddrRole.correction)
  binary.accumulatorBase :=
    contextReg.baseRows(AddrRole.accumulator)
  binary.inputHeight := contextReg.inputHeight
  binary.inputWidth := contextReg.inputWidth
  binary.outputHeight := contextReg.outputHeight
  binary.outputWidth := contextReg.outputWidth
  binary.inputChannels := contextReg.inputChannels
  binary.outputBlocks := blocks
  binary.kernelHeight := contextReg.kernelHeight
  binary.kernelWidth := contextReg.kernelWidth
  binary.paddingY := contextReg.paddingHeight
  binary.paddingX := contextReg.paddingWidth
  binary.writeBinaryOutput := contextReg.writeBinary

  io.multiCommand.valid :=
    state === issueCommand && !binaryMode
  io.multiCommand.bits := multi
  io.binaryCommand.valid :=
    state === issueCommand && binaryMode
  io.binaryCommand.bits := binary
  when(io.multiCommand.fire || io.binaryCommand.fire) {
    state := run
  }
  when(state === run && io.coreStatus.done) {
    completionError := ErrorCode.none.U
    state := complete
  }

  io.completion.valid := state === complete
  io.completion.bits.contextId := taskReg.contextId
  io.completion.bits.commandSequence := taskReg.commandSequence
  io.completion.bits.errorCode := completionError
  when(io.completion.fire) {
    state := idle
  }
}

/** Complete Rocket-independent compute/storage data plane. */
class DataPlane(p: AccelParams = AccelParams()) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new Context(p)))
    val execTask = Flipped(Decoupled(new ExecTask(p)))
    val execCompletion = Decoupled(new Completion(p))
    val localWrite = Flipped(Decoupled(new LocalWrite))
    val localReadRequest =
      Flipped(Decoupled(new LocalReadReq))
    val localReadResponse = Valid(new LocalReadResp)
    val busy = Output(Bool())
  })

  private val core = Module(new Core(p))
  private val parameters = Module(new ParamBuffer(p))
  private val exec = Module(new ExecBridge(p))

  exec.io.contexts := io.contexts
  exec.io.task <> io.execTask
  exec.io.parameterRequest <> parameters.io.readRequest
  exec.io.parameterResponse <> parameters.io.readResponse
  core.io.correctionReadRequest <>
    parameters.io.correctionReadRequest
  core.io.correctionReadResponse <>
    parameters.io.correctionReadResponse
  exec.io.multiParameter <> core.io.parameterWrite
  exec.io.binaryParameter <> core.io.binaryParameterWrite
  exec.io.multiCommand <> core.io.command
  exec.io.binaryCommand <> core.io.binaryCommand
  exec.io.coreStatus := core.io.status
  io.execCompletion <> exec.io.completion
  io.busy := core.io.status.busy || io.execTask.valid

  core.io.fullWrite.valid := false.B
  core.io.fullWrite.bits :=
    0.U.asTypeOf(core.io.fullWrite.bits)
  core.io.binaryWrite.valid := false.B
  core.io.binaryWrite.bits :=
    0.U.asTypeOf(core.io.binaryWrite.bits)
  core.io.accumulatorWrite.valid := false.B
  core.io.accumulatorWrite.bits :=
    0.U.asTypeOf(core.io.accumulatorWrite.bits)
  parameters.io.write.valid := false.B
  parameters.io.write.bits :=
    0.U.asTypeOf(parameters.io.write.bits)

  private val writeMemory = io.localWrite.bits.memory
  private val writeFull =
    writeMemory === LocalMem.full.U
  private val writeBinary =
    writeMemory === LocalMem.binary.U
  private val writeAccumulator =
    writeMemory === LocalMem.accumulator.U
  private val writeParameter =
    writeMemory === LocalMem.parameter.U

  core.io.fullWrite.valid := io.localWrite.valid && writeFull
  core.io.fullWrite.bits.address :=
    io.localWrite.bits.address(p.fullAddressBits - 1, 0)
  for (lane <- 0 until p.dim) {
    core.io.fullWrite.bits.data(lane) :=
      io.localWrite.bits.data(
        (lane + 1) * p.activationBits - 1,
        lane * p.activationBits
      )
  }

  core.io.binaryWrite.valid :=
    io.localWrite.valid && writeBinary
  core.io.binaryWrite.bits.address :=
    io.localWrite.bits.address(p.binaryAddressBits - 1, 0)
  for (lane <- 0 until p.dim) {
    core.io.binaryWrite.bits.data(lane) :=
      io.localWrite.bits.data(lane)
  }

  core.io.accumulatorWrite.valid :=
    io.localWrite.valid && writeAccumulator
  core.io.accumulatorWrite.bits.address :=
    io.localWrite.bits.address(p.accumulatorAddressBits - 1, 0)
  for (lane <- 0 until p.dim) {
    core.io.accumulatorWrite.bits.data(lane) :=
      io.localWrite.bits.data(
        (lane + 1) * p.accumulatorBits - 1,
        lane * p.accumulatorBits
      ).asSInt
  }

  parameters.io.write.valid :=
    io.localWrite.valid && writeParameter
  parameters.io.write.bits.address :=
    io.localWrite.bits.address(p.parameterAddressBits - 1, 0)
  parameters.io.write.bits.data := io.localWrite.bits.data

  io.localWrite.ready := MuxCase(
    false.B,
    Seq(
      writeFull -> core.io.fullWrite.ready,
      writeBinary -> core.io.binaryWrite.ready,
      writeAccumulator -> core.io.accumulatorWrite.ready,
      writeParameter -> parameters.io.write.ready
    )
  )

  private val Seq(
    readIdle,
    waitFull,
    waitBinary,
    waitAccumulator,
    readError
  ) = Enum(5)
  private val readState = RegInit(readIdle)

  // StoreCtrl has at most one local read in flight.  Capture that request at
  // the data-plane boundary so its state transition depends only on registered
  // occupancy, rather than on a combinational path through the selected SPAD
  // bank arbiter.  The queued request remains stable until the memory accepts
  // it; the eventual response still uses readState to select the return data.
  private val localReadBuffer = Module(
    new Queue(
      new LocalReadReq,
      entries = 1,
      pipe = false,
      flow = false
    )
  )
  localReadBuffer.io.enq <> io.localReadRequest

  private val bufferedRead = localReadBuffer.io.deq
  core.io.fullReadRequest.valid :=
    readState === readIdle &&
      bufferedRead.valid &&
      bufferedRead.bits.memory === LocalMem.full.U
  core.io.fullReadRequest.bits :=
    bufferedRead.bits.address(p.fullAddressBits - 1, 0)
  core.io.binaryReadRequest.valid :=
    readState === readIdle &&
      bufferedRead.valid &&
      bufferedRead.bits.memory === LocalMem.binary.U
  core.io.binaryReadRequest.bits :=
    bufferedRead.bits.address(p.binaryAddressBits - 1, 0)
  core.io.accumulatorReadRequest.valid :=
    readState === readIdle &&
      bufferedRead.valid &&
      bufferedRead.bits.memory === LocalMem.accumulator.U
  core.io.accumulatorReadRequest.bits :=
    bufferedRead.bits.address(
      p.accumulatorAddressBits - 1,
      0
    )

  bufferedRead.ready :=
    readState === readIdle &&
      MuxLookup(
        bufferedRead.bits.memory,
        true.B
      )(
        Seq(
          LocalMem.full.U -> core.io.fullReadRequest.ready,
          LocalMem.binary.U ->
            core.io.binaryReadRequest.ready,
          LocalMem.accumulator.U ->
            core.io.accumulatorReadRequest.ready
        )
      )
  when(bufferedRead.fire) {
    readState := MuxLookup(
      bufferedRead.bits.memory,
      readError
    )(
      Seq(
        LocalMem.full.U -> waitFull,
        LocalMem.binary.U -> waitBinary,
        LocalMem.accumulator.U -> waitAccumulator
      )
    )
  }

  io.localReadResponse.valid :=
    (readState === waitFull && core.io.fullReadResponse.valid) ||
      (readState === waitBinary &&
        core.io.binaryReadResponse.valid) ||
      (readState === waitAccumulator &&
        core.io.accumulatorReadResponse.valid) ||
      readState === readError
  io.localReadResponse.bits.data := MuxCase(
    0.U,
    Seq(
      (readState === waitFull) ->
        core.io.fullReadResponse.bits.asUInt.pad(512),
      (readState === waitBinary) ->
        core.io.binaryReadResponse.bits.asUInt.pad(512),
      (readState === waitAccumulator) ->
        core.io.accumulatorReadResponse.bits.asUInt
    )
  )
  io.localReadResponse.bits.errorCode :=
    Mux(
      readState === readError,
      ErrorCode.badRole.U,
      ErrorCode.none.U
    )
  when(io.localReadResponse.valid) {
    readState := readIdle
  }
}
