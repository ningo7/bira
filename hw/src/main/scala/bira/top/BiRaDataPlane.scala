package bira

// Complete Rocket-independent data plane.

import chisel3._
import chisel3.util._

/** Converts one scheduled EXEC task and its Context into the existing core
  * command, loading every output block's decoded parameters first.
  */
class BiRaExecBridge(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new BiRaContext(p)))
    val task = Flipped(Decoupled(new BiRaExecTask(p)))
    val parameterRequest =
      Decoupled(new BiRaParameterBlockRead(p))
    val parameterResponse =
      Flipped(Decoupled(new BiRaDecodedParameterBlock(p)))
    val multiParameter =
      Decoupled(new ConvolutionParameterWrite(p))
    val binaryParameter =
      Decoupled(new BinaryConvolutionParameterWrite(p))
    val multiCommand =
      Decoupled(new ConvolutionCommand(p))
    val binaryCommand =
      Decoupled(new BinaryConvolutionCommand(p))
    val coreStatus = Input(new AcceleratorStatus)
    val completion = Decoupled(new BiRaTaskCompletion(p))
  })

  private val Seq(
    idle,
    requestParameter,
    waitParameter,
    writeParameter,
    issueCommand,
    run,
    complete
  ) = Enum(7)
  private val state = RegInit(idle)
  private val taskReg = Reg(new BiRaExecTask(p))
  private val contextReg = Reg(new BiRaContext(p))
  private val block = RegInit(0.U(p.blockIndexBits.W))
  private val blocks = Reg(UInt(p.blockIndexBits.W))
  private val decoded =
    Reg(new BiRaDecodedParameterBlock(p))
  private val completionError = RegInit(BiRaError.none.U(8.W))

  private val selectedContext = MuxLookup(
    io.task.bits.contextId,
    0.U.asTypeOf(new BiRaContext(p))
  )(
    (0 until p.nContexts).map(index =>
      index.U -> io.contexts(index)
    )
  )
  private val selectedBlocks =
    ((selectedContext.outputChannels + (p.dim - 1).U) >>
      p.laneIndexBits)(p.blockIndexBits - 1, 0)
  private val taskError = MuxCase(
    BiRaError.none.U,
    Seq(
      (io.task.bits.contextId >= p.nContexts.U) ->
        BiRaError.invalidContext.U,
      (!selectedContext.ready || !selectedContext.committed) ->
        BiRaError.contextNotReady.U,
      (selectedContext.errorCode =/= BiRaError.none.U) ->
        BiRaError.contextFailed.U,
      (selectedBlocks === 0.U ||
        selectedBlocks > p.maxOutputBlocks.U) ->
        BiRaError.badShape.U
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
      taskError =/= BiRaError.none.U,
      complete,
      requestParameter
    )
  }

  io.parameterRequest.valid := state === requestParameter
  io.parameterRequest.bits.baseRow :=
    contextReg.baseRows(BiRaAddrRole.parameter)(
      p.parameterAddressBits - 1,
      0
    )
  io.parameterRequest.bits.block := block
  when(io.parameterRequest.fire) {
    state := waitParameter
  }

  io.parameterResponse.ready := state === waitParameter
  when(io.parameterResponse.fire) {
    decoded := io.parameterResponse.bits
    state := writeParameter
  }

  private val binaryMode =
    contextReg.arrayMode === BiRaArrayMode.binary.U
  io.multiParameter.valid :=
    state === writeParameter && !binaryMode
  io.multiParameter.bits := decoded.multiBit
  io.binaryParameter.valid :=
    state === writeParameter && binaryMode
  io.binaryParameter.bits := decoded.binary
  private val parameterWritten =
    io.multiParameter.fire || io.binaryParameter.fire
  when(parameterWritten) {
    when(block === blocks - 1.U) {
      state := issueCommand
    }.otherwise {
      block := block + 1.U
      state := requestParameter
    }
  }

  private val multi = WireDefault(
    0.U.asTypeOf(new ConvolutionCommand(p))
  )
  multi.inputBase :=
    contextReg.baseRows(BiRaAddrRole.input)
  multi.weightLowBase :=
    contextReg.baseRows(BiRaAddrRole.weightLow)
  multi.weightHighBase :=
    contextReg.baseRows(BiRaAddrRole.weightHigh)
  multi.outputBase :=
    contextReg.baseRows(BiRaAddrRole.outputFull)
  multi.binaryOutputBase :=
    contextReg.baseRows(BiRaAddrRole.outputBinary)
  multi.accumulatorBase :=
    contextReg.baseRows(BiRaAddrRole.accumulator)
  multi.residualBase :=
    contextReg.baseRows(BiRaAddrRole.residual)
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
      BiRaWeightPrecision.w2.U -> 2.U,
      BiRaWeightPrecision.w4.U -> 4.U,
      BiRaWeightPrecision.w8.U -> 8.U,
      BiRaWeightPrecision.w16.U -> 16.U
    )
  )
  multi.activationPrecision := p.activationBits.U
  multi.inputSigned := contextReg.inputSigned
  multi.depthwise :=
    contextReg.arrayMode === BiRaArrayMode.depthwise.U
  multi.columnReduce :=
    contextReg.arrayMode === BiRaArrayMode.columnReduce.U
  multi.bilinearResidual :=
    contextReg.postMode === BiRaPostMode.finalBilinearResidual.U
  multi.shufflePack2 := contextReg.shufflePack2
  multi.shuffleLog2 := log2Ceil(p.maxShuffleScale).U
  multi.writeFullOutput := contextReg.writeFull
  multi.writeBinaryOutput := contextReg.writeBinary

  private val binary = WireDefault(
    0.U.asTypeOf(new BinaryConvolutionCommand(p))
  )
  binary.inputBase :=
    contextReg.baseRows(BiRaAddrRole.input)
  binary.weightBase :=
    contextReg.baseRows(BiRaAddrRole.weightLow)
  binary.binaryOutputBase :=
    contextReg.baseRows(BiRaAddrRole.outputBinary)
  binary.residualBase :=
    contextReg.baseRows(BiRaAddrRole.residual)
  binary.fullOutputBase :=
    contextReg.baseRows(BiRaAddrRole.outputFull)
  binary.correctionBase :=
    contextReg.baseRows(BiRaAddrRole.correction)
  binary.accumulatorBase :=
    contextReg.baseRows(BiRaAddrRole.accumulator)
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
    completionError := BiRaError.none.U
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
class BiRaDataPlane(p: BiRaParams = BiRaParams()) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new BiRaContext(p)))
    val execTask = Flipped(Decoupled(new BiRaExecTask(p)))
    val execCompletion = Decoupled(new BiRaTaskCompletion(p))
    val localWrite = Flipped(Decoupled(new BiRaLocalRowWrite))
    val localReadRequest =
      Flipped(Decoupled(new BiRaLocalRowReadRequest))
    val localReadResponse = Valid(new BiRaLocalRowReadResponse)
    val busy = Output(Bool())
  })

  private val core = Module(new BiRaCore(p))
  private val parameters = Module(new BiRaParameterBuffer(p))
  private val exec = Module(new BiRaExecBridge(p))

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
    writeMemory === BiRaLocalMemory.full.U
  private val writeBinary =
    writeMemory === BiRaLocalMemory.binary.U
  private val writeAccumulator =
    writeMemory === BiRaLocalMemory.accumulator.U
  private val writeParameter =
    writeMemory === BiRaLocalMemory.parameter.U

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
  core.io.fullReadRequest.valid :=
    readState === readIdle &&
      io.localReadRequest.valid &&
      io.localReadRequest.bits.memory === BiRaLocalMemory.full.U
  core.io.fullReadRequest.bits :=
    io.localReadRequest.bits.address(p.fullAddressBits - 1, 0)
  core.io.binaryReadRequest.valid :=
    readState === readIdle &&
      io.localReadRequest.valid &&
      io.localReadRequest.bits.memory === BiRaLocalMemory.binary.U
  core.io.binaryReadRequest.bits :=
    io.localReadRequest.bits.address(p.binaryAddressBits - 1, 0)
  core.io.accumulatorReadRequest.valid :=
    readState === readIdle &&
      io.localReadRequest.valid &&
      io.localReadRequest.bits.memory === BiRaLocalMemory.accumulator.U
  core.io.accumulatorReadRequest.bits :=
    io.localReadRequest.bits.address(
      p.accumulatorAddressBits - 1,
      0
    )

  io.localReadRequest.ready :=
    readState === readIdle &&
      MuxLookup(
        io.localReadRequest.bits.memory,
        true.B
      )(
        Seq(
          BiRaLocalMemory.full.U -> core.io.fullReadRequest.ready,
          BiRaLocalMemory.binary.U ->
            core.io.binaryReadRequest.ready,
          BiRaLocalMemory.accumulator.U ->
            core.io.accumulatorReadRequest.ready
        )
      )
  when(io.localReadRequest.fire) {
    readState := MuxLookup(
      io.localReadRequest.bits.memory,
      readError
    )(
      Seq(
        BiRaLocalMemory.full.U -> waitFull,
        BiRaLocalMemory.binary.U -> waitBinary,
        BiRaLocalMemory.accumulator.U -> waitAccumulator
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
      BiRaError.badRole.U,
      BiRaError.none.U
    )
  when(io.localReadResponse.valid) {
    readState := readIdle
  }
}
