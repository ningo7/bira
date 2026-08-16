package bira

// Two-dimensional Load and Store controllers.

import chisel3._
import chisel3.util._

/** Physical local-memory selector used only inside the accelerator. */
object BiRaLocalMemory {
  val full = 0
  val binary = 1
  val accumulator = 2
  val parameter = 3
}

/** One bus-independent external row read.
  *
  * The later virtual-memory/TileLink bridge may split this request at page and
  * beat boundaries. It must return exactly the requested low bytes in data.
  */
class BiRaExternalReadRequest extends Bundle {
  val virtualAddress = UInt(64.W)
  val bytes = UInt(7.W)
  val translationStatus = UInt(64.W)
}

class BiRaExternalReadResponse extends Bundle {
  val data = UInt(512.W)
  val errorCode = UInt(8.W)
}

/** One bus-independent external row write. */
class BiRaExternalWriteRequest extends Bundle {
  val virtualAddress = UInt(64.W)
  val bytes = UInt(7.W)
  val data = UInt(512.W)
  val translationStatus = UInt(64.W)
}

class BiRaExternalWriteResponse extends Bundle {
  val errorCode = UInt(8.W)
}

/** Common 512-bit local row interfaces.
  *
  * Narrow memories use only the low native row bytes. This keeps the DMA
  * controllers independent of the concrete SPAD/Accumulator implementations.
  */
class BiRaLocalRowWrite extends Bundle {
  val memory = UInt(2.W)
  val address = UInt(16.W)
  val data = UInt(512.W)
}

class BiRaLocalRowReadRequest extends Bundle {
  val memory = UInt(2.W)
  val address = UInt(16.W)
}

class BiRaLocalRowReadResponse extends Bundle {
  val data = UInt(512.W)
  val errorCode = UInt(8.W)
}

/** Shared role mapping and descriptor validation for LoadCtrl and StoreCtrl. */
private object BiRaDmaHelpers {
  def contextFor(
    contexts: Vec[BiRaContext],
    contextId: UInt,
    p: BiRaParams
  ): BiRaContext =
    MuxLookup(
      contextId,
      0.U.asTypeOf(new BiRaContext(p))
    )(
      (0 until p.nContexts).map(index =>
        index.U -> contexts(index)
      )
    )

  def roleAddressValid(context: BiRaContext, role: UInt): Bool =
    MuxLookup(role, false.B)(
      (0 until BiRaAddrRole.count).map(index =>
        index.U -> context.addressValid(index)
      )
    )

  def roleBase(context: BiRaContext, role: UInt): UInt =
    MuxLookup(role, 0.U(16.W))(
      (0 until BiRaAddrRole.count).map(index =>
        index.U -> context.baseRows(index)
      )
    )

  def memoryFor(context: BiRaContext, role: UInt): UInt = {
    val binaryMode =
      context.arrayMode === BiRaArrayMode.binary.U
    MuxCase(
      BiRaLocalMemory.full.U,
      Seq(
        (role === BiRaAddrRole.parameter.U) ->
          BiRaLocalMemory.parameter.U,
        (role === BiRaAddrRole.correction.U) ->
          BiRaLocalMemory.parameter.U,
        (role === BiRaAddrRole.accumulator.U) ->
          BiRaLocalMemory.accumulator.U,
        ((role === BiRaAddrRole.outputBinary.U) ||
          (binaryMode &&
            ((role === BiRaAddrRole.input.U) ||
              (role === BiRaAddrRole.weightLow.U)))) ->
          BiRaLocalMemory.binary.U
      )
    )
  }

  def nativeRowBytes(memory: UInt, p: BiRaParams): UInt =
    MuxLookup(memory, (p.dim * p.activationBits / 8).U(7.W))(
      Seq(
        BiRaLocalMemory.full.U ->
          (p.dim * p.activationBits / 8).U(7.W),
        BiRaLocalMemory.binary.U ->
          ((p.dim + 7) / 8).U(7.W),
        BiRaLocalMemory.accumulator.U ->
          (p.dim * p.accumulatorBits / 8).U(7.W),
        BiRaLocalMemory.parameter.U -> 64.U(7.W)
      )
    )

  def memoryRows(memory: UInt, p: BiRaParams): UInt =
    MuxLookup(memory, p.fullRows.U(32.W))(
      Seq(
        BiRaLocalMemory.full.U -> p.fullRows.U(32.W),
        BiRaLocalMemory.binary.U -> p.binaryRows.U(32.W),
        BiRaLocalMemory.accumulator.U ->
          p.accumulatorRows.U(32.W),
        BiRaLocalMemory.parameter.U ->
          p.parameterRows.U(32.W)
      )
    )

  def loadRoleSupported(context: BiRaContext, role: UInt): Bool = {
    val known = role < BiRaAddrRole.count.U
    val illegalBinaryHigh =
      context.arrayMode === BiRaArrayMode.binary.U &&
        role === BiRaAddrRole.weightHigh.U
    known && !illegalBinaryHigh
  }

  def storeRoleSupported(role: UInt): Bool =
    role === BiRaAddrRole.outputFull.U ||
      role === BiRaAddrRole.outputBinary.U ||
      role === BiRaAddrRole.accumulator.U ||
      role === BiRaAddrRole.input.U

  def descriptorError(
    task: BiRaDmaTask,
    context: BiRaContext,
    memory: UInt,
    baseRow: UInt,
    roleSupported: Bool,
    p: BiRaParams
  ): UInt = {
    val lastIndex =
      Mux(task.rows === 0.U, 0.U, task.rows - 1.U)
    val lastLocal =
      baseRow.pad(32) +
        task.localRowOffset.pad(32) +
        lastIndex.pad(32) * task.localStrideRows.pad(32)
    val lastExternal =
      task.dramVirtualAddress.pad(96) +
        lastIndex.pad(96) * task.dramStrideBytes.pad(96)
    val maxVirtualAddress = ((BigInt(1) << 64) - 1).U(96.W)

    MuxCase(
      BiRaError.none.U,
      Seq(
        (task.contextId >= p.nContexts.U) ->
          BiRaError.invalidContext.U,
        (!context.committed || !context.ready) ->
          BiRaError.contextNotReady.U,
        (context.errorCode =/= BiRaError.none.U) ->
          BiRaError.contextFailed.U,
        (!roleSupported) -> BiRaError.badRole.U,
        (!roleAddressValid(context, task.role)) ->
          BiRaError.missingAddress.U,
        ((task.rows === 0.U) || (task.bytesPerRow === 0.U)) ->
          BiRaError.zeroSize.U,
        (task.bytesPerRow > nativeRowBytes(memory, p)) ->
          BiRaError.rowTooWide.U,
        ((task.localStrideRows === 0.U) ||
          (task.dramStrideBytes < task.bytesPerRow)) ->
          BiRaError.badStride.U,
        (lastLocal >= memoryRows(memory, p)) ->
          BiRaError.localOutOfBounds.U,
        (lastExternal > maxVirtualAddress) ->
          BiRaError.access.U
      )
    )
  }

  /** Zero every byte above the descriptor's valid byte count. */
  def maskRow(data: UInt, bytes: UInt): UInt = {
    val result = Wire(Vec(64, UInt(8.W)))
    for (byte <- 0 until 64) {
      result(byte) := Mux(
        byte.U < bytes,
        data(8 * byte + 7, 8 * byte),
        0.U
      )
    }
    result.asUInt
  }
}

/** One-task-at-a-time 2-D DRAM-to-local row controller.
  *
  * Multiple independent LOAD commands can still wait in the Reservation
  * Station. This controller starts a new task after the previous completion
  * is accepted.
  */
class BiRaLoadCtrl(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new BiRaContext(p)))
    val task = Flipped(Decoupled(new BiRaDmaTask(p)))
    val externalRequest = Decoupled(new BiRaExternalReadRequest)
    val externalResponse =
      Flipped(Decoupled(new BiRaExternalReadResponse))
    val localWrite = Decoupled(new BiRaLocalRowWrite)
    val completion = Decoupled(new BiRaTaskCompletion(p))
    val busy = Output(Bool())
  })

  private val Seq(
    idle,
    requestExternal,
    waitExternal,
    writeLocal,
    respond
  ) = Enum(5)
  private val state = RegInit(idle)
  private val taskReg = Reg(new BiRaDmaTask(p))
  private val memoryReg = Reg(UInt(2.W))
  private val baseRowReg = Reg(UInt(16.W))
  private val rowIndex = RegInit(0.U(14.W))
  private val rowData = Reg(UInt(512.W))
  private val completionError = RegInit(BiRaError.none.U(8.W))

  private val incomingContext =
    BiRaDmaHelpers.contextFor(io.contexts, io.task.bits.contextId, p)
  private val incomingMemory =
    BiRaDmaHelpers.memoryFor(incomingContext, io.task.bits.role)
  private val incomingBase =
    BiRaDmaHelpers.roleBase(incomingContext, io.task.bits.role)
  private val incomingError = BiRaDmaHelpers.descriptorError(
    io.task.bits,
    incomingContext,
    incomingMemory,
    incomingBase,
    BiRaDmaHelpers.loadRoleSupported(
      incomingContext,
      io.task.bits.role
    ),
    p
  )

  io.task.ready := state === idle
  io.busy := state =/= idle

  when(io.task.fire) {
    taskReg := io.task.bits
    memoryReg := incomingMemory
    baseRowReg := incomingBase
    rowIndex := 0.U
    completionError := incomingError
    state := Mux(
      incomingError === BiRaError.none.U,
      requestExternal,
      respond
    )
  }

  private val externalOffset =
    rowIndex.pad(64) * taskReg.dramStrideBytes.pad(64)
  io.externalRequest.valid := state === requestExternal
  io.externalRequest.bits.virtualAddress :=
    (taskReg.dramVirtualAddress.pad(96) + externalOffset)(
      63,
      0
    )
  io.externalRequest.bits.bytes := taskReg.bytesPerRow
  io.externalRequest.bits.translationStatus :=
    taskReg.translationStatus
  when(io.externalRequest.fire) {
    state := waitExternal
  }

  io.externalResponse.ready := state === waitExternal
  when(io.externalResponse.fire) {
    when(io.externalResponse.bits.errorCode =/= BiRaError.none.U) {
      completionError := io.externalResponse.bits.errorCode
      state := respond
    }.otherwise {
      rowData := BiRaDmaHelpers.maskRow(
        io.externalResponse.bits.data,
        taskReg.bytesPerRow
      )
      state := writeLocal
    }
  }

  private val localOffset =
    taskReg.localRowOffset.pad(32) +
      rowIndex.pad(32) * taskReg.localStrideRows.pad(32)
  io.localWrite.valid := state === writeLocal
  io.localWrite.bits.memory := memoryReg
  io.localWrite.bits.address :=
    (baseRowReg.pad(32) + localOffset)(15, 0)
  io.localWrite.bits.data := rowData
  when(io.localWrite.fire) {
    when(rowIndex === taskReg.rows - 1.U) {
      completionError := BiRaError.none.U
      state := respond
    }.otherwise {
      rowIndex := rowIndex + 1.U
      state := requestExternal
    }
  }

  io.completion.valid := state === respond
  io.completion.bits.contextId := taskReg.contextId
  io.completion.bits.commandSequence := taskReg.commandSequence
  io.completion.bits.errorCode := completionError
  when(io.completion.fire) {
    state := idle
  }
}

/** One-task-at-a-time 2-D local-to-DRAM row controller. */
class BiRaStoreCtrl(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new BiRaContext(p)))
    val task = Flipped(Decoupled(new BiRaDmaTask(p)))
    val localReadRequest = Decoupled(new BiRaLocalRowReadRequest)
    val localReadResponse = Flipped(Valid(new BiRaLocalRowReadResponse))
    val externalRequest = Decoupled(new BiRaExternalWriteRequest)
    val externalResponse =
      Flipped(Decoupled(new BiRaExternalWriteResponse))
    val completion = Decoupled(new BiRaTaskCompletion(p))
    val busy = Output(Bool())
  })

  private val Seq(
    idle,
    requestLocal,
    waitLocal,
    requestExternal,
    waitExternal,
    respond
  ) = Enum(6)
  private val state = RegInit(idle)
  private val taskReg = Reg(new BiRaDmaTask(p))
  private val memoryReg = Reg(UInt(2.W))
  private val baseRowReg = Reg(UInt(16.W))
  private val rowIndex = RegInit(0.U(14.W))
  private val rowData = Reg(UInt(512.W))
  private val completionError = RegInit(BiRaError.none.U(8.W))

  private val incomingContext =
    BiRaDmaHelpers.contextFor(io.contexts, io.task.bits.contextId, p)
  private val incomingMemory =
    BiRaDmaHelpers.memoryFor(incomingContext, io.task.bits.role)
  private val incomingBase =
    BiRaDmaHelpers.roleBase(incomingContext, io.task.bits.role)
  private val incomingError = BiRaDmaHelpers.descriptorError(
    io.task.bits,
    incomingContext,
    incomingMemory,
    incomingBase,
    BiRaDmaHelpers.storeRoleSupported(io.task.bits.role),
    p
  )

  io.task.ready := state === idle
  io.busy := state =/= idle

  when(io.task.fire) {
    taskReg := io.task.bits
    memoryReg := incomingMemory
    baseRowReg := incomingBase
    rowIndex := 0.U
    completionError := incomingError
    state := Mux(
      incomingError === BiRaError.none.U,
      requestLocal,
      respond
    )
  }

  private val localOffset =
    taskReg.localRowOffset.pad(32) +
      rowIndex.pad(32) * taskReg.localStrideRows.pad(32)
  io.localReadRequest.valid := state === requestLocal
  io.localReadRequest.bits.memory := memoryReg
  io.localReadRequest.bits.address :=
    (baseRowReg.pad(32) + localOffset)(15, 0)
  when(io.localReadRequest.fire) {
    state := waitLocal
  }

  when(state === waitLocal && io.localReadResponse.valid) {
    when(io.localReadResponse.bits.errorCode =/= BiRaError.none.U) {
      completionError := io.localReadResponse.bits.errorCode
      state := respond
    }.otherwise {
      rowData := BiRaDmaHelpers.maskRow(
        io.localReadResponse.bits.data,
        taskReg.bytesPerRow
      )
      state := requestExternal
    }
  }

  private val externalOffset =
    rowIndex.pad(64) * taskReg.dramStrideBytes.pad(64)
  io.externalRequest.valid := state === requestExternal
  io.externalRequest.bits.virtualAddress :=
    (taskReg.dramVirtualAddress.pad(96) + externalOffset)(
      63,
      0
    )
  io.externalRequest.bits.bytes := taskReg.bytesPerRow
  io.externalRequest.bits.data := rowData
  io.externalRequest.bits.translationStatus :=
    taskReg.translationStatus
  when(io.externalRequest.fire) {
    state := waitExternal
  }

  io.externalResponse.ready := state === waitExternal
  when(io.externalResponse.fire) {
    when(io.externalResponse.bits.errorCode =/= BiRaError.none.U) {
      completionError := io.externalResponse.bits.errorCode
      state := respond
    }.elsewhen(rowIndex === taskReg.rows - 1.U) {
      completionError := BiRaError.none.U
      state := respond
    }.otherwise {
      rowIndex := rowIndex + 1.U
      state := requestLocal
    }
  }

  io.completion.valid := state === respond
  io.completion.bits.contextId := taskReg.contextId
  io.completion.bits.commandSequence := taskReg.commandSequence
  io.completion.bits.errorCode := completionError
  when(io.completion.fire) {
    state := idle
  }
}
