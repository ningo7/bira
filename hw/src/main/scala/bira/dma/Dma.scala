package bira

// Two-dimensional Load and Store controllers.

import chisel3._
import chisel3.util._

/** Physical local-memory selector used only inside the accelerator. */
object LocalMem {
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
class ExtReadReq extends Bundle {
  val virtualAddress = UInt(64.W)
  val bytes = UInt(7.W)
  val translationStatus = UInt(64.W)
}

class ExtReadResp extends Bundle {
  val data = UInt(512.W)
  val errorCode = UInt(8.W)
}

/** One bus-independent external row write. */
class ExtWriteReq extends Bundle {
  val virtualAddress = UInt(64.W)
  val bytes = UInt(7.W)
  val data = UInt(512.W)
  val translationStatus = UInt(64.W)
}

class ExtWriteResp extends Bundle {
  val errorCode = UInt(8.W)
}

/** Common 512-bit local row interfaces.
  *
  * Narrow memories use only the low native row bytes. This keeps the DMA
  * controllers independent of the concrete SPAD/Accumulator implementations.
  */
class LocalWrite extends Bundle {
  val memory = UInt(2.W)
  val address = UInt(16.W)
  val data = UInt(512.W)
}

class LocalReadReq extends Bundle {
  val memory = UInt(2.W)
  val address = UInt(16.W)
}

class LocalReadResp extends Bundle {
  val data = UInt(512.W)
  val errorCode = UInt(8.W)
}

/** Shared role mapping and descriptor validation for LoadCtrl and StoreCtrl. */
private object DmaUtil {
  def contextFor(
    contexts: Vec[Context],
    contextId: UInt,
    p: AccelParams
  ): Context =
    MuxLookup(
      contextId,
      0.U.asTypeOf(new Context(p))
    )(
      (0 until p.nContexts).map(index =>
        index.U -> contexts(index)
      )
    )

  def roleAddressValid(context: Context, role: UInt): Bool =
    MuxLookup(role, false.B)(
      (0 until AddrRole.count).map(index =>
        index.U -> context.addressValid(index)
      )
    )

  def roleBase(context: Context, role: UInt): UInt =
    MuxLookup(role, 0.U(16.W))(
      (0 until AddrRole.count).map(index =>
        index.U -> context.baseRows(index)
      )
    )

  def memoryFor(context: Context, role: UInt): UInt = {
    val binaryMode =
      context.arrayMode === ArrayMode.binary.U
    MuxCase(
      LocalMem.full.U,
      Seq(
        (role === AddrRole.parameter.U) ->
          LocalMem.parameter.U,
        (role === AddrRole.correction.U) ->
          LocalMem.parameter.U,
        (role === AddrRole.accumulator.U) ->
          LocalMem.accumulator.U,
        ((role === AddrRole.outputBinary.U) ||
          (binaryMode &&
            ((role === AddrRole.input.U) ||
              (role === AddrRole.weightLow.U)))) ->
          LocalMem.binary.U
      )
    )
  }

  def nativeRowBytes(memory: UInt, p: AccelParams): UInt =
    MuxLookup(memory, (p.dim * p.activationBits / 8).U(7.W))(
      Seq(
        LocalMem.full.U ->
          (p.dim * p.activationBits / 8).U(7.W),
        LocalMem.binary.U ->
          ((p.dim + 7) / 8).U(7.W),
        LocalMem.accumulator.U ->
          (p.dim * p.accumulatorBits / 8).U(7.W),
        LocalMem.parameter.U -> 64.U(7.W)
      )
    )

  def memoryRows(memory: UInt, p: AccelParams): UInt =
    MuxLookup(memory, p.fullRows.U(32.W))(
      Seq(
        LocalMem.full.U -> p.fullRows.U(32.W),
        LocalMem.binary.U -> p.binaryRows.U(32.W),
        LocalMem.accumulator.U ->
          p.accumulatorRows.U(32.W),
        LocalMem.parameter.U ->
          p.parameterRows.U(32.W)
      )
    )

  def loadRoleSupported(context: Context, role: UInt): Bool = {
    val known = role < AddrRole.count.U
    val illegalBinaryHigh =
      context.arrayMode === ArrayMode.binary.U &&
        role === AddrRole.weightHigh.U
    known && !illegalBinaryHigh
  }

  def storeRoleSupported(role: UInt): Bool =
    role === AddrRole.outputFull.U ||
      role === AddrRole.outputBinary.U ||
      role === AddrRole.accumulator.U ||
      role === AddrRole.input.U

  /** Descriptor checks which do not depend on the final row addresses. */
  def descriptorPrefixError(
    task: DmaTask,
    context: Context,
    memory: UInt,
    roleSupported: Bool,
    p: AccelParams
  ): UInt =
    MuxCase(
      ErrorCode.none.U,
      Seq(
        (task.contextId >= p.nContexts.U) ->
          ErrorCode.invalidContext.U,
        (!context.committed || !context.ready) ->
          ErrorCode.contextNotReady.U,
        (context.errorCode =/= ErrorCode.none.U) ->
          ErrorCode.contextFailed.U,
        (!roleSupported) -> ErrorCode.badRole.U,
        (!roleAddressValid(context, task.role)) ->
          ErrorCode.missingAddress.U,
        ((task.rows === 0.U) || (task.bytesPerRow === 0.U)) ->
          ErrorCode.zeroSize.U,
        (task.bytesPerRow > nativeRowBytes(memory, p)) ->
          ErrorCode.rowTooWide.U,
        ((task.localStrideRows === 0.U) ||
          (task.dramStrideBytes < task.bytesPerRow)) ->
          ErrorCode.badStride.U
      )
    )

  def descriptorError(
    task: DmaTask,
    context: Context,
    memory: UInt,
    baseRow: UInt,
    roleSupported: Bool,
    p: AccelParams
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
      ErrorCode.none.U,
      Seq(
        (task.contextId >= p.nContexts.U) ->
          ErrorCode.invalidContext.U,
        (!context.committed || !context.ready) ->
          ErrorCode.contextNotReady.U,
        (context.errorCode =/= ErrorCode.none.U) ->
          ErrorCode.contextFailed.U,
        (!roleSupported) -> ErrorCode.badRole.U,
        (!roleAddressValid(context, task.role)) ->
          ErrorCode.missingAddress.U,
        ((task.rows === 0.U) || (task.bytesPerRow === 0.U)) ->
          ErrorCode.zeroSize.U,
        (task.bytesPerRow > nativeRowBytes(memory, p)) ->
          ErrorCode.rowTooWide.U,
        ((task.localStrideRows === 0.U) ||
          (task.dramStrideBytes < task.bytesPerRow)) ->
          ErrorCode.badStride.U,
        (lastLocal >= memoryRows(memory, p)) ->
          ErrorCode.localOutOfBounds.U,
        (lastExternal > maxVirtualAddress) ->
          ErrorCode.access.U
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
class LoadCtrl(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new Context(p)))
    val task = Flipped(Decoupled(new DmaTask(p)))
    val externalRequest = Decoupled(new ExtReadReq)
    val externalResponse =
      Flipped(Decoupled(new ExtReadResp))
    val localWrite = Decoupled(new LocalWrite)
    val completion = Decoupled(new Completion(p))
    val busy = Output(Bool())
  })

  private val Seq(
    idle,
    validate,
    requestExternal,
    waitExternal,
    writeLocal,
    respond
  ) = Enum(6)
  private val state = RegInit(idle)
  private val taskReg = Reg(new DmaTask(p))
  private val memoryReg = Reg(UInt(2.W))
  private val baseRowReg = Reg(UInt(16.W))
  // Hold the current row addresses explicitly.  Advancing them once per
  // completed row keeps stride multiplication out of the live DMA
  // ready/valid path.
  private val localAddressReg = Reg(UInt(16.W))
  private val externalAddressReg = Reg(UInt(64.W))
  private val rowIndex = RegInit(0.U(14.W))
  private val rowData = Reg(UInt(512.W))
  private val completionError = RegInit(ErrorCode.none.U(8.W))
  private val prefixErrorReg = Reg(UInt(8.W))
  private val localSpanReg = Reg(UInt(20.W))
  private val externalSpanReg = Reg(UInt(30.W))

  private val incomingContext =
    DmaUtil.contextFor(io.contexts, io.task.bits.contextId, p)
  private val incomingMemory =
    DmaUtil.memoryFor(incomingContext, io.task.bits.role)
  private val incomingBase =
    DmaUtil.roleBase(incomingContext, io.task.bits.role)
  private val incomingPrefixError = DmaUtil.descriptorPrefixError(
    io.task.bits,
    incomingContext,
    incomingMemory,
    DmaUtil.loadRoleSupported(
      incomingContext,
      io.task.bits.role
    ),
    p
  )

  io.task.ready := state === idle
  io.busy := state =/= idle

  when(io.task.fire) {
    val lastIndex =
      Mux(io.task.bits.rows === 0.U, 0.U, io.task.bits.rows - 1.U)
    taskReg := io.task.bits
    memoryReg := incomingMemory
    baseRowReg := incomingBase
    localAddressReg :=
      (incomingBase.pad(32) + io.task.bits.localRowOffset.pad(32))(15, 0)
    externalAddressReg := io.task.bits.dramVirtualAddress
    rowIndex := 0.U
    prefixErrorReg := incomingPrefixError
    localSpanReg := lastIndex * io.task.bits.localStrideRows
    externalSpanReg := lastIndex * io.task.bits.dramStrideBytes
    state := validate
  }

  when(state === validate) {
    val lastLocal =
      baseRowReg.pad(22) +
        taskReg.localRowOffset.pad(22) +
        localSpanReg.pad(22)
    val lastExternal =
      taskReg.dramVirtualAddress.pad(65) +
        externalSpanReg.pad(65)
    val rangeError = Mux(
      lastLocal >= DmaUtil.memoryRows(memoryReg, p),
      ErrorCode.localOutOfBounds.U,
      Mux(lastExternal(64), ErrorCode.access.U, ErrorCode.none.U)
    )
    val finalError = Mux(
      prefixErrorReg =/= ErrorCode.none.U,
      prefixErrorReg,
      rangeError
    )
    completionError := finalError
    state := Mux(
      finalError === ErrorCode.none.U,
      requestExternal,
      respond
    )
  }

  io.externalRequest.valid := state === requestExternal
  io.externalRequest.bits.virtualAddress := externalAddressReg
  io.externalRequest.bits.bytes := taskReg.bytesPerRow
  io.externalRequest.bits.translationStatus :=
    taskReg.translationStatus
  when(io.externalRequest.fire) {
    state := waitExternal
  }

  io.externalResponse.ready := state === waitExternal
  when(io.externalResponse.fire) {
    when(io.externalResponse.bits.errorCode =/= ErrorCode.none.U) {
      completionError := io.externalResponse.bits.errorCode
      state := respond
    }.otherwise {
      rowData := DmaUtil.maskRow(
        io.externalResponse.bits.data,
        taskReg.bytesPerRow
      )
      state := writeLocal
    }
  }

  io.localWrite.valid := state === writeLocal
  io.localWrite.bits.memory := memoryReg
  io.localWrite.bits.address := localAddressReg
  io.localWrite.bits.data := rowData
  when(io.localWrite.fire) {
    when(rowIndex === taskReg.rows - 1.U) {
      completionError := ErrorCode.none.U
      state := respond
    }.otherwise {
      rowIndex := rowIndex + 1.U
      localAddressReg := localAddressReg + taskReg.localStrideRows
      externalAddressReg :=
        externalAddressReg + taskReg.dramStrideBytes
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
class StoreCtrl(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new Context(p)))
    val task = Flipped(Decoupled(new DmaTask(p)))
    val localReadRequest = Decoupled(new LocalReadReq)
    val localReadResponse = Flipped(Valid(new LocalReadResp))
    val externalRequest = Decoupled(new ExtWriteReq)
    val externalResponse =
      Flipped(Decoupled(new ExtWriteResp))
    val completion = Decoupled(new Completion(p))
    val busy = Output(Bool())
  })

  private val Seq(
    idle,
    validate,
    requestLocal,
    waitLocal,
    requestExternal,
    waitExternal,
    respond
  ) = Enum(7)
  private val state = RegInit(idle)
  private val taskReg = Reg(new DmaTask(p))
  private val memoryReg = Reg(UInt(2.W))
  private val baseRowReg = Reg(UInt(16.W))
  // As in LoadCtrl, use running addresses so normal row traffic contains
  // only registered addresses and stride adders.
  private val localAddressReg = Reg(UInt(16.W))
  private val externalAddressReg = Reg(UInt(64.W))
  private val rowIndex = RegInit(0.U(14.W))
  private val rowData = Reg(UInt(512.W))
  private val completionError = RegInit(ErrorCode.none.U(8.W))
  private val prefixErrorReg = Reg(UInt(8.W))
  private val localSpanReg = Reg(UInt(20.W))
  private val externalSpanReg = Reg(UInt(30.W))

  private val incomingContext =
    DmaUtil.contextFor(io.contexts, io.task.bits.contextId, p)
  private val incomingMemory =
    DmaUtil.memoryFor(incomingContext, io.task.bits.role)
  private val incomingBase =
    DmaUtil.roleBase(incomingContext, io.task.bits.role)
  private val incomingPrefixError = DmaUtil.descriptorPrefixError(
    io.task.bits,
    incomingContext,
    incomingMemory,
    DmaUtil.storeRoleSupported(io.task.bits.role),
    p
  )

  io.task.ready := state === idle
  io.busy := state =/= idle

  when(io.task.fire) {
    val lastIndex =
      Mux(io.task.bits.rows === 0.U, 0.U, io.task.bits.rows - 1.U)
    taskReg := io.task.bits
    memoryReg := incomingMemory
    baseRowReg := incomingBase
    localAddressReg :=
      (incomingBase.pad(32) + io.task.bits.localRowOffset.pad(32))(15, 0)
    externalAddressReg := io.task.bits.dramVirtualAddress
    rowIndex := 0.U
    prefixErrorReg := incomingPrefixError
    localSpanReg := lastIndex * io.task.bits.localStrideRows
    externalSpanReg := lastIndex * io.task.bits.dramStrideBytes
    state := validate
  }

  when(state === validate) {
    val lastLocal =
      baseRowReg.pad(22) +
        taskReg.localRowOffset.pad(22) +
        localSpanReg.pad(22)
    val lastExternal =
      taskReg.dramVirtualAddress.pad(65) +
        externalSpanReg.pad(65)
    val rangeError = Mux(
      lastLocal >= DmaUtil.memoryRows(memoryReg, p),
      ErrorCode.localOutOfBounds.U,
      Mux(lastExternal(64), ErrorCode.access.U, ErrorCode.none.U)
    )
    val finalError = Mux(
      prefixErrorReg =/= ErrorCode.none.U,
      prefixErrorReg,
      rangeError
    )
    completionError := finalError
    state := Mux(
      finalError === ErrorCode.none.U,
      requestLocal,
      respond
    )
  }

  io.localReadRequest.valid := state === requestLocal
  io.localReadRequest.bits.memory := memoryReg
  io.localReadRequest.bits.address := localAddressReg
  when(io.localReadRequest.fire) {
    state := waitLocal
  }

  when(state === waitLocal && io.localReadResponse.valid) {
    when(io.localReadResponse.bits.errorCode =/= ErrorCode.none.U) {
      completionError := io.localReadResponse.bits.errorCode
      state := respond
    }.otherwise {
      rowData := DmaUtil.maskRow(
        io.localReadResponse.bits.data,
        taskReg.bytesPerRow
      )
      state := requestExternal
    }
  }

  io.externalRequest.valid := state === requestExternal
  io.externalRequest.bits.virtualAddress := externalAddressReg
  io.externalRequest.bits.bytes := taskReg.bytesPerRow
  io.externalRequest.bits.data := rowData
  io.externalRequest.bits.translationStatus :=
    taskReg.translationStatus
  when(io.externalRequest.fire) {
    state := waitExternal
  }

  io.externalResponse.ready := state === waitExternal
  when(io.externalResponse.fire) {
    when(io.externalResponse.bits.errorCode =/= ErrorCode.none.U) {
      completionError := io.externalResponse.bits.errorCode
      state := respond
    }.elsewhen(rowIndex === taskReg.rows - 1.U) {
      completionError := ErrorCode.none.U
      state := respond
    }.otherwise {
      rowIndex := rowIndex + 1.U
      localAddressReg := localAddressReg + taskReg.localStrideRows
      externalAddressReg :=
        externalAddressReg + taskReg.dramStrideBytes
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
