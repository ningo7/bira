package bira

// Load/Execute/Store scheduling and dependency tracking.

import chisel3._
import chisel3.util._

object TaskType {
  val load = 0
  val execute = 1
  val store = 2
}

/** Bank-level read/write footprint used by the first Reservation Station. */
class ResourceSet(p: AccelParams) extends Bundle {
  val fullRead = UInt(p.fullBanks.W)
  val fullWrite = UInt(p.fullBanks.W)
  val binaryRead = UInt(p.binaryBanks.W)
  val binaryWrite = UInt(p.binaryBanks.W)
  val accumulatorRead = UInt(p.accumulatorBanks.W)
  val accumulatorWrite = UInt(p.accumulatorBanks.W)
  val parameterRead = Bool()
  val parameterWrite = Bool()
}

class RsEntry(p: AccelParams) extends Bundle {
  val valid = Bool()
  val issued = Bool()
  val taskType = UInt(2.W)
  val contextId = UInt(p.contextIdBits.W)
  val commandSequence = UInt(16.W)
  val dmaTask = new DmaTask(p)
  val execTask = new ExecTask(p)
  val resources = new ResourceSet(p)
  // A bit remains set while the older entry occupying that reservation slot
  // is an unresolved resource dependency of this task.
  val dependencies = UInt(p.rsEntries.W)
}

/** Unified Reservation Station with independent Load/Execute/Store issue.
  *
  * Entries may issue out of program order only when their bank-level resource
  * sets have no RAW, WAR, or WAW conflict with any older live entry. The first
  * implementation deliberately treats an entire Parameter Buffer as one
  * resource; row-range refinement can be added without changing the ISA.
  */
class Scheduler(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new Context(p)))

    val loadEnqueue = Flipped(Decoupled(new DmaTask(p)))
    val execEnqueue = Flipped(Decoupled(new ExecTask(p)))
    val storeEnqueue = Flipped(Decoupled(new DmaTask(p)))

    val loadIssue = Decoupled(new DmaTask(p))
    val execIssue = Decoupled(new ExecTask(p))
    val storeIssue = Decoupled(new DmaTask(p))

    val loadCompletion =
      Flipped(Decoupled(new Completion(p)))
    val execCompletion =
      Flipped(Decoupled(new Completion(p)))
    val storeCompletion =
      Flipped(Decoupled(new Completion(p)))
    val completion = Decoupled(new Completion(p))

    val status = Output(new SchedStatus)
  })

  private val entries = RegInit(
    VecInit.fill(p.rsEntries)(
      0.U.asTypeOf(new RsEntry(p))
    )
  )
  // One registered preparation slot cuts task-dependent DMA range arithmetic
  // and the EXEC resource-cache mux away from reservation-entry insertion.
  private val preparedValid = RegInit(false.B)
  private val preparedEntry = RegInit(
    0.U.asTypeOf(new RsEntry(p))
  )
  private val dmaRangeValid = RegInit(false.B)
  private val dmaRangeTask = Reg(new DmaTask(p))
  private val dmaRangeIsLoad = RegInit(false.B)
  private val dmaRangeBinaryMode = RegInit(false.B)
  private val dmaRangeFirstRow = Reg(UInt(22.W))
  private val dmaRangeSpan = Reg(UInt(20.W))

  // EXEC footprints depend only on immutable committed-context fields. Build
  // each footprint once and keep it beside the reservation station instead of
  // putting the complete dimension arithmetic on every enqueue path.
  private val execResourceCompiler =
    Module(new ResourceCompiler(p))
  private val execResourceCache = RegInit(
    VecInit.fill(p.nContexts)(
      0.U.asTypeOf(new ResourceSet(p))
    )
  )
  private val execResourceValid = RegInit(VecInit.fill(p.nContexts)(false.B))
  private val compileNeeded = VecInit((0 until p.nContexts).map { index =>
    io.contexts(index).committed && !execResourceValid(index)
  })
  private val compileContextOH = PriorityEncoderOH(compileNeeded.asUInt)
  private val compileContextIndex =
    OHToUInt(compileContextOH)(p.contextIndexBits - 1, 0)
  private val compilingContextIndex = RegInit(0.U(p.contextIndexBits.W))

  execResourceCompiler.io.input.valid := compileNeeded.asUInt.orR
  execResourceCompiler.io.input.bits := io.contexts(compileContextIndex)
  execResourceCompiler.io.output.ready := true.B

  when(execResourceCompiler.io.input.fire) {
    compilingContextIndex := compileContextIndex
  }
  for (index <- 0 until p.nContexts) {
    when(!io.contexts(index).committed) {
      execResourceValid(index) := false.B
    }
  }
  when(
    execResourceCompiler.io.output.fire &&
      io.contexts(compilingContextIndex).committed
  ) {
    execResourceCache(compilingContextIndex) :=
      execResourceCompiler.io.output.bits
    execResourceValid(compilingContextIndex) := true.B
  }

  private val validBits = VecInit(entries.map(_.valid)).asUInt
  private val hasFreeEntry = !validBits.andR
  private val freeEntryOH = PriorityEncoderOH(~validBits)
  private val freeEntryIndex =
    OHToUInt(freeEntryOH)(p.reservationIndexBits - 1, 0)

  private val loadCount =
    PopCount(entries.map(e => e.valid && e.taskType === TaskType.load.U)) +
      (preparedValid && preparedEntry.taskType === TaskType.load.U) +
      (dmaRangeValid && dmaRangeIsLoad)
  private val execCount =
    PopCount(entries.map(e =>
      e.valid && e.taskType === TaskType.execute.U
    )) +
      (preparedValid && preparedEntry.taskType === TaskType.execute.U)
  private val storeCount =
    PopCount(entries.map(e => e.valid && e.taskType === TaskType.store.U)) +
      (preparedValid && preparedEntry.taskType === TaskType.store.U) +
      (dmaRangeValid && !dmaRangeIsLoad)

  io.status.loadQueueCount := loadCount
  io.status.execQueueCount := execCount
  io.status.storeQueueCount := storeCount
  io.status.loadBusy :=
    entries.exists(e =>
      e.valid && e.issued && e.taskType === TaskType.load.U
    )
  io.status.execBusy :=
    entries.exists(e =>
      e.valid && e.issued && e.taskType === TaskType.execute.U
    )
  io.status.storeBusy :=
    entries.exists(e =>
      e.valid && e.issued && e.taskType === TaskType.store.U
    )

  // The command frontend currently presents at most one enqueue per cycle.
  // Priority logic keeps the scheduler well-defined if a standalone test
  // drives more than one.
  io.loadEnqueue.ready :=
    !preparedValid &&
      !dmaRangeValid &&
      hasFreeEntry &&
      loadCount < p.loadQueueEntries.U
  io.execEnqueue.ready :=
    !preparedValid &&
      !dmaRangeValid &&
      hasFreeEntry &&
      !io.loadEnqueue.valid &&
      execCount < p.execQueueEntries.U &&
      io.execEnqueue.bits.contextId < p.nContexts.U &&
      execResourceValid(
        io.execEnqueue.bits.contextId(p.contextIndexBits - 1, 0)
      )
  io.storeEnqueue.ready :=
    !preparedValid &&
      !dmaRangeValid &&
      hasFreeEntry &&
      !io.loadEnqueue.valid &&
      !io.execEnqueue.valid &&
      storeCount < p.storeQueueEntries.U

  private def contextFor(id: UInt): Context = {
    val index = id(p.contextIndexBits - 1, 0)
    io.contexts(index)
  }

  /** Conservative mask for every bank touched by an already-bounded range. */
  private def bankMaskFromRange(
    firstRow: UInt,
    lastRow: UInt,
    rowCount: UInt,
    banks: Int
  ): UInt = {
    VecInit((0 until banks).map { bank =>
      val bankStart = (bank * p.bankRows).U(22.W)
      val bankEnd = ((bank + 1) * p.bankRows - 1).U(22.W)
      rowCount =/= 0.U && firstRow <= bankEnd && lastRow >= bankStart
    }).asUInt
  }

  private def emptyResources: ResourceSet =
    0.U.asTypeOf(new ResourceSet(p))

  private def dmaResourcesFromRange(
    task: DmaTask,
    isLoad: Bool,
    binaryMode: Bool,
    firstRow: UInt,
    lastRow: UInt
  ): ResourceSet = {
    val resources = WireDefault(emptyResources)
    val fullMask =
      bankMaskFromRange(firstRow, lastRow, task.rows, p.fullBanks)
    val binaryMask =
      bankMaskFromRange(
        firstRow,
        lastRow,
        task.rows,
        p.binaryBanks
      )
    val accumulatorMask =
      bankMaskFromRange(
        firstRow,
        lastRow,
        task.rows,
        p.accumulatorBanks
      )

    val targetsFull =
      task.role === AddrRole.weightHigh.U ||
        task.role === AddrRole.residual.U ||
        task.role === AddrRole.outputFull.U ||
        ((task.role === AddrRole.input.U ||
          task.role === AddrRole.weightLow.U) && !binaryMode)
    val targetsBinary =
      task.role === AddrRole.outputBinary.U ||
        ((task.role === AddrRole.input.U ||
          task.role === AddrRole.weightLow.U) && binaryMode)
    val targetsAccumulator =
      task.role === AddrRole.accumulator.U
    val targetsParameter =
      task.role === AddrRole.parameter.U ||
        task.role === AddrRole.correction.U

    when(targetsFull) {
      when(isLoad) {
        resources.fullWrite := fullMask
      }.otherwise {
        resources.fullRead := fullMask
      }
    }
    when(targetsBinary) {
      when(isLoad) {
        resources.binaryWrite := binaryMask
      }.otherwise {
        resources.binaryRead := binaryMask
      }
    }
    when(targetsAccumulator) {
      when(isLoad) {
        resources.accumulatorWrite := accumulatorMask
      }.otherwise {
        resources.accumulatorRead := accumulatorMask
      }
    }
    when(targetsParameter) {
      resources.parameterWrite := isLoad
      resources.parameterRead := !isLoad
    }
    resources
  }

  private val selectedDmaTask = Mux(
    io.loadEnqueue.valid,
    io.loadEnqueue.bits,
    io.storeEnqueue.bits
  )
  private val selectedDmaContext = contextFor(selectedDmaTask.contextId)

  when(io.loadEnqueue.fire || io.storeEnqueue.fire) {
    val rowDelta = selectedDmaTask.rows - 1.U
    dmaRangeValid := true.B
    dmaRangeTask := selectedDmaTask
    dmaRangeIsLoad := io.loadEnqueue.fire
    dmaRangeBinaryMode :=
      selectedDmaContext.arrayMode === ArrayMode.binary.U
    dmaRangeFirstRow :=
      selectedDmaContext.baseRows(selectedDmaTask.role).pad(22) +
        selectedDmaTask.localRowOffset.pad(22)
    dmaRangeSpan := rowDelta * selectedDmaTask.localStrideRows
  }

  when(io.execEnqueue.fire) {
    preparedValid := true.B
    preparedEntry.valid := true.B
    preparedEntry.issued := false.B
    preparedEntry.taskType := TaskType.execute.U
    preparedEntry.contextId := io.execEnqueue.bits.contextId
    preparedEntry.commandSequence := io.execEnqueue.bits.commandSequence
    preparedEntry.dmaTask := 0.U.asTypeOf(new DmaTask(p))
    preparedEntry.execTask := io.execEnqueue.bits
    preparedEntry.resources := execResourceCache(
      io.execEnqueue.bits.contextId(p.contextIndexBits - 1, 0)
    )
    preparedEntry.dependencies := 0.U
  }

  when(dmaRangeValid && !preparedValid) {
    val lastRow = dmaRangeFirstRow + dmaRangeSpan.pad(22)
    dmaRangeValid := false.B
    preparedValid := true.B
    preparedEntry.valid := true.B
    preparedEntry.issued := false.B
    preparedEntry.taskType := Mux(
      dmaRangeIsLoad,
      TaskType.load.U,
      TaskType.store.U
    )
    preparedEntry.contextId := dmaRangeTask.contextId
    preparedEntry.commandSequence := dmaRangeTask.commandSequence
    preparedEntry.dmaTask := dmaRangeTask
    preparedEntry.execTask := 0.U.asTypeOf(new ExecTask(p))
    preparedEntry.resources := dmaResourcesFromRange(
      dmaRangeTask,
      dmaRangeIsLoad,
      dmaRangeBinaryMode,
      dmaRangeFirstRow,
      lastRow
    )
    preparedEntry.dependencies := 0.U
  }

  private def hasHazard(
    younger: ResourceSet,
    older: ResourceSet
  ): Bool = {
    val raw =
      (younger.fullRead & older.fullWrite).orR ||
        (younger.binaryRead & older.binaryWrite).orR ||
        (younger.accumulatorRead & older.accumulatorWrite).orR ||
        (younger.parameterRead && older.parameterWrite)
    val war =
      (younger.fullWrite & older.fullRead).orR ||
        (younger.binaryWrite & older.binaryRead).orR ||
        (younger.accumulatorWrite & older.accumulatorRead).orR ||
        (younger.parameterWrite && older.parameterRead)
    val waw =
      (younger.fullWrite & older.fullWrite).orR ||
        (younger.binaryWrite & older.binaryWrite).orR ||
        (younger.accumulatorWrite & older.accumulatorWrite).orR ||
        (younger.parameterWrite && older.parameterWrite)
    raw || war || waw
  }

  private val completionMatches = VecInit(entries.map { entry =>
    entry.valid &&
    entry.contextId === io.completion.bits.contextId &&
    entry.commandSequence === io.completion.bits.commandSequence
  })
  private val retiringMask = Mux(
    io.completion.fire,
    completionMatches.asUInt,
    0.U
  )
  private val newDependencies = VecInit(entries.zipWithIndex.map {
    case (older, index) =>
    older.valid &&
    !retiringMask(index) &&
    hasHazard(preparedEntry.resources, older.resources)
  }).asUInt

  private val enqueueFire = preparedValid && hasFreeEntry
  when(enqueueFire) {
    preparedValid := false.B
    entries(freeEntryIndex).valid := true.B
    entries(freeEntryIndex).issued := false.B
    entries(freeEntryIndex).taskType := preparedEntry.taskType
    entries(freeEntryIndex).contextId := preparedEntry.contextId
    entries(freeEntryIndex).commandSequence := preparedEntry.commandSequence
    entries(freeEntryIndex).dmaTask := preparedEntry.dmaTask
    entries(freeEntryIndex).execTask := preparedEntry.execTask
    entries(freeEntryIndex).resources := preparedEntry.resources
    entries(freeEntryIndex).dependencies := newDependencies
  }

  private val eligible = VecInit(entries.map { entry =>
    entry.valid && !entry.issued && !entry.dependencies.orR
  })

  private def selectEligible(taskType: Int): (Bool, UInt) = {
    val matches = VecInit((0 until p.rsEntries).map { i =>
      eligible(i) && entries(i).taskType === taskType.U
    })
    val oneHot = PriorityEncoderOH(matches.asUInt)
    (matches.asUInt.orR,
      OHToUInt(oneHot)(p.reservationIndexBits - 1, 0))
  }

  private val (loadIssueValid, loadIssueIndex) =
    selectEligible(TaskType.load)
  private val (execIssueValid, execIssueIndex) =
    selectEligible(TaskType.execute)
  private val (storeIssueValid, storeIssueIndex) =
    selectEligible(TaskType.store)

  // Keep reservation-station selection and the receiving controller's task
  // decode in separate timing stages.  These are single elastic pipeline
  // slots, not work queues: a consumed task can be replaced in the same
  // cycle, preserving an issue interval of one cycle.
  private val loadIssueStage =
    Module(new ElasticRegister(new DmaTask(p)))
  private val execIssueStage =
    Module(new ElasticRegister(new ExecTask(p)))
  private val storeIssueStage =
    Module(new ElasticRegister(new DmaTask(p)))

  loadIssueStage.io.enq.valid := loadIssueValid
  loadIssueStage.io.enq.bits := entries(loadIssueIndex).dmaTask
  execIssueStage.io.enq.valid := execIssueValid
  execIssueStage.io.enq.bits := entries(execIssueIndex).execTask
  storeIssueStage.io.enq.valid := storeIssueValid
  storeIssueStage.io.enq.bits := entries(storeIssueIndex).dmaTask

  io.loadIssue.valid := loadIssueStage.io.deq.valid
  io.loadIssue.bits := loadIssueStage.io.deq.bits
  loadIssueStage.io.deq.ready := io.loadIssue.ready
  io.execIssue.valid := execIssueStage.io.deq.valid
  io.execIssue.bits := execIssueStage.io.deq.bits
  execIssueStage.io.deq.ready := io.execIssue.ready
  io.storeIssue.valid := storeIssueStage.io.deq.valid
  io.storeIssue.bits := storeIssueStage.io.deq.bits
  storeIssueStage.io.deq.ready := io.storeIssue.ready

  // Reserve the entry when it enters the issue stage.  Waiting until the
  // downstream transfer would allow the same unissued entry to be selected
  // again while a controller is applying backpressure.
  when(loadIssueStage.io.enq.fire) {
    entries(loadIssueIndex).issued := true.B
  }
  when(execIssueStage.io.enq.fire) {
    entries(execIssueIndex).issued := true.B
  }
  when(storeIssueStage.io.enq.fire) {
    entries(storeIssueIndex).issued := true.B
  }

  private val completionArbiter =
    Module(new Arbiter(new Completion(p), 3))
  completionArbiter.io.in(0) <> io.execCompletion
  completionArbiter.io.in(1) <> io.loadCompletion
  completionArbiter.io.in(2) <> io.storeCompletion
  io.completion <> completionArbiter.io.out

  when(io.completion.fire) {
    assert(
      PopCount(completionMatches) === 1.U,
      "completion must match exactly one Reservation Station entry"
    )
    val completedIndex =
      OHToUInt(completionMatches.asUInt)(
        p.reservationIndexBits - 1,
        0
      )
    assert(entries(completedIndex).issued)
    entries(completedIndex).valid := false.B
    entries(completedIndex).issued := false.B
    for (index <- 0 until p.rsEntries) {
      // A newly allocated entry already excluded the retiring slot while
      // constructing newDependencies; do not overwrite its new dependency
      // vector with the old contents of the free slot.
      when(!(enqueueFire && freeEntryIndex === index.U)) {
        entries(index).dependencies :=
          entries(index).dependencies & ~completionMatches.asUInt
      }
    }
  }
}
