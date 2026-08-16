package bira

// Load/Execute/Store scheduling and dependency tracking.

import chisel3._
import chisel3.util._

object BiRaTaskType {
  val load = 0
  val execute = 1
  val store = 2
}

/** Bank-level read/write footprint used by the first Reservation Station. */
class BiRaResourceSet(p: BiRaParams) extends Bundle {
  val fullRead = UInt(p.fullBanks.W)
  val fullWrite = UInt(p.fullBanks.W)
  val binaryRead = UInt(p.binaryBanks.W)
  val binaryWrite = UInt(p.binaryBanks.W)
  val accumulatorRead = UInt(p.accumulatorBanks.W)
  val accumulatorWrite = UInt(p.accumulatorBanks.W)
  val parameterRead = Bool()
  val parameterWrite = Bool()
}

class BiRaReservationEntry(p: BiRaParams) extends Bundle {
  val valid = Bool()
  val issued = Bool()
  val taskType = UInt(2.W)
  val contextId = UInt(p.contextIdBits.W)
  val commandSequence = UInt(16.W)
  val dmaTask = new BiRaDmaTask(p)
  val execTask = new BiRaExecTask(p)
  val resources = new BiRaResourceSet(p)
}

/** Unified Reservation Station with independent Load/Execute/Store issue.
  *
  * Entries may issue out of program order only when their bank-level resource
  * sets have no RAW, WAR, or WAW conflict with any older live entry. The first
  * implementation deliberately treats an entire Parameter Buffer as one
  * resource; row-range refinement can be added without changing the ISA.
  */
class BiRaScheduler(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val contexts = Input(Vec(p.nContexts, new BiRaContext(p)))

    val loadEnqueue = Flipped(Decoupled(new BiRaDmaTask(p)))
    val execEnqueue = Flipped(Decoupled(new BiRaExecTask(p)))
    val storeEnqueue = Flipped(Decoupled(new BiRaDmaTask(p)))

    val loadIssue = Decoupled(new BiRaDmaTask(p))
    val execIssue = Decoupled(new BiRaExecTask(p))
    val storeIssue = Decoupled(new BiRaDmaTask(p))

    val loadCompletion =
      Flipped(Decoupled(new BiRaTaskCompletion(p)))
    val execCompletion =
      Flipped(Decoupled(new BiRaTaskCompletion(p)))
    val storeCompletion =
      Flipped(Decoupled(new BiRaTaskCompletion(p)))
    val completion = Decoupled(new BiRaTaskCompletion(p))

    val status = Output(new BiRaSchedulerStatus)
  })

  private val entries = RegInit(
    VecInit.fill(p.reservationStationEntries)(
      0.U.asTypeOf(new BiRaReservationEntry(p))
    )
  )

  private val validBits = VecInit(entries.map(_.valid)).asUInt
  private val hasFreeEntry = !validBits.andR
  private val freeEntryOH = PriorityEncoderOH(~validBits)
  private val freeEntryIndex =
    OHToUInt(freeEntryOH)(p.reservationIndexBits - 1, 0)

  private val loadCount =
    PopCount(entries.map(e => e.valid && e.taskType === BiRaTaskType.load.U))
  private val execCount =
    PopCount(entries.map(e =>
      e.valid && e.taskType === BiRaTaskType.execute.U
    ))
  private val storeCount =
    PopCount(entries.map(e => e.valid && e.taskType === BiRaTaskType.store.U))

  io.status.loadQueueCount := loadCount
  io.status.execQueueCount := execCount
  io.status.storeQueueCount := storeCount
  io.status.loadBusy :=
    entries.exists(e =>
      e.valid && e.issued && e.taskType === BiRaTaskType.load.U
    )
  io.status.execBusy :=
    entries.exists(e =>
      e.valid && e.issued && e.taskType === BiRaTaskType.execute.U
    )
  io.status.storeBusy :=
    entries.exists(e =>
      e.valid && e.issued && e.taskType === BiRaTaskType.store.U
    )

  // The command frontend currently presents at most one enqueue per cycle.
  // Priority logic keeps the scheduler well-defined if a standalone test
  // drives more than one.
  io.loadEnqueue.ready :=
    hasFreeEntry &&
      loadCount < p.loadQueueEntries.U
  io.execEnqueue.ready :=
    hasFreeEntry &&
      !io.loadEnqueue.valid &&
      execCount < p.execQueueEntries.U
  io.storeEnqueue.ready :=
    hasFreeEntry &&
      !io.loadEnqueue.valid &&
      !io.execEnqueue.valid &&
      storeCount < p.storeQueueEntries.U

  private def contextFor(id: UInt): BiRaContext = {
    val index = id(p.contextIndexBits - 1, 0)
    io.contexts(index)
  }

  /** Conservative mask for every bank touched by a strided linear range. */
  private def bankMask(
    baseRow: UInt,
    rowCount: UInt,
    strideRows: UInt,
    banks: Int
  ): UInt = {
    val start = baseRow.pad(33)
    val last = start +
      (rowCount.pad(33) - 1.U) * strideRows.pad(33)
    VecInit((0 until banks).map { bank =>
      val bankStart = (bank * p.bankRows).U(33.W)
      val bankEnd = ((bank + 1) * p.bankRows - 1).U(33.W)
      rowCount =/= 0.U && start <= bankEnd && last >= bankStart
    }).asUInt
  }

  private def contiguousBankMask(
    baseRow: UInt,
    rowCount: UInt,
    banks: Int
  ): UInt = bankMask(baseRow, rowCount, 1.U, banks)

  private def emptyResources: BiRaResourceSet =
    0.U.asTypeOf(new BiRaResourceSet(p))

  private def dmaResources(
    task: BiRaDmaTask,
    isLoad: Bool
  ): BiRaResourceSet = {
    val resources = WireDefault(emptyResources)
    val context = contextFor(task.contextId)
    val roleBase = context.baseRows(task.role)
    val firstRow = roleBase + task.localRowOffset

    val fullMask =
      bankMask(firstRow, task.rows, task.localStrideRows, p.fullBanks)
    val binaryMask =
      bankMask(
        firstRow,
        task.rows,
        task.localStrideRows,
        p.binaryBanks
      )
    val accumulatorMask =
      bankMask(
        firstRow,
        task.rows,
        task.localStrideRows,
        p.accumulatorBanks
      )

    val binaryMode = context.arrayMode === BiRaArrayMode.binary.U
    val targetsFull =
      task.role === BiRaAddrRole.weightHigh.U ||
        task.role === BiRaAddrRole.residual.U ||
        task.role === BiRaAddrRole.outputFull.U ||
        ((task.role === BiRaAddrRole.input.U ||
          task.role === BiRaAddrRole.weightLow.U) && !binaryMode)
    val targetsBinary =
      task.role === BiRaAddrRole.outputBinary.U ||
        ((task.role === BiRaAddrRole.input.U ||
          task.role === BiRaAddrRole.weightLow.U) && binaryMode)
    val targetsAccumulator =
      task.role === BiRaAddrRole.accumulator.U
    val targetsParameter =
      task.role === BiRaAddrRole.parameter.U ||
        task.role === BiRaAddrRole.correction.U

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

  private def execResources(task: BiRaExecTask): BiRaResourceSet = {
    val resources = WireDefault(emptyResources)
    val context = contextFor(task.contextId)

    val inputPixels =
      context.inputHeight * context.inputWidth
    val outputPixels =
      context.outputHeight * context.outputWidth
    val inputBlocks =
      (context.inputChannels + (p.dim - 1).U) >> p.laneIndexBits
    val outputBlocks =
      (context.outputChannels + (p.dim - 1).U) >> p.laneIndexBits
    val kernelElements =
      context.kernelHeight * context.kernelWidth
    val binaryMode = context.arrayMode === BiRaArrayMode.binary.U
    val columnReduce =
      context.arrayMode === BiRaArrayMode.columnReduce.U
    val depthwise =
      context.arrayMode === BiRaArrayMode.depthwise.U

    val inputRows = Mux(
      columnReduce,
      (inputPixels + 1.U) >> 1,
      inputPixels * inputBlocks
    )
    val outputRows = Mux(
      columnReduce,
      (outputPixels + (p.dim - 1).U) >> p.laneIndexBits,
      Mux(
        context.shufflePack2,
        outputPixels *
          (p.maxShuffleScale * p.maxShuffleScale / 2).U,
        outputPixels * outputBlocks
      )
    )
    val binaryOutputRows = outputPixels * outputBlocks

    val operandsPerGroup = MuxLookup(
      context.weightPrecision,
      1.U
    )(
      Seq(
        BiRaWeightPrecision.w2.U -> 8.U,
        BiRaWeightPrecision.w4.U -> 4.U,
        BiRaWeightPrecision.w8.U -> 2.U,
        BiRaWeightPrecision.w16.U -> 1.U
      )
    )
    val denseInputGroups =
      (context.inputChannels + operandsPerGroup - 1.U) /
        operandsPerGroup
    val inputGroups =
      Mux(depthwise || columnReduce, 1.U, denseInputGroups)
    val multiWeightRows =
      outputBlocks * kernelElements * inputGroups *
        Mux(
          context.weightPrecision === BiRaWeightPrecision.w16.U,
          1.U,
          operandsPerGroup
        )
    val binaryWeightRows =
      context.outputChannels * kernelElements * inputBlocks

    val inputBase = context.baseRows(BiRaAddrRole.input)
    val weightLowBase = context.baseRows(BiRaAddrRole.weightLow)
    val weightHighBase = context.baseRows(BiRaAddrRole.weightHigh)
    val residualBase = context.baseRows(BiRaAddrRole.residual)
    val accumulatorBase = context.baseRows(BiRaAddrRole.accumulator)
    val outputFullBase = context.baseRows(BiRaAddrRole.outputFull)
    val outputBinaryBase =
      context.baseRows(BiRaAddrRole.outputBinary)

    val usesResidual =
      context.postMode === BiRaPostMode.binaryFused.U ||
        context.postMode === BiRaPostMode.finalBilinearResidual.U
    val lowResidualPixels =
      (context.outputHeight / p.maxShuffleScale.U) *
        (context.outputWidth / p.maxShuffleScale.U)
    val residualRows = Mux(
      context.postMode === BiRaPostMode.finalBilinearResidual.U,
      (lowResidualPixels + (p.dim - 1).U) >> p.laneIndexBits,
      outputPixels * outputBlocks
    )

    val multiInputAndLowWeightMask =
      contiguousBankMask(inputBase, inputRows, p.fullBanks) |
        contiguousBankMask(
          weightLowBase,
          multiWeightRows,
          p.fullBanks
        )
    val highWeightMask = Mux(
      context.weightPrecision === BiRaWeightPrecision.w16.U,
      contiguousBankMask(
        weightHighBase,
        multiWeightRows,
        p.fullBanks
      ),
      0.U
    )
    val residualMask = Mux(
      usesResidual,
      contiguousBankMask(residualBase, residualRows, p.fullBanks),
      0.U
    )
    resources.fullRead :=
      Mux(
        binaryMode,
        0.U,
        multiInputAndLowWeightMask | highWeightMask
      ) | residualMask
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

    val accumulatorRows = Mux(
      columnReduce,
      (outputPixels + (p.dim - 1).U) >> p.laneIndexBits,
      outputPixels
    )
    val workingAccumulatorMask =
      contiguousBankMask(
        accumulatorBase,
        accumulatorRows,
        p.accumulatorBanks
      )
    resources.accumulatorRead := workingAccumulatorMask
    resources.accumulatorWrite := workingAccumulatorMask

    // Multi-bit EXEC reads lane parameters. Binary EXEC additionally reads
    // packed per-pixel -N corrections from the same Parameter Buffer.
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
    resources
  }

  private val selectedTaskType = WireDefault(BiRaTaskType.store.U(2.W))
  private val selectedContextId =
    WireDefault(io.storeEnqueue.bits.contextId)
  private val selectedSequence =
    WireDefault(io.storeEnqueue.bits.commandSequence)
  private val selectedDmaTask = WireDefault(io.storeEnqueue.bits)
  private val selectedExecTask = WireDefault(io.execEnqueue.bits)
  private val selectedResources =
    WireDefault(dmaResources(io.storeEnqueue.bits, false.B))

  when(io.loadEnqueue.valid) {
    selectedTaskType := BiRaTaskType.load.U
    selectedContextId := io.loadEnqueue.bits.contextId
    selectedSequence := io.loadEnqueue.bits.commandSequence
    selectedDmaTask := io.loadEnqueue.bits
    selectedResources := dmaResources(io.loadEnqueue.bits, true.B)
  }.elsewhen(io.execEnqueue.valid) {
    selectedTaskType := BiRaTaskType.execute.U
    selectedContextId := io.execEnqueue.bits.contextId
    selectedSequence := io.execEnqueue.bits.commandSequence
    selectedExecTask := io.execEnqueue.bits
    selectedResources := execResources(io.execEnqueue.bits)
  }

  private val enqueueFire =
    io.loadEnqueue.fire || io.execEnqueue.fire || io.storeEnqueue.fire
  when(enqueueFire) {
    entries(freeEntryIndex).valid := true.B
    entries(freeEntryIndex).issued := false.B
    entries(freeEntryIndex).taskType := selectedTaskType
    entries(freeEntryIndex).contextId := selectedContextId
    entries(freeEntryIndex).commandSequence := selectedSequence
    entries(freeEntryIndex).dmaTask := selectedDmaTask
    entries(freeEntryIndex).execTask := selectedExecTask
    entries(freeEntryIndex).resources := selectedResources
  }

  private def hasHazard(
    younger: BiRaResourceSet,
    older: BiRaResourceSet
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

  private def sequenceIsOlder(older: UInt, younger: UInt): Bool = {
    val distance = younger - older
    distance =/= 0.U && !distance(15)
  }

  private val eligible = Wire(Vec(p.reservationStationEntries, Bool()))
  for (candidate <- 0 until p.reservationStationEntries) {
    val blockedByOlder = (0 until p.reservationStationEntries)
      .filter(_ != candidate)
      .map { older =>
        entries(older).valid &&
        sequenceIsOlder(
          entries(older).commandSequence,
          entries(candidate).commandSequence
        ) &&
        hasHazard(
          entries(candidate).resources,
          entries(older).resources
        )
      }
      .reduceOption(_ || _)
      .getOrElse(false.B)
    eligible(candidate) :=
      entries(candidate).valid &&
        !entries(candidate).issued &&
        !blockedByOlder
  }

  private def selectEligible(taskType: Int): (Bool, UInt) = {
    val matches = VecInit((0 until p.reservationStationEntries).map { i =>
      eligible(i) && entries(i).taskType === taskType.U
    })
    val oneHot = PriorityEncoderOH(matches.asUInt)
    (matches.asUInt.orR,
      OHToUInt(oneHot)(p.reservationIndexBits - 1, 0))
  }

  private val (loadIssueValid, loadIssueIndex) =
    selectEligible(BiRaTaskType.load)
  private val (execIssueValid, execIssueIndex) =
    selectEligible(BiRaTaskType.execute)
  private val (storeIssueValid, storeIssueIndex) =
    selectEligible(BiRaTaskType.store)

  io.loadIssue.valid := loadIssueValid
  io.loadIssue.bits := entries(loadIssueIndex).dmaTask
  io.execIssue.valid := execIssueValid
  io.execIssue.bits := entries(execIssueIndex).execTask
  io.storeIssue.valid := storeIssueValid
  io.storeIssue.bits := entries(storeIssueIndex).dmaTask

  when(io.loadIssue.fire) {
    entries(loadIssueIndex).issued := true.B
  }
  when(io.execIssue.fire) {
    entries(execIssueIndex).issued := true.B
  }
  when(io.storeIssue.fire) {
    entries(storeIssueIndex).issued := true.B
  }

  private val completionArbiter =
    Module(new Arbiter(new BiRaTaskCompletion(p), 3))
  completionArbiter.io.in(0) <> io.execCompletion
  completionArbiter.io.in(1) <> io.loadCompletion
  completionArbiter.io.in(2) <> io.storeCompletion
  io.completion <> completionArbiter.io.out

  when(io.completion.fire) {
    val matchingEntries = VecInit(entries.map { entry =>
      entry.valid &&
      entry.contextId === io.completion.bits.contextId &&
      entry.commandSequence === io.completion.bits.commandSequence
    })
    assert(
      PopCount(matchingEntries) === 1.U,
      "completion must match exactly one Reservation Station entry"
    )
    val completedIndex =
      OHToUInt(matchingEntries.asUInt)(
        p.reservationIndexBits - 1,
        0
      )
    assert(entries(completedIndex).issued)
    entries(completedIndex).valid := false.B
    entries(completedIndex).issued := false.B
  }
}
