package bira

// Command decoding and Context management.

import chisel3._
import chisel3.util._

/** Rocket-independent implementation of the frozen BIRA command frontend.
  *
  * CONFIG commands update a small context table in program order. Dynamic
  * LOAD/EXEC/STORE commands are forwarded to three independent queues owned by
  * the future scheduler. FENCE and STATUS return through an rd-tagged response
  * channel. A later LazyRoCC wrapper can connect RoCCCommand/RoCCResponse
  * without changing this module.
  */
class CmdFrontend(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new RawCmd))
    val response = Decoupled(new RawResp)

    val loadTask = Decoupled(new DmaTask(p))
    val execTask = Decoupled(new ExecTask(p))
    val storeTask = Decoupled(new DmaTask(p))
    val completion = Flipped(Decoupled(new Completion(p)))

    val schedulerStatus = Input(new SchedStatus)
    val contexts = Output(Vec(p.nContexts, new Context(p)))

    val tlbFlush = Decoupled(new FlushReq)
    val tlbFlushDone = Flipped(Valid(UInt(8.W)))
  })

  private val contexts =
    RegInit(VecInit.fill(p.nContexts)(0.U.asTypeOf(new Context(p))))
  io.contexts := contexts

  private val nextCommandSequence = RegInit(0.U(16.W))

  private val globalErrorCode = RegInit(0.U(8.W))
  private val globalErrorContext = RegInit(0.U(p.contextIdBits.W))
  private val globalErrorSequence = RegInit(0.U(16.W))

  private val responseValid = RegInit(false.B)
  private val responseRd = RegInit(0.U(5.W))
  private val responseData = RegInit(0.U(64.W))

  io.response.valid := responseValid
  io.response.bits.rd := responseRd
  io.response.bits.data := responseData
  when(io.response.fire) {
    responseValid := false.B
  }

  private val fencePending = RegInit(false.B)
  private val fenceContextScope = RegInit(false.B)
  private val fenceContextId = RegInit(0.U(p.contextIdBits.W))
  private val fenceRd = RegInit(0.U(5.W))

  private val flushPending = RegInit(false.B)
  private val flushIssued = RegInit(false.B)
  private val flushRd = RegInit(0.U(5.W))

  // CFG_COMMIT is deliberately multi-cycle. Capturing a Context, deriving its
  // validation flags, classifying the result, and applying the update in
  // separate cycles prevents a dynamically selected Context from feeding the
  // arithmetic validation tree and another dynamically selected Context's
  // write enable in one path.
  private val commitIdle :: commitDerive :: commitClassify :: commitApply :: Nil =
    Enum(4)
  private val commitState = RegInit(commitIdle)
  private val commitContext = Reg(new Context(p))
  private val commitContextId = Reg(UInt(p.contextIdBits.W))
  private val commitContextIndex = Reg(UInt(p.contextIndexBits.W))
  private val commitContextImplemented = RegInit(false.B)
  private val commitEncodingValid = RegInit(false.B)
  private val commitSequence = Reg(UInt(16.W))

  private val commitContextStateInvalid = RegInit(false.B)
  private val commitConfigStateInvalid = RegInit(false.B)
  private val commitBadShape = RegInit(false.B)
  private val commitMissingAddress = RegInit(false.B)
  private val commitIllegalCombination = RegInit(false.B)
  private val commitHasExistingError = RegInit(false.B)

  private val commitNoAction :: commitGlobalError :: commitContextError :: commitContextReady :: Nil =
    Enum(4)
  private val commitAction = RegInit(commitNoAction)
  private val commitErrorCode = RegInit(ErrorCode.none.U(8.W))

  private val contextId = io.command.bits.rs1(p.contextIdBits - 1, 0)
  private val contextIndex =
    contextId(p.contextIndexBits - 1, 0)
  private val contextImplemented = contextId < p.nContexts.U
  private val selectedContext = contexts(contextIndex)

  private val dmaContextId =
    io.command.bits.rs2(p.contextIdBits - 1, 0)
  private val dmaContextIndex =
    dmaContextId(p.contextIndexBits - 1, 0)
  private val dmaContextImplemented = dmaContextId < p.nContexts.U
  private val dmaSelectedContext = contexts(dmaContextIndex)
  private val dmaRole = io.command.bits.rs2(6, 3)

  private val totalInflight =
    contexts.map(_.inflightCount).reduce(_ +& _)
  private val anyInflight = totalInflight =/= 0.U

  private def recordGlobalError(
    code: UInt,
    ctx: UInt,
    sequence: UInt
  ): Unit = {
    when(globalErrorCode === ErrorCode.none.U) {
      globalErrorCode := code
      globalErrorContext := ctx
      globalErrorSequence := sequence
    }
  }

  private def recordContextError(
    ctx: UInt,
    code: UInt,
    sequence: UInt
  ): Unit = {
    val index = ctx(p.contextIndexBits - 1, 0)
    when(ctx < p.nContexts.U) {
      when(contexts(index).errorCode === ErrorCode.none.U) {
        contexts(index).errorCode := code
        contexts(index).errorCommandSequence := sequence
      }
      recordGlobalError(code, ctx, sequence)
    }.otherwise {
      recordGlobalError(
        ErrorCode.invalidContext.U,
        ctx,
        sequence
      )
    }
  }

  private def dmaTaskBits: DmaTask = {
    val task = Wire(new DmaTask(p))
    task.contextId := dmaContextId
    task.role := dmaRole
    task.dramVirtualAddress := io.command.bits.rs1
    task.localRowOffset := io.command.bits.rs2(20, 7)
    task.rows := io.command.bits.rs2(34, 21)
    task.bytesPerRow := io.command.bits.rs2(41, 35)
    task.dramStrideBytes := io.command.bits.rs2(57, 42)
    task.localStrideRows := io.command.bits.rs2(63, 58)
    task.commandSequence := nextCommandSequence
    task.translationStatus := io.command.bits.translationStatus
    task
  }

  io.loadTask.valid := false.B
  io.loadTask.bits := dmaTaskBits
  io.storeTask.valid := false.B
  io.storeTask.bits := dmaTaskBits
  io.execTask.valid := false.B
  io.execTask.bits.contextId := contextId
  io.execTask.bits.commandSequence := nextCommandSequence

  io.completion.ready := true.B

  io.tlbFlush.valid :=
    flushPending && !flushIssued && !anyInflight
  io.tlbFlush.bits.all := true.B
  when(io.tlbFlush.fire) {
    flushIssued := true.B
  }

  private val frontendBlocked =
    responseValid || fencePending || flushPending || commitState =/= commitIdle
  io.command.ready := false.B

  private val acceptedDynamicTask = WireInit(false.B)
  private val acceptedDynamicContext =
    WireInit(0.U(p.contextIdBits.W))

  private val commandFunct = io.command.bits.funct
  private val isLoad = commandFunct === Funct.load2d.U
  private val isExec = commandFunct === Funct.execConv.U
  private val isStore = commandFunct === Funct.store2d.U
  private val knownFunct = Seq(
    Funct.cfgShape,
    Funct.cfgAddr,
    Funct.cfgMode,
    Funct.cfgCommit,
    Funct.load2d,
    Funct.execConv,
    Funct.store2d,
    Funct.fence,
    Funct.status,
    Funct.tlbFlush
  ).map(value => commandFunct === value.U).reduce(_ || _)

  private val dmaContextUsable =
    dmaContextImplemented &&
      dmaSelectedContext.ready &&
      dmaSelectedContext.committed &&
      dmaSelectedContext.errorCode === ErrorCode.none.U
  private val dmaDescriptorValid =
    io.command.bits.rs2(34, 21) =/= 0.U &&
      io.command.bits.rs2(41, 35) =/= 0.U &&
      io.command.bits.rs2(57, 42) >=
        io.command.bits.rs2(41, 35) &&
      io.command.bits.rs2(63, 58) =/= 0.U
  private val dmaRoleConfigured =
    dmaRole < AddrRole.count.U &&
      dmaSelectedContext.addressValid(dmaRole)
  private val dmaTaskUsable =
    dmaContextUsable &&
      dmaRoleConfigured &&
      dmaDescriptorValid &&
      !io.command.bits.xd &&
      io.command.bits.xs1 &&
      io.command.bits.xs2
  private val execEncodingValid =
    !io.command.bits.xd &&
      io.command.bits.xs1 &&
      io.command.bits.xs2 &&
      io.command.bits.rs1(63, p.contextIdBits) === 0.U &&
      io.command.bits.rs2 === 0.U
  private val execContextUsable =
    contextImplemented &&
      selectedContext.ready &&
      selectedContext.committed &&
      selectedContext.errorCode === ErrorCode.none.U &&
      execEncodingValid

  when(!frontendBlocked) {
    // Unknown commands are accepted so that they cannot deadlock the scalar
    // pipeline; the error is reported through STATUS/FENCE below.
    io.command.ready := true.B
    switch(commandFunct) {
      is(Funct.load2d.U) {
        when(dmaTaskUsable) {
          io.loadTask.valid := io.command.valid
          io.command.ready := io.loadTask.ready
        }.otherwise {
          io.command.ready := true.B
        }
      }

      is(Funct.execConv.U) {
        when(execContextUsable) {
          io.execTask.valid := io.command.valid
          io.command.ready := io.execTask.ready
        }.otherwise {
          io.command.ready := true.B
        }
      }

      is(Funct.store2d.U) {
        when(dmaTaskUsable) {
          io.storeTask.valid := io.command.valid
          io.command.ready := io.storeTask.ready
        }.otherwise {
          io.command.ready := true.B
        }
      }

      is(Funct.cfgShape.U) { io.command.ready := true.B }
      is(Funct.cfgAddr.U) { io.command.ready := true.B }
      is(Funct.cfgMode.U) { io.command.ready := true.B }
      is(Funct.cfgCommit.U) { io.command.ready := true.B }
      is(Funct.fence.U) { io.command.ready := true.B }
      is(Funct.status.U) { io.command.ready := true.B }
      is(Funct.tlbFlush.U) { io.command.ready := true.B }
    }
  }

  when(io.loadTask.fire) {
    acceptedDynamicTask := true.B
    acceptedDynamicContext := dmaContextId
  }
  when(io.execTask.fire) {
    acceptedDynamicTask := true.B
    acceptedDynamicContext := contextId
  }
  when(io.storeTask.fire) {
    acceptedDynamicTask := true.B
    acceptedDynamicContext := dmaContextId
  }

  when(acceptedDynamicTask) {
    nextCommandSequence := nextCommandSequence + 1.U
  }

  // Increment and completion decrement can occur for the same Context in the
  // same cycle. Compute the net update explicitly so neither event is lost.
  for (i <- 0 until p.nContexts) {
    val increment =
      acceptedDynamicTask && acceptedDynamicContext === i.U
    val decrement =
      io.completion.fire &&
        io.completion.bits.contextId === i.U &&
        contexts(i).inflightCount =/= 0.U

    when(increment && !decrement) {
      contexts(i).inflightCount := contexts(i).inflightCount + 1.U
    }.elsewhen(!increment && decrement) {
      contexts(i).inflightCount := contexts(i).inflightCount - 1.U
    }
  }

  when(io.completion.fire) {
    val completionContextIndex =
      io.completion.bits.contextId(p.contextIndexBits - 1, 0)
    when(io.completion.bits.contextId < p.nContexts.U) {
      when(
        io.completion.bits.errorCode =/= ErrorCode.none.U
      ) {
        recordContextError(
          io.completion.bits.contextId,
          io.completion.bits.errorCode,
          io.completion.bits.commandSequence
        )
      }
      assert(
        contexts(completionContextIndex).inflightCount =/= 0.U,
        "completion received for a Context with no inflight command"
      )
    }.otherwise {
      recordGlobalError(
        ErrorCode.invalidContext.U,
        io.completion.bits.contextId,
        io.completion.bits.commandSequence
      )
    }
  }

  private def cfgEncodingValid(
    reservedIsZero: Bool
  ): Bool = {
    io.command.bits.xs1 &&
    io.command.bits.xs2 &&
    !io.command.bits.xd &&
    reservedIsZero
  }

  when(io.command.fire) {
    switch(commandFunct) {
      is(Funct.cfgShape.U) {
        val encodingValid = cfgEncodingValid(
          io.command.bits.rs1(63, 51) === 0.U &&
            io.command.bits.rs2(63, 40) === 0.U
        )
        when(!contextImplemented) {
          recordGlobalError(
            ErrorCode.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!encodingValid) {
          recordContextError(
            contextId,
            ErrorCode.badEnum.U,
            nextCommandSequence
          )
        }.elsewhen(selectedContext.inflightCount =/= 0.U) {
          recordContextError(
            contextId,
            ErrorCode.contextBusy.U,
            nextCommandSequence
          )
        }.otherwise {
          contexts(contextIndex) :=
            0.U.asTypeOf(new Context(p))
          contexts(contextIndex).building := true.B
          contexts(contextIndex).shapeValid := true.B
          contexts(contextIndex).inputHeight :=
            io.command.bits.rs1(14, 3)
          contexts(contextIndex).inputWidth :=
            io.command.bits.rs1(26, 15)
          contexts(contextIndex).inputChannels :=
            io.command.bits.rs1(38, 27)
          contexts(contextIndex).outputHeight :=
            io.command.bits.rs1(50, 39)
          contexts(contextIndex).outputWidth :=
            io.command.bits.rs2(11, 0)
          contexts(contextIndex).outputChannels :=
            io.command.bits.rs2(23, 12)
          contexts(contextIndex).kernelHeight :=
            io.command.bits.rs2(27, 24)
          contexts(contextIndex).kernelWidth :=
            io.command.bits.rs2(31, 28)
          contexts(contextIndex).paddingHeight :=
            io.command.bits.rs2(35, 32)
          contexts(contextIndex).paddingWidth :=
            io.command.bits.rs2(39, 36)
        }
      }

      is(Funct.cfgAddr.U) {
        val role = io.command.bits.rs1(6, 3)
        val encodingValid = cfgEncodingValid(
          io.command.bits.rs1(63, 23) === 0.U &&
            io.command.bits.rs2 === 0.U
        )
        when(!contextImplemented) {
          recordGlobalError(
            ErrorCode.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!encodingValid) {
          recordContextError(
            contextId,
            ErrorCode.badEnum.U,
            nextCommandSequence
          )
        }.elsewhen(
          !selectedContext.building ||
            selectedContext.inflightCount =/= 0.U
        ) {
          recordContextError(
            contextId,
            ErrorCode.contextState.U,
            nextCommandSequence
          )
        }.elsewhen(role >= AddrRole.count.U) {
          recordContextError(
            contextId,
            ErrorCode.badRole.U,
            nextCommandSequence
          )
        }.otherwise {
          contexts(contextIndex).baseRows(role) :=
            io.command.bits.rs1(22, 7)
          contexts(contextIndex).addressValid :=
            selectedContext.addressValid | (1.U << role)
        }
      }

      is(Funct.cfgMode.U) {
        val postMode = io.command.bits.rs1(10, 8)
        val encodingValid = cfgEncodingValid(
          io.command.bits.rs1(63, 14) === 0.U &&
            io.command.bits.rs2 === 0.U
        )
        when(!contextImplemented) {
          recordGlobalError(
            ErrorCode.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!encodingValid) {
          recordContextError(
            contextId,
            ErrorCode.badEnum.U,
            nextCommandSequence
          )
        }.elsewhen(
          !selectedContext.building ||
            selectedContext.inflightCount =/= 0.U
        ) {
          recordContextError(
            contextId,
            ErrorCode.contextState.U,
            nextCommandSequence
          )
        }.elsewhen(postMode > PostMode.finalBilinearResidual.U) {
          recordContextError(
            contextId,
            ErrorCode.badEnum.U,
            nextCommandSequence
          )
        }.otherwise {
          contexts(contextIndex).modeValid := true.B
          contexts(contextIndex).arrayMode :=
            io.command.bits.rs1(4, 3)
          contexts(contextIndex).weightPrecision :=
            io.command.bits.rs1(6, 5)
          contexts(contextIndex).inputSigned :=
            io.command.bits.rs1(7)
          contexts(contextIndex).postMode := postMode
          contexts(contextIndex).shufflePack2 :=
            io.command.bits.rs1(11)
          contexts(contextIndex).writeFull :=
            io.command.bits.rs1(12)
          contexts(contextIndex).writeBinary :=
            io.command.bits.rs1(13)
        }
      }

      is(Funct.cfgCommit.U) {
        val encodingValid = cfgEncodingValid(
          io.command.bits.rs1(63, p.contextIdBits) === 0.U &&
            io.command.bits.rs2 === 0.U
        )
        commitContext := selectedContext
        commitContextId := contextId
        commitContextIndex := contextIndex
        commitContextImplemented := contextImplemented
        commitEncodingValid := encodingValid
        commitSequence := nextCommandSequence
        commitState := commitDerive
      }

      is(Funct.load2d.U) {
        when(
          io.command.bits.xd ||
            !io.command.bits.xs1 ||
            !io.command.bits.xs2
        ) {
          recordContextError(
            dmaContextId,
            ErrorCode.badEnum.U,
            nextCommandSequence
          )
          when(io.command.bits.xd) {
            responseValid := true.B
            responseRd := io.command.bits.rd
            responseData := ErrorCode.badEnum.U
          }
        }.elsewhen(!dmaContextImplemented) {
          recordGlobalError(
            ErrorCode.invalidContext.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(dmaRole >= AddrRole.count.U) {
          recordContextError(
            dmaContextId,
            ErrorCode.badRole.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.addressValid(dmaRole)) {
          recordContextError(
            dmaContextId,
            ErrorCode.missingAddress.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.ready) {
          recordContextError(
            dmaContextId,
            ErrorCode.contextNotReady.U,
            nextCommandSequence
          )
        }.elsewhen(
          dmaSelectedContext.errorCode =/= ErrorCode.none.U
        ) {
          recordGlobalError(
            ErrorCode.contextFailed.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(34, 21) === 0.U ||
            io.command.bits.rs2(41, 35) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            ErrorCode.zeroSize.U,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(57, 42) <
            io.command.bits.rs2(41, 35) ||
            io.command.bits.rs2(63, 58) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            ErrorCode.badStride.U,
            nextCommandSequence
          )
        }
      }

      is(Funct.execConv.U) {
        when(!execEncodingValid) {
          recordContextError(
            contextId,
            ErrorCode.badEnum.U,
            nextCommandSequence
          )
          when(io.command.bits.xd) {
            responseValid := true.B
            responseRd := io.command.bits.rd
            responseData := ErrorCode.badEnum.U
          }
        }.elsewhen(!contextImplemented) {
          recordGlobalError(
            ErrorCode.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!selectedContext.ready) {
          recordContextError(
            contextId,
            ErrorCode.contextNotReady.U,
            nextCommandSequence
          )
        }.elsewhen(
          selectedContext.errorCode =/= ErrorCode.none.U
        ) {
          recordGlobalError(
            ErrorCode.contextFailed.U,
            contextId,
            nextCommandSequence
          )
        }
      }

      is(Funct.store2d.U) {
        when(
          io.command.bits.xd ||
            !io.command.bits.xs1 ||
            !io.command.bits.xs2
        ) {
          recordContextError(
            dmaContextId,
            ErrorCode.badEnum.U,
            nextCommandSequence
          )
          when(io.command.bits.xd) {
            responseValid := true.B
            responseRd := io.command.bits.rd
            responseData := ErrorCode.badEnum.U
          }
        }.elsewhen(!dmaContextImplemented) {
          recordGlobalError(
            ErrorCode.invalidContext.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(dmaRole >= AddrRole.count.U) {
          recordContextError(
            dmaContextId,
            ErrorCode.badRole.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.addressValid(dmaRole)) {
          recordContextError(
            dmaContextId,
            ErrorCode.missingAddress.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.ready) {
          recordContextError(
            dmaContextId,
            ErrorCode.contextNotReady.U,
            nextCommandSequence
          )
        }.elsewhen(
          dmaSelectedContext.errorCode =/= ErrorCode.none.U
        ) {
          recordGlobalError(
            ErrorCode.contextFailed.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(34, 21) === 0.U ||
            io.command.bits.rs2(41, 35) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            ErrorCode.zeroSize.U,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(57, 42) <
            io.command.bits.rs2(41, 35) ||
            io.command.bits.rs2(63, 58) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            ErrorCode.badStride.U,
            nextCommandSequence
          )
        }
      }

      is(Funct.fence.U) {
        val encodingValid =
          io.command.bits.xd &&
            io.command.bits.xs1 &&
            io.command.bits.xs2 &&
            io.command.bits.rs1(63, 4) === 0.U &&
            io.command.bits.rs2 === 0.U
        when(!encodingValid) {
          responseValid := true.B
          responseRd := io.command.bits.rd
          responseData :=
            (ErrorCode.badEnum.U(8.W) << 1)
        }.otherwise {
          fencePending := true.B
          fenceContextScope := io.command.bits.rs1(0)
          fenceContextId :=
            io.command.bits.rs1(p.contextIdBits, 1)
          fenceRd := io.command.bits.rd
        }
      }

      is(Funct.status.U) {
        val statusContextScope = io.command.bits.rs1(0)
        val statusContextId =
          io.command.bits.rs1(p.contextIdBits, 1)
        val statusContextIndex =
          statusContextId(p.contextIndexBits - 1, 0)
        val clearError = io.command.bits.rs1(4)
        val encodingValid =
          io.command.bits.xd &&
            io.command.bits.xs1 &&
            io.command.bits.xs2 &&
            io.command.bits.rs1(63, 5) === 0.U &&
            io.command.bits.rs2 === 0.U

        responseValid := true.B
        responseRd := io.command.bits.rd
        responseData := 0.U

        when(!encodingValid) {
          responseData :=
            (ErrorCode.badEnum.U(8.W) << 13) | (1.U << 12)
        }.elsewhen(statusContextScope) {
          when(statusContextId < p.nContexts.U) {
            val ctx = contexts(statusContextIndex)
            val clearBlocked = clearError && ctx.inflightCount =/= 0.U
            val reportedError = Mux(
              clearBlocked,
              ErrorCode.contextBusy.U,
              ctx.errorCode
            )
            responseData :=
              1.U |
                (ctx.building.asUInt << 1) |
                (ctx.ready.asUInt << 2) |
                (ctx.committed.asUInt << 3) |
                (ctx.inflightCount << 4) |
                ((reportedError =/= ErrorCode.none.U).asUInt << 12) |
                (reportedError << 13) |
                (ctx.errorCommandSequence << 21)
            when(clearError && !clearBlocked) {
              contexts(statusContextIndex).errorCode :=
                ErrorCode.none.U
              contexts(statusContextIndex).errorCommandSequence := 0.U
            }
          }.otherwise {
            responseData :=
              (ErrorCode.invalidContext.U(8.W) << 13) |
                (1.U << 12)
          }
        }.otherwise {
          val clearBlocked = clearError && anyInflight
          val reportedError = Mux(
            clearBlocked,
            ErrorCode.contextBusy.U,
            globalErrorCode
          )
          responseData :=
            (io.schedulerStatus.loadQueueCount.orR.asUInt << 0) |
              (io.schedulerStatus.execQueueCount.orR.asUInt << 1) |
              (io.schedulerStatus.storeQueueCount.orR.asUInt << 2) |
              (io.schedulerStatus.loadBusy.asUInt << 3) |
              (io.schedulerStatus.execBusy.asUInt << 4) |
              (io.schedulerStatus.storeBusy.asUInt << 5) |
              ((reportedError =/= ErrorCode.none.U).asUInt << 6) |
              (reportedError << 7) |
              (globalErrorContext << 15) |
              (io.schedulerStatus.loadQueueCount << 18) |
              (io.schedulerStatus.execQueueCount << 26) |
              (io.schedulerStatus.storeQueueCount << 34)
          when(clearError && !clearBlocked) {
            globalErrorCode := ErrorCode.none.U
            globalErrorContext := 0.U
            globalErrorSequence := 0.U
            for (i <- 0 until p.nContexts) {
              contexts(i).errorCode := ErrorCode.none.U
              contexts(i).errorCommandSequence := 0.U
            }
          }
        }
      }

      is(Funct.tlbFlush.U) {
        val encodingValid =
          io.command.bits.xd &&
            io.command.bits.xs1 &&
            io.command.bits.xs2 &&
            io.command.bits.rs1 === 0.U &&
            io.command.bits.rs2 === 0.U
        when(!encodingValid) {
          responseValid := true.B
          responseRd := io.command.bits.rd
          responseData := ErrorCode.badEnum.U
        }.otherwise {
          flushPending := true.B
          flushIssued := false.B
          flushRd := io.command.bits.rd
        }
      }

    }
  }

  switch(commitState) {
    is(commitDerive) {
      val inputExtentHeight =
        commitContext.inputHeight + (commitContext.paddingHeight << 1)
      val inputExtentWidth =
        commitContext.inputWidth + (commitContext.paddingWidth << 1)
      val expectedOutputHeight =
        inputExtentHeight - commitContext.kernelHeight + 1.U
      val expectedOutputWidth =
        inputExtentWidth - commitContext.kernelWidth + 1.U

      val shapeNonZero =
        commitContext.inputHeight =/= 0.U &&
          commitContext.inputWidth =/= 0.U &&
          commitContext.inputChannels =/= 0.U &&
          commitContext.outputHeight =/= 0.U &&
          commitContext.outputWidth =/= 0.U &&
          commitContext.outputChannels =/= 0.U &&
          commitContext.kernelHeight =/= 0.U &&
          commitContext.kernelWidth =/= 0.U
      val shapeWithinLimits =
        commitContext.inputHeight <= p.maxImageHeight.U &&
          commitContext.inputWidth <= p.maxImageWidth.U &&
          commitContext.outputHeight <= p.maxImageHeight.U &&
          commitContext.outputWidth <= p.maxImageWidth.U &&
          commitContext.inputChannels <= p.maxInputChannels.U &&
          commitContext.outputChannels <= (p.maxOutputBlocks * p.dim).U &&
          commitContext.kernelHeight <= p.kernelSize.U &&
          commitContext.kernelWidth <= p.kernelSize.U
      val convolutionShapeMatches =
        inputExtentHeight >= commitContext.kernelHeight &&
          inputExtentWidth >= commitContext.kernelWidth &&
          commitContext.outputHeight === expectedOutputHeight &&
          commitContext.outputWidth === expectedOutputWidth

      val addressValid = commitContext.addressValid
      val binaryMode = commitContext.arrayMode === ArrayMode.binary.U
      val columnReduce =
        commitContext.arrayMode === ArrayMode.columnReduce.U
      val depthwise = commitContext.arrayMode === ArrayMode.depthwise.U
      val requiresWeightHigh =
        !binaryMode && commitContext.weightPrecision === WgtPrecision.w16.U
      val requiresResidual =
        commitContext.postMode === PostMode.binaryFused.U ||
          commitContext.postMode === PostMode.finalBilinearResidual.U
      val requiresParameter =
        !binaryMode || commitContext.postMode =/= PostMode.none.U
      val requiredAddressesPresent =
        addressValid(AddrRole.input) &&
          addressValid(AddrRole.weightLow) &&
          addressValid(AddrRole.accumulator) &&
          Mux(requiresParameter, addressValid(AddrRole.parameter), true.B) &&
          Mux(requiresWeightHigh, addressValid(AddrRole.weightHigh), true.B) &&
          Mux(binaryMode, addressValid(AddrRole.correction), true.B) &&
          Mux(requiresResidual, addressValid(AddrRole.residual), true.B) &&
          Mux(commitContext.writeFull, addressValid(AddrRole.outputFull), true.B) &&
          Mux(
            commitContext.writeBinary,
            addressValid(AddrRole.outputBinary),
            true.B
          )

      val outputSelectionValid =
        commitContext.writeFull ||
          commitContext.writeBinary ||
          commitContext.postMode === PostMode.none.U
      val depthwiseValid =
        !depthwise ||
          commitContext.inputChannels === commitContext.outputChannels
      val binaryValid =
        !binaryMode || (commitContext.inputChannels % p.dim.U === 0.U)
      val columnReduceValid =
        !columnReduce ||
          (commitContext.inputChannels <= (p.dim / 2).U &&
            commitContext.outputChannels === 1.U)
      val shuffleOutputChannels =
        (p.dim / 2) * p.maxShuffleScale * p.maxShuffleScale
      val shuffleValid =
        !commitContext.shufflePack2 ||
          (commitContext.arrayMode === ArrayMode.dense.U &&
            commitContext.postMode === PostMode.intPrelu.U &&
            commitContext.outputChannels === shuffleOutputChannels.U &&
            commitContext.writeFull &&
            !commitContext.writeBinary)
      val binaryPostValid =
        commitContext.postMode =/= PostMode.binaryFused.U || binaryMode
      val finalPostValid =
        commitContext.postMode =/= PostMode.finalBilinearResidual.U ||
          (columnReduce &&
            commitContext.outputChannels === 1.U &&
            commitContext.outputHeight % p.maxShuffleScale.U === 0.U &&
            commitContext.outputWidth % p.maxShuffleScale.U === 0.U)

      commitContextStateInvalid :=
        !commitContext.building || commitContext.inflightCount =/= 0.U
      commitConfigStateInvalid :=
        !commitContext.shapeValid || !commitContext.modeValid
      commitBadShape :=
        !shapeNonZero || !shapeWithinLimits || !convolutionShapeMatches
      commitMissingAddress := !requiredAddressesPresent
      commitIllegalCombination :=
        !outputSelectionValid ||
          !depthwiseValid ||
          !binaryValid ||
          !columnReduceValid ||
          !shuffleValid ||
          !binaryPostValid ||
          !finalPostValid
      commitHasExistingError :=
        commitContext.errorCode =/= ErrorCode.none.U
      commitState := commitClassify
    }

    is(commitClassify) {
      commitAction := commitNoAction
      commitErrorCode := ErrorCode.none.U
      when(!commitContextImplemented) {
        commitAction := commitGlobalError
        commitErrorCode := ErrorCode.invalidContext.U
      }.elsewhen(!commitEncodingValid) {
        commitAction := commitContextError
        commitErrorCode := ErrorCode.badEnum.U
      }.elsewhen(commitContextStateInvalid || commitConfigStateInvalid) {
        commitAction := commitContextError
        commitErrorCode := ErrorCode.contextState.U
      }.elsewhen(commitBadShape) {
        commitAction := commitContextError
        commitErrorCode := ErrorCode.badShape.U
      }.elsewhen(commitMissingAddress) {
        commitAction := commitContextError
        commitErrorCode := ErrorCode.missingAddress.U
      }.elsewhen(commitIllegalCombination) {
        commitAction := commitContextError
        commitErrorCode := ErrorCode.illegalCombination.U
      }.elsewhen(!commitHasExistingError) {
        commitAction := commitContextReady
      }
      commitState := commitApply
    }

    is(commitApply) {
      when(commitAction === commitGlobalError) {
        recordGlobalError(
          commitErrorCode,
          commitContextId,
          commitSequence
        )
      }.elsewhen(commitAction === commitContextError) {
        recordContextError(
          commitContextId,
          commitErrorCode,
          commitSequence
        )
      }.elsewhen(commitAction === commitContextReady) {
        contexts(commitContextIndex).building := false.B
        contexts(commitContextIndex).ready := true.B
        contexts(commitContextIndex).committed := true.B
      }
      commitState := commitIdle
    }
  }

  when(io.command.fire && !knownFunct) {
    recordGlobalError(
      ErrorCode.badEnum.U,
      0.U,
      nextCommandSequence
    )
    when(io.command.bits.xd) {
      responseValid := true.B
      responseRd := io.command.bits.rd
      responseData := ErrorCode.badEnum.U
    }
  }

  private val fenceContextImplemented =
    fenceContextId < p.nContexts.U
  private val fenceContextIndex =
    fenceContextId(p.contextIndexBits - 1, 0)
  private val fenceContextDone =
    !fenceContextImplemented ||
      contexts(fenceContextIndex).inflightCount === 0.U
  private val fenceDone =
    Mux(fenceContextScope, fenceContextDone, !anyInflight)

  when(fencePending && fenceDone && !responseValid) {
    val contextErrorCode = Mux(
      fenceContextImplemented,
      contexts(fenceContextIndex).errorCode,
      ErrorCode.invalidContext.U
    )
    val contextErrorSequence = Mux(
      fenceContextImplemented,
      contexts(fenceContextIndex).errorCommandSequence,
      0.U
    )
    val errorCode = Mux(
      fenceContextScope,
      contextErrorCode,
      globalErrorCode
    )
    val errorContext = Mux(
      fenceContextScope,
      fenceContextId,
      globalErrorContext
    )
    val errorSequence = Mux(
      fenceContextScope,
      contextErrorSequence,
      globalErrorSequence
    )
    responseValid := true.B
    responseRd := fenceRd
    responseData :=
      (errorCode === ErrorCode.none.U).asUInt |
        (errorCode << 1) |
        (errorContext << 9) |
        (errorSequence << 12)
    fencePending := false.B
  }

  when(
    flushPending &&
      (flushIssued || io.tlbFlush.fire) &&
      io.tlbFlushDone.valid
  ) {
    responseValid := true.B
    responseRd := flushRd
    responseData := io.tlbFlushDone.bits
    flushPending := false.B
    flushIssued := false.B
    when(io.tlbFlushDone.bits =/= ErrorCode.none.U) {
      recordGlobalError(
        io.tlbFlushDone.bits,
        0.U,
        nextCommandSequence
      )
    }
  }
}
