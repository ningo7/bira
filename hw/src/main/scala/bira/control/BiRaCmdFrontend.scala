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
class BiRaCmdFrontend(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new BiRaRawCommand))
    val response = Decoupled(new BiRaRawResponse)

    val loadTask = Decoupled(new BiRaDmaTask(p))
    val execTask = Decoupled(new BiRaExecTask(p))
    val storeTask = Decoupled(new BiRaDmaTask(p))
    val completion = Flipped(Decoupled(new BiRaTaskCompletion(p)))

    val schedulerStatus = Input(new BiRaSchedulerStatus)
    val contexts = Output(Vec(p.nContexts, new BiRaContext(p)))

    val tlbFlush = Decoupled(new BiRaTlbFlushRequest)
    val tlbFlushDone = Flipped(Valid(UInt(8.W)))
  })

  private val contexts =
    RegInit(VecInit.fill(p.nContexts)(0.U.asTypeOf(new BiRaContext(p))))
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
    when(globalErrorCode === BiRaError.none.U) {
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
      when(contexts(index).errorCode === BiRaError.none.U) {
        contexts(index).errorCode := code
        contexts(index).errorCommandSequence := sequence
      }
      recordGlobalError(code, ctx, sequence)
    }.otherwise {
      recordGlobalError(
        BiRaError.invalidContext.U,
        ctx,
        sequence
      )
    }
  }

  private def dmaTaskBits: BiRaDmaTask = {
    val task = Wire(new BiRaDmaTask(p))
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
    responseValid || fencePending || flushPending
  io.command.ready := false.B

  private val acceptedDynamicTask = WireInit(false.B)
  private val acceptedDynamicContext =
    WireInit(0.U(p.contextIdBits.W))

  private val commandFunct = io.command.bits.funct
  private val isLoad = commandFunct === BiRaFunct.load2d.U
  private val isExec = commandFunct === BiRaFunct.execConv.U
  private val isStore = commandFunct === BiRaFunct.store2d.U
  private val knownFunct = Seq(
    BiRaFunct.cfgShape,
    BiRaFunct.cfgAddr,
    BiRaFunct.cfgMode,
    BiRaFunct.cfgCommit,
    BiRaFunct.load2d,
    BiRaFunct.execConv,
    BiRaFunct.store2d,
    BiRaFunct.fence,
    BiRaFunct.status,
    BiRaFunct.tlbFlush
  ).map(value => commandFunct === value.U).reduce(_ || _)

  private val dmaContextUsable =
    dmaContextImplemented &&
      dmaSelectedContext.ready &&
      dmaSelectedContext.committed &&
      dmaSelectedContext.errorCode === BiRaError.none.U
  private val dmaDescriptorValid =
    io.command.bits.rs2(34, 21) =/= 0.U &&
      io.command.bits.rs2(41, 35) =/= 0.U &&
      io.command.bits.rs2(57, 42) >=
        io.command.bits.rs2(41, 35) &&
      io.command.bits.rs2(63, 58) =/= 0.U
  private val dmaRoleConfigured =
    dmaRole < BiRaAddrRole.count.U &&
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
      selectedContext.errorCode === BiRaError.none.U &&
      execEncodingValid

  when(!frontendBlocked) {
    // Unknown commands are accepted so that they cannot deadlock the scalar
    // pipeline; the error is reported through STATUS/FENCE below.
    io.command.ready := true.B
    switch(commandFunct) {
      is(BiRaFunct.load2d.U) {
        when(dmaTaskUsable) {
          io.loadTask.valid := io.command.valid
          io.command.ready := io.loadTask.ready
        }.otherwise {
          io.command.ready := true.B
        }
      }

      is(BiRaFunct.execConv.U) {
        when(execContextUsable) {
          io.execTask.valid := io.command.valid
          io.command.ready := io.execTask.ready
        }.otherwise {
          io.command.ready := true.B
        }
      }

      is(BiRaFunct.store2d.U) {
        when(dmaTaskUsable) {
          io.storeTask.valid := io.command.valid
          io.command.ready := io.storeTask.ready
        }.otherwise {
          io.command.ready := true.B
        }
      }

      is(BiRaFunct.cfgShape.U) { io.command.ready := true.B }
      is(BiRaFunct.cfgAddr.U) { io.command.ready := true.B }
      is(BiRaFunct.cfgMode.U) { io.command.ready := true.B }
      is(BiRaFunct.cfgCommit.U) { io.command.ready := true.B }
      is(BiRaFunct.fence.U) { io.command.ready := true.B }
      is(BiRaFunct.status.U) { io.command.ready := true.B }
      is(BiRaFunct.tlbFlush.U) { io.command.ready := true.B }
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
        io.completion.bits.errorCode =/= BiRaError.none.U
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
        BiRaError.invalidContext.U,
        io.completion.bits.contextId,
        io.completion.bits.commandSequence
      )
    }
  }

  private def configurationEncodingIsValid(
    reservedIsZero: Bool
  ): Bool = {
    io.command.bits.xs1 &&
    io.command.bits.xs2 &&
    !io.command.bits.xd &&
    reservedIsZero
  }

  when(io.command.fire) {
    switch(commandFunct) {
      is(BiRaFunct.cfgShape.U) {
        val encodingValid = configurationEncodingIsValid(
          io.command.bits.rs1(63, 51) === 0.U &&
            io.command.bits.rs2(63, 40) === 0.U
        )
        when(!contextImplemented) {
          recordGlobalError(
            BiRaError.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!encodingValid) {
          recordContextError(
            contextId,
            BiRaError.badEnum.U,
            nextCommandSequence
          )
        }.elsewhen(selectedContext.inflightCount =/= 0.U) {
          recordContextError(
            contextId,
            BiRaError.contextBusy.U,
            nextCommandSequence
          )
        }.otherwise {
          contexts(contextIndex) :=
            0.U.asTypeOf(new BiRaContext(p))
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

      is(BiRaFunct.cfgAddr.U) {
        val role = io.command.bits.rs1(6, 3)
        val encodingValid = configurationEncodingIsValid(
          io.command.bits.rs1(63, 23) === 0.U &&
            io.command.bits.rs2 === 0.U
        )
        when(!contextImplemented) {
          recordGlobalError(
            BiRaError.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!encodingValid) {
          recordContextError(
            contextId,
            BiRaError.badEnum.U,
            nextCommandSequence
          )
        }.elsewhen(
          !selectedContext.building ||
            selectedContext.inflightCount =/= 0.U
        ) {
          recordContextError(
            contextId,
            BiRaError.contextState.U,
            nextCommandSequence
          )
        }.elsewhen(role >= BiRaAddrRole.count.U) {
          recordContextError(
            contextId,
            BiRaError.badRole.U,
            nextCommandSequence
          )
        }.otherwise {
          contexts(contextIndex).baseRows(role) :=
            io.command.bits.rs1(22, 7)
          contexts(contextIndex).addressValid :=
            selectedContext.addressValid | (1.U << role)
        }
      }

      is(BiRaFunct.cfgMode.U) {
        val postMode = io.command.bits.rs1(10, 8)
        val encodingValid = configurationEncodingIsValid(
          io.command.bits.rs1(63, 14) === 0.U &&
            io.command.bits.rs2 === 0.U
        )
        when(!contextImplemented) {
          recordGlobalError(
            BiRaError.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!encodingValid) {
          recordContextError(
            contextId,
            BiRaError.badEnum.U,
            nextCommandSequence
          )
        }.elsewhen(
          !selectedContext.building ||
            selectedContext.inflightCount =/= 0.U
        ) {
          recordContextError(
            contextId,
            BiRaError.contextState.U,
            nextCommandSequence
          )
        }.elsewhen(postMode > BiRaPostMode.finalBilinearResidual.U) {
          recordContextError(
            contextId,
            BiRaError.badEnum.U,
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

      is(BiRaFunct.cfgCommit.U) {
        val encodingValid = configurationEncodingIsValid(
          io.command.bits.rs1(63, p.contextIdBits) === 0.U &&
            io.command.bits.rs2 === 0.U
        )
        when(!contextImplemented) {
          recordGlobalError(
            BiRaError.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!encodingValid) {
          recordContextError(
            contextId,
            BiRaError.badEnum.U,
            nextCommandSequence
          )
        }.elsewhen(
          !selectedContext.building ||
            selectedContext.inflightCount =/= 0.U
        ) {
          recordContextError(
            contextId,
            BiRaError.contextState.U,
            nextCommandSequence
          )
        }.otherwise {
          val inputExtentHeight =
            selectedContext.inputHeight +
              (selectedContext.paddingHeight << 1)
          val inputExtentWidth =
            selectedContext.inputWidth +
              (selectedContext.paddingWidth << 1)
          val expectedOutputHeight =
            inputExtentHeight - selectedContext.kernelHeight + 1.U
          val expectedOutputWidth =
            inputExtentWidth - selectedContext.kernelWidth + 1.U

          val shapeNonZero =
            selectedContext.inputHeight =/= 0.U &&
              selectedContext.inputWidth =/= 0.U &&
              selectedContext.inputChannels =/= 0.U &&
              selectedContext.outputHeight =/= 0.U &&
              selectedContext.outputWidth =/= 0.U &&
              selectedContext.outputChannels =/= 0.U &&
              selectedContext.kernelHeight =/= 0.U &&
              selectedContext.kernelWidth =/= 0.U
          val shapeWithinLimits =
            selectedContext.inputHeight <= p.maxImageHeight.U &&
              selectedContext.inputWidth <= p.maxImageWidth.U &&
              selectedContext.outputHeight <= p.maxImageHeight.U &&
              selectedContext.outputWidth <= p.maxImageWidth.U &&
              selectedContext.inputChannels <= p.maxInputChannels.U &&
              selectedContext.outputChannels <=
                (p.maxOutputBlocks * p.dim).U &&
              selectedContext.kernelHeight <= p.kernelSize.U &&
              selectedContext.kernelWidth <= p.kernelSize.U
          val convolutionShapeMatches =
            inputExtentHeight >= selectedContext.kernelHeight &&
              inputExtentWidth >= selectedContext.kernelWidth &&
              selectedContext.outputHeight === expectedOutputHeight &&
              selectedContext.outputWidth === expectedOutputWidth

          val addressValid = selectedContext.addressValid
          val hasInput = addressValid(BiRaAddrRole.input)
          val hasWeightLow = addressValid(BiRaAddrRole.weightLow)
          val hasWeightHigh = addressValid(BiRaAddrRole.weightHigh)
          val hasParameter = addressValid(BiRaAddrRole.parameter)
          val hasResidual = addressValid(BiRaAddrRole.residual)
          val hasCorrection = addressValid(BiRaAddrRole.correction)
          val hasAccumulator = addressValid(BiRaAddrRole.accumulator)
          val hasOutputFull = addressValid(BiRaAddrRole.outputFull)
          val hasOutputBinary =
            addressValid(BiRaAddrRole.outputBinary)

          val binaryMode =
            selectedContext.arrayMode === BiRaArrayMode.binary.U
          val columnReduce =
            selectedContext.arrayMode === BiRaArrayMode.columnReduce.U
          val depthwise =
            selectedContext.arrayMode === BiRaArrayMode.depthwise.U
          val requiresWeightHigh =
            !binaryMode &&
              selectedContext.weightPrecision ===
                BiRaWeightPrecision.w16.U
          val requiresResidual =
            selectedContext.postMode === BiRaPostMode.binaryFused.U ||
              selectedContext.postMode ===
                BiRaPostMode.finalBilinearResidual.U
          val requiresParameter =
            !binaryMode ||
              selectedContext.postMode =/= BiRaPostMode.none.U

          val requiredAddressesPresent =
            hasInput &&
              hasWeightLow &&
              hasAccumulator &&
              Mux(requiresParameter, hasParameter, true.B) &&
              Mux(requiresWeightHigh, hasWeightHigh, true.B) &&
              Mux(binaryMode, hasCorrection, true.B) &&
              Mux(requiresResidual, hasResidual, true.B) &&
              Mux(selectedContext.writeFull, hasOutputFull, true.B) &&
              Mux(
                selectedContext.writeBinary,
                hasOutputBinary,
                true.B
              )

          val outputSelectionValid =
            selectedContext.writeFull ||
              selectedContext.writeBinary ||
              selectedContext.postMode === BiRaPostMode.none.U
          val depthwiseValid =
            !depthwise ||
              selectedContext.inputChannels ===
                selectedContext.outputChannels
          val binaryValid =
            !binaryMode ||
              (selectedContext.inputChannels % p.dim.U === 0.U)
          val columnReduceValid =
            !columnReduce ||
              (selectedContext.inputChannels <= (p.dim / 2).U &&
                selectedContext.outputChannels === 1.U)
          val shuffleOutputChannels =
            (p.dim / 2) * p.maxShuffleScale * p.maxShuffleScale
          val shuffleValid =
            !selectedContext.shufflePack2 ||
              (selectedContext.arrayMode === BiRaArrayMode.dense.U &&
                selectedContext.postMode === BiRaPostMode.intPrelu.U &&
                selectedContext.outputChannels ===
                  shuffleOutputChannels.U &&
                selectedContext.writeFull &&
                !selectedContext.writeBinary)
          val binaryPostValid =
            selectedContext.postMode =/= BiRaPostMode.binaryFused.U ||
              binaryMode
          val finalPostValid =
            selectedContext.postMode =/=
              BiRaPostMode.finalBilinearResidual.U ||
              (columnReduce &&
                selectedContext.outputChannels === 1.U &&
                selectedContext.outputHeight %
                  p.maxShuffleScale.U === 0.U &&
                selectedContext.outputWidth %
                  p.maxShuffleScale.U === 0.U)

          when(
            !selectedContext.shapeValid ||
              !selectedContext.modeValid
          ) {
            recordContextError(
              contextId,
              BiRaError.contextState.U,
              nextCommandSequence
            )
          }.elsewhen(
            !shapeNonZero ||
              !shapeWithinLimits ||
              !convolutionShapeMatches
          ) {
            recordContextError(
              contextId,
              BiRaError.badShape.U,
              nextCommandSequence
            )
          }.elsewhen(!requiredAddressesPresent) {
            recordContextError(
              contextId,
              BiRaError.missingAddress.U,
              nextCommandSequence
            )
          }.elsewhen(
            !outputSelectionValid ||
              !depthwiseValid ||
              !binaryValid ||
              !columnReduceValid ||
              !shuffleValid ||
              !binaryPostValid ||
              !finalPostValid
          ) {
            recordContextError(
              contextId,
              BiRaError.illegalCombination.U,
              nextCommandSequence
            )
          }.elsewhen(
            selectedContext.errorCode =/= BiRaError.none.U
          ) {
            // Preserve the first configuration error.
          }.otherwise {
            contexts(contextIndex).building := false.B
            contexts(contextIndex).ready := true.B
            contexts(contextIndex).committed := true.B
          }
        }
      }

      is(BiRaFunct.load2d.U) {
        when(
          io.command.bits.xd ||
            !io.command.bits.xs1 ||
            !io.command.bits.xs2
        ) {
          recordContextError(
            dmaContextId,
            BiRaError.badEnum.U,
            nextCommandSequence
          )
          when(io.command.bits.xd) {
            responseValid := true.B
            responseRd := io.command.bits.rd
            responseData := BiRaError.badEnum.U
          }
        }.elsewhen(!dmaContextImplemented) {
          recordGlobalError(
            BiRaError.invalidContext.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(dmaRole >= BiRaAddrRole.count.U) {
          recordContextError(
            dmaContextId,
            BiRaError.badRole.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.addressValid(dmaRole)) {
          recordContextError(
            dmaContextId,
            BiRaError.missingAddress.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.ready) {
          recordContextError(
            dmaContextId,
            BiRaError.contextNotReady.U,
            nextCommandSequence
          )
        }.elsewhen(
          dmaSelectedContext.errorCode =/= BiRaError.none.U
        ) {
          recordGlobalError(
            BiRaError.contextFailed.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(34, 21) === 0.U ||
            io.command.bits.rs2(41, 35) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            BiRaError.zeroSize.U,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(57, 42) <
            io.command.bits.rs2(41, 35) ||
            io.command.bits.rs2(63, 58) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            BiRaError.badStride.U,
            nextCommandSequence
          )
        }
      }

      is(BiRaFunct.execConv.U) {
        when(!execEncodingValid) {
          recordContextError(
            contextId,
            BiRaError.badEnum.U,
            nextCommandSequence
          )
          when(io.command.bits.xd) {
            responseValid := true.B
            responseRd := io.command.bits.rd
            responseData := BiRaError.badEnum.U
          }
        }.elsewhen(!contextImplemented) {
          recordGlobalError(
            BiRaError.invalidContext.U,
            contextId,
            nextCommandSequence
          )
        }.elsewhen(!selectedContext.ready) {
          recordContextError(
            contextId,
            BiRaError.contextNotReady.U,
            nextCommandSequence
          )
        }.elsewhen(
          selectedContext.errorCode =/= BiRaError.none.U
        ) {
          recordGlobalError(
            BiRaError.contextFailed.U,
            contextId,
            nextCommandSequence
          )
        }
      }

      is(BiRaFunct.store2d.U) {
        when(
          io.command.bits.xd ||
            !io.command.bits.xs1 ||
            !io.command.bits.xs2
        ) {
          recordContextError(
            dmaContextId,
            BiRaError.badEnum.U,
            nextCommandSequence
          )
          when(io.command.bits.xd) {
            responseValid := true.B
            responseRd := io.command.bits.rd
            responseData := BiRaError.badEnum.U
          }
        }.elsewhen(!dmaContextImplemented) {
          recordGlobalError(
            BiRaError.invalidContext.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(dmaRole >= BiRaAddrRole.count.U) {
          recordContextError(
            dmaContextId,
            BiRaError.badRole.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.addressValid(dmaRole)) {
          recordContextError(
            dmaContextId,
            BiRaError.missingAddress.U,
            nextCommandSequence
          )
        }.elsewhen(!dmaSelectedContext.ready) {
          recordContextError(
            dmaContextId,
            BiRaError.contextNotReady.U,
            nextCommandSequence
          )
        }.elsewhen(
          dmaSelectedContext.errorCode =/= BiRaError.none.U
        ) {
          recordGlobalError(
            BiRaError.contextFailed.U,
            dmaContextId,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(34, 21) === 0.U ||
            io.command.bits.rs2(41, 35) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            BiRaError.zeroSize.U,
            nextCommandSequence
          )
        }.elsewhen(
          io.command.bits.rs2(57, 42) <
            io.command.bits.rs2(41, 35) ||
            io.command.bits.rs2(63, 58) === 0.U
        ) {
          recordContextError(
            dmaContextId,
            BiRaError.badStride.U,
            nextCommandSequence
          )
        }
      }

      is(BiRaFunct.fence.U) {
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
            (BiRaError.badEnum.U(8.W) << 1)
        }.otherwise {
          fencePending := true.B
          fenceContextScope := io.command.bits.rs1(0)
          fenceContextId :=
            io.command.bits.rs1(p.contextIdBits, 1)
          fenceRd := io.command.bits.rd
        }
      }

      is(BiRaFunct.status.U) {
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
            (BiRaError.badEnum.U(8.W) << 13) | (1.U << 12)
        }.elsewhen(statusContextScope) {
          when(statusContextId < p.nContexts.U) {
            val ctx = contexts(statusContextIndex)
            val clearBlocked = clearError && ctx.inflightCount =/= 0.U
            val reportedError = Mux(
              clearBlocked,
              BiRaError.contextBusy.U,
              ctx.errorCode
            )
            responseData :=
              1.U |
                (ctx.building.asUInt << 1) |
                (ctx.ready.asUInt << 2) |
                (ctx.committed.asUInt << 3) |
                (ctx.inflightCount << 4) |
                ((reportedError =/= BiRaError.none.U).asUInt << 12) |
                (reportedError << 13) |
                (ctx.errorCommandSequence << 21)
            when(clearError && !clearBlocked) {
              contexts(statusContextIndex).errorCode :=
                BiRaError.none.U
              contexts(statusContextIndex).errorCommandSequence := 0.U
            }
          }.otherwise {
            responseData :=
              (BiRaError.invalidContext.U(8.W) << 13) |
                (1.U << 12)
          }
        }.otherwise {
          val clearBlocked = clearError && anyInflight
          val reportedError = Mux(
            clearBlocked,
            BiRaError.contextBusy.U,
            globalErrorCode
          )
          responseData :=
            (io.schedulerStatus.loadQueueCount.orR.asUInt << 0) |
              (io.schedulerStatus.execQueueCount.orR.asUInt << 1) |
              (io.schedulerStatus.storeQueueCount.orR.asUInt << 2) |
              (io.schedulerStatus.loadBusy.asUInt << 3) |
              (io.schedulerStatus.execBusy.asUInt << 4) |
              (io.schedulerStatus.storeBusy.asUInt << 5) |
              ((reportedError =/= BiRaError.none.U).asUInt << 6) |
              (reportedError << 7) |
              (globalErrorContext << 15) |
              (io.schedulerStatus.loadQueueCount << 18) |
              (io.schedulerStatus.execQueueCount << 26) |
              (io.schedulerStatus.storeQueueCount << 34)
          when(clearError && !clearBlocked) {
            globalErrorCode := BiRaError.none.U
            globalErrorContext := 0.U
            globalErrorSequence := 0.U
            for (i <- 0 until p.nContexts) {
              contexts(i).errorCode := BiRaError.none.U
              contexts(i).errorCommandSequence := 0.U
            }
          }
        }
      }

      is(BiRaFunct.tlbFlush.U) {
        val encodingValid =
          io.command.bits.xd &&
            io.command.bits.xs1 &&
            io.command.bits.xs2 &&
            io.command.bits.rs1 === 0.U &&
            io.command.bits.rs2 === 0.U
        when(!encodingValid) {
          responseValid := true.B
          responseRd := io.command.bits.rd
          responseData := BiRaError.badEnum.U
        }.otherwise {
          flushPending := true.B
          flushIssued := false.B
          flushRd := io.command.bits.rd
        }
      }

    }
  }

  when(io.command.fire && !knownFunct) {
    recordGlobalError(
      BiRaError.badEnum.U,
      0.U,
      nextCommandSequence
    )
    when(io.command.bits.xd) {
      responseValid := true.B
      responseRd := io.command.bits.rd
      responseData := BiRaError.badEnum.U
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
      BiRaError.invalidContext.U
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
      (errorCode === BiRaError.none.U).asUInt |
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
    when(io.tlbFlushDone.bits =/= BiRaError.none.U) {
      recordGlobalError(
        io.tlbFlushDone.bits,
        0.U,
        nextCommandSequence
      )
    }
  }
}
