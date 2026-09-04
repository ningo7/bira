package bira

// Compute and local-memory integration top.

import chisel3._
import chisel3.util._
import _root_.circt.stage.ChiselStage

/** Standalone accelerator core used before the RoCC/Gemmini shell is added.
  *
  * Multi-bit and packed-binary controllers share the scratchpads,
  * accumulator, and host programming ports. Commands are mutually exclusive;
  * the future ISA frontend can retain this boundary and replace the temporary
  * host-facing Decoupled ports with decoded RoCC commands and DMA traffic.
  */
class Core(p: AccelParams = AccelParams()) extends Module {
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new ConvolutionCommand(p)))
    val binaryCommand =
      Flipped(Decoupled(new BinaryConvolutionCommand(p)))
    val parameterWrite =
      Flipped(Decoupled(new ConvParamWrite(p)))
    val binaryParameterWrite =
      Flipped(Decoupled(new BinParamWrite(p)))
    val accumulatorWrite =
      Flipped(Decoupled(new AccProgWrite(p)))
    val accumulatorReadRequest =
      Flipped(Decoupled(UInt(p.accumulatorAddressBits.W)))
    val accumulatorReadResponse =
      Valid(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val correctionReadRequest =
      Decoupled(new CorrectionReq(p))
    val correctionReadResponse =
      Flipped(Decoupled(UInt(512.W)))
    val status = Output(new AcceleratorStatus)

    val fullWrite = Flipped(
      Decoupled(
        new VectorWriteRequest(
          p.fullAddressBits,
          p.dim,
          p.activationBits
        )
      )
    )
    val fullReadRequest =
      Flipped(Decoupled(UInt(p.fullAddressBits.W)))
    val fullReadResponse =
      Valid(Vec(p.dim, UInt(p.activationBits.W)))

    val binaryWrite = Flipped(
      Decoupled(
        new VectorWriteRequest(
          p.binaryAddressBits,
          p.dim,
          1
        )
      )
    )
    val binaryReadRequest =
      Flipped(Decoupled(UInt(p.binaryAddressBits.W)))
    val binaryReadResponse = Valid(Vec(p.dim, UInt(1.W)))
  })

  private val fullSpad = Module(
    new BankedSpad(
      banks = p.fullBanks,
      rowsPerBank = p.bankRows,
      lanes = p.dim,
      elementBits = p.activationBits,
      readPorts = 6
    )
  )
  private val binSpad = Module(
    new BankedSpad(
      banks = p.binaryBanks,
      rowsPerBank = p.bankRows,
      lanes = p.dim,
      elementBits = 1,
      readPorts = 3
    )
  )
  private val accum = Module(new VecAccum(p))
  private val arrayDispatch =
    Module(new ComputeArrayDispatch(p))
  private val array =
    Module(new ComputeArray(p))
  private val intPost = Module(new IntPostProc(p))
  private val binPost =
    Module(new BinPostProc(p))
  private val bilinear = Module(new BilinearInterp(p))
  private val convFetch = Module(new ConvFetch(p))
  private val convCtrl = Module(new ConvCtrl(p))
  private val binCtrl =
    Module(new BinConvCtrl(p))

  private val biasRegs = Reg(
    Vec(
      p.maxOutputBlocks,
      Vec(p.dim, SInt(p.accumulatorBits.W))
    )
  )
  private val postParamRegs = Reg(
    Vec(
      p.maxOutputBlocks,
      Vec(p.dim, new PostProcessParameters(p))
    )
  )
  private val binThreshRegs = Reg(
    Vec(
      p.maxOutputBlocks,
      Vec(p.dim, SInt(p.accumulatorBits.W))
    )
  )
  private val binPostParamRegs = Reg(
    Vec(
      p.maxOutputBlocks,
      Vec(p.dim, new BinPostParams(p))
    )
  )
  private val binSignThreshRegs = Reg(
    Vec(
      p.maxOutputBlocks,
      Vec(p.dim, SInt(p.accumulatorBits.W))
    )
  )

  private val accProgPending = RegInit(false.B)
  private val accReadPending = RegInit(false.B)
  private val hostAccPending =
    accProgPending || accReadPending
  private val ctrlBusy =
    convCtrl.io.status.busy || binCtrl.io.status.busy
  private val Seq(accOwnerIdle, accOwnerMulti, accOwnerBinary) = Enum(3)
  private val accumulatorOwner = RegInit(accOwnerIdle)

  // Controller ownership changes only at command boundaries.  Use this local
  // register for the shared Accumulator mux so one controller's ready cone
  // cannot propagate through the other controller's live busy/state logic.
  when(io.command.fire) {
    accumulatorOwner := accOwnerMulti
  }.elsewhen(io.binaryCommand.fire) {
    accumulatorOwner := accOwnerBinary
  }.elsewhen(
    (accumulatorOwner === accOwnerMulti && convCtrl.io.status.done) ||
      (accumulatorOwner === accOwnerBinary && binCtrl.io.status.done)
  ) {
    accumulatorOwner := accOwnerIdle
  }

  io.parameterWrite.ready :=
    !ctrlBusy &&
      !hostAccPending &&
      io.parameterWrite.bits.block < p.maxOutputBlocks.U
  when(io.parameterWrite.fire) {
    if (p.maxOutputBlocks == 1) {
      biasRegs(0) := io.parameterWrite.bits.bias
      postParamRegs(0) := io.parameterWrite.bits.post
      binThreshRegs(0) :=
        io.parameterWrite.bits.binaryThreshold
    } else {
      val indexBits = log2Ceil(p.maxOutputBlocks)
      val parameterBlock =
        io.parameterWrite.bits.block(indexBits - 1, 0)
      biasRegs(parameterBlock) :=
        io.parameterWrite.bits.bias
      postParamRegs(parameterBlock) :=
        io.parameterWrite.bits.post
      binThreshRegs(parameterBlock) :=
        io.parameterWrite.bits.binaryThreshold
    }
  }

  io.binaryParameterWrite.ready :=
    !ctrlBusy &&
      !hostAccPending &&
      io.binaryParameterWrite.bits.block < p.maxOutputBlocks.U
  when(io.binaryParameterWrite.fire) {
    if (p.maxOutputBlocks == 1) {
      binPostParamRegs(0) :=
        io.binaryParameterWrite.bits.post
      binSignThreshRegs(0) :=
        io.binaryParameterWrite.bits.outputSignThreshold
    } else {
      val indexBits = log2Ceil(p.maxOutputBlocks)
      val parameterBlock =
        io.binaryParameterWrite.bits.block(indexBits - 1, 0)
      binPostParamRegs(parameterBlock) :=
        io.binaryParameterWrite.bits.post
      binSignThreshRegs(parameterBlock) :=
        io.binaryParameterWrite.bits.outputSignThreshold
    }
  }

  // Multi-bit commands have priority when both command ports are asserted in
  // the same idle cycle. Once one controller starts, the other remains gated.
  convCtrl.io.command.valid :=
    io.command.valid &&
      !binCtrl.io.status.busy &&
      !hostAccPending
  convCtrl.io.command.bits := io.command.bits
  io.command.ready :=
    convCtrl.io.command.ready &&
      !binCtrl.io.status.busy &&
      !hostAccPending

  binCtrl.io.command.valid :=
    io.binaryCommand.valid &&
      !convCtrl.io.status.busy &&
      !hostAccPending &&
      !io.command.valid
  binCtrl.io.command.bits := io.binaryCommand.bits
  io.binaryCommand.ready :=
    binCtrl.io.command.ready &&
      !convCtrl.io.status.busy &&
      !hostAccPending &&
      !io.command.valid

  convCtrl.io.biases := biasRegs
  convCtrl.io.postParameters := postParamRegs
  convCtrl.io.binaryThresholds := binThreshRegs
  binCtrl.io.postParameters :=
    binPostParamRegs
  binCtrl.io.outputSignThresholds :=
    binSignThreshRegs
  io.correctionReadRequest <> binCtrl.io.correctionReadReq
  binCtrl.io.correctionReadResp <> io.correctionReadResponse

  io.status.busy :=
    ctrlBusy || hostAccPending
  io.status.done :=
    convCtrl.io.status.done || binCtrl.io.status.done

  convCtrl.io.actFetchReq <>
    convFetch.io.actReq
  convCtrl.io.actFetchResp <>
    convFetch.io.actResp
  convCtrl.io.wgtFetchReq <>
    convFetch.io.wgtReq
  convCtrl.io.wgtFetchResp <>
    convFetch.io.wgtResp

  bilinear.io.request <> convCtrl.io.interpolationReq
  convCtrl.io.interpolationResp <> bilinear.io.response

  fullSpad.io.readRequest(0) <>
    convFetch.io.actReadReq
  convFetch.io.actReadResp <>
    fullSpad.io.readResponse(0)
  fullSpad.io.readRequest(1) <>
    convFetch.io.wgtLoRdReq
  convFetch.io.wgtLoRdResp <>
    fullSpad.io.readResponse(1)
  fullSpad.io.readRequest(2) <>
    convFetch.io.wgtHiRdReq
  convFetch.io.wgtHiRdResp <>
    fullSpad.io.readResponse(2)
  // Keep binary residual and bilinear traffic on distinct logical ports.
  // They remain mutually exclusive at command level, but separating them
  // prevents one client's address decoder from entering the other client's
  // ready/state-enable timing cone.
  fullSpad.io.readRequest(3) <> binCtrl.io.resReadReq
  binCtrl.io.resReadResp :=
    fullSpad.io.readResponse(3)
  fullSpad.io.readRequest(4) <> bilinear.io.readRequest
  bilinear.io.readResponse :=
    fullSpad.io.readResponse(4)
  fullSpad.io.readRequest(5) <> io.fullReadRequest
  io.fullReadResponse := fullSpad.io.readResponse(5)

  private val fullWrArb = Module(
    new Arbiter(
      new VectorWriteRequest(
        p.fullAddressBits,
        p.dim,
        p.activationBits
      ),
      3
    )
  )
  fullWrArb.io.in(0) <> convCtrl.io.fullOutWrite
  fullWrArb.io.in(1) <>
    binCtrl.io.fullOutWrite
  fullWrArb.io.in(2) <> io.fullWrite
  fullSpad.io.writeRequest <> fullWrArb.io.out

  // The accumulator has one ordered request/response port. Controller
  // ownership is exclusive, and host correction programming is accepted only
  // while both controllers are idle.
  accum.io.request.valid := false.B
  accum.io.request.bits :=
    0.U.asTypeOf(new AccumulatorRequest(p))
  convCtrl.io.accReq.ready := false.B
  binCtrl.io.accReq.ready := false.B
  io.accumulatorWrite.ready := false.B
  io.accumulatorReadRequest.ready := false.B
  io.accumulatorReadResponse.valid := false.B
  io.accumulatorReadResponse.bits := accum.io.response.bits.data

  when(accumulatorOwner === accOwnerBinary) {
    accum.io.request.valid :=
      binCtrl.io.accReq.valid
    accum.io.request.bits :=
      binCtrl.io.accReq.bits
    binCtrl.io.accReq.ready :=
      accum.io.request.ready
  }.elsewhen(accumulatorOwner === accOwnerMulti) {
    accum.io.request.valid :=
      convCtrl.io.accReq.valid
    accum.io.request.bits :=
      convCtrl.io.accReq.bits
    convCtrl.io.accReq.ready :=
      accum.io.request.ready
  }.otherwise {
    accum.io.request.valid :=
      (io.accumulatorWrite.valid ||
        io.accumulatorReadRequest.valid) &&
        !hostAccPending &&
        !io.command.valid &&
        !io.binaryCommand.valid
    val chooseWrite = io.accumulatorWrite.valid
    accum.io.request.bits.operation := Mux(
      chooseWrite,
      AccumulatorOperation.write,
      AccumulatorOperation.read
    )
    accum.io.request.bits.address := Mux(
      chooseWrite,
      io.accumulatorWrite.bits.address,
      io.accumulatorReadRequest.bits
    )
    accum.io.request.bits.data := io.accumulatorWrite.bits.data
    accum.io.request.bits.shift := 0.U
    accum.io.request.bits.negate := false.B
    io.accumulatorWrite.ready :=
      accum.io.request.ready &&
        !accProgPending &&
        !accReadPending &&
        !io.command.valid &&
        !io.binaryCommand.valid
    io.accumulatorReadRequest.ready :=
      accum.io.request.ready &&
        !io.accumulatorWrite.valid &&
        !hostAccPending &&
        !io.command.valid &&
        !io.binaryCommand.valid
  }

  when(io.accumulatorWrite.fire) {
    accProgPending := true.B
  }
  when(io.accumulatorReadRequest.fire) {
    accReadPending := true.B
  }

  convCtrl.io.accResp.valid :=
    accum.io.response.valid &&
      accumulatorOwner === accOwnerMulti
  convCtrl.io.accResp.bits :=
    accum.io.response.bits
  binCtrl.io.accResp.valid :=
    accum.io.response.valid &&
      accumulatorOwner === accOwnerBinary
  binCtrl.io.accResp.bits :=
    accum.io.response.bits
  accum.io.response.ready := Mux(
    accumulatorOwner === accOwnerBinary,
    binCtrl.io.accResp.ready,
    Mux(
      accumulatorOwner === accOwnerMulti,
      convCtrl.io.accResp.ready,
      hostAccPending
    )
  )
  when(
    accProgPending &&
      accum.io.response.fire &&
      !ctrlBusy
  ) {
    accProgPending := false.B
  }
  io.accumulatorReadResponse.valid :=
    accReadPending && accum.io.response.valid && !ctrlBusy
  when(
    accReadPending &&
      accum.io.response.fire &&
      !ctrlBusy
  ) {
    accReadPending := false.B
  }

  arrayDispatch.io.binaryMode :=
    accumulatorOwner === accOwnerBinary
  arrayDispatch.io.multiInputValid :=
    convCtrl.io.arrayInValid
  arrayDispatch.io.multiActivationBits :=
    convCtrl.io.arrayActBits
  arrayDispatch.io.multiWeights := convCtrl.io.arrayWgts
  arrayDispatch.io.weightPrecision :=
    convCtrl.io.arrayWPrec
  arrayDispatch.io.columnReduceMode :=
    convCtrl.io.arrayColumnReduce
  arrayDispatch.io.binaryInputValid :=
    binCtrl.io.arrayInValid
  arrayDispatch.io.binaryActivation :=
    binCtrl.io.arrayAct
  arrayDispatch.io.binaryWeights :=
    binCtrl.io.arrayWgts
  array.io.input := arrayDispatch.io.output
  convCtrl.io.arraySums :=
    array.io.columnSums
  convCtrl.io.arrayOutValid :=
    array.io.outputValid &&
      convCtrl.io.status.busy
  convCtrl.io.arrayReducedSum :=
    array.io.columnReducedSum
  convCtrl.io.arrayReducedValid :=
    array.io.columnReducedValid &&
      convCtrl.io.status.busy
  binCtrl.io.arraySums :=
    array.io.columnSums
  binCtrl.io.arrayOutValid :=
    array.io.outputValid &&
      binCtrl.io.status.busy

  intPost.io.inputValid := convCtrl.io.postInValid
  intPost.io.input := convCtrl.io.postIn
  intPost.io.parameters := convCtrl.io.postParams
  convCtrl.io.postOut := intPost.io.output
  convCtrl.io.postOutValid := intPost.io.outputValid

  binPost.io.inputValid :=
    binCtrl.io.postInValid
  binPost.io.accumulator :=
    binCtrl.io.postAcc
  binPost.io.correction :=
    binCtrl.io.postCorrection
  binPost.io.residual :=
    binCtrl.io.postRes
  binPost.io.parameters :=
    binCtrl.io.postParams
  binCtrl.io.postOut :=
    binPost.io.output
  binCtrl.io.postOutValid :=
    binPost.io.outputValid

  binSpad.io.readRequest(0) <>
    binCtrl.io.actReadReq
  binCtrl.io.actReadResp :=
    binSpad.io.readResponse(0)
  binSpad.io.readRequest(1) <>
    binCtrl.io.wgtReadReq
  binCtrl.io.wgtReadResp :=
    binSpad.io.readResponse(1)
  binSpad.io.readRequest(2) <> io.binaryReadRequest
  io.binaryReadResponse := binSpad.io.readResponse(2)

  private val binWrArb = Module(
    new Arbiter(
      new VectorWriteRequest(
        p.binaryAddressBits,
        p.dim,
        1
      ),
      3
    )
  )
  binWrArb.io.in(0) <> convCtrl.io.binOutWrite
  binWrArb.io.in(1) <>
    binCtrl.io.binOutWrite
  binWrArb.io.in(2) <> io.binaryWrite
  binSpad.io.writeRequest <> binWrArb.io.out

  when(ctrlBusy) {
    assert(
      !io.parameterWrite.fire &&
        !io.binaryParameterWrite.fire &&
        !io.accumulatorWrite.fire &&
        !io.accumulatorReadRequest.fire,
      "decoded parameters must remain stable while running"
    )
  }
}

/** Generate a standalone SystemVerilog top for inspection and synthesis. */
object GenCore extends App {
  private val targetDir =
    args.headOption.getOrElse("build/generated-rtl")
  ChiselStage.emitSystemVerilogFile(
    new Core(),
    args = Array("--target-dir", targetDir),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info"
    )
  )
}
