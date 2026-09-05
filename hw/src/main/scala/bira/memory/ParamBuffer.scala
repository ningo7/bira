package bira

// Parameter Buffer storage and ABI decoding.

import chisel3._
import chisel3.util._
import _root_.circt.stage.ChiselStage

/** One native 64-byte Parameter Buffer row. */
class ParamWrite(p: AccelParams) extends Bundle {
  val address = UInt(p.parameterAddressBits.W)
  val data = UInt(512.W)
}

/** Request the parameter records for one 16-lane output block. */
class ParamRead(p: AccelParams) extends Bundle {
  val baseRow = UInt(p.parameterAddressBits.W)
  val block = UInt(p.blockIndexBits.W)
}

/** One raw ABI row returned with its destination block and lane pair. */
class ParamRowResponse(p: AccelParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val lanePair = UInt(p.parameterLanePairBits.W)
  val data = UInt(512.W)
  val last = Bool()
}

/** Read packed per-pixel binary accumulator biases from a Parameter row. */
class CorrectionReq(p: AccelParams) extends Bundle {
  val address = UInt(p.parameterAddressBits.W)
}

/** DMA-writable raw Parameter Buffer with streaming and packed-table reads.
  *
  * Each 512-bit row stores two 256-bit lane records. A default 16-lane output
  * block therefore consumes eight rows. Multi-bit and binary layers use two
  * frozen record layouts over the same raw storage. A separate row range may
  * pack 16 per-pixel signed-int32 binary accumulator biases into each
  * 512-bit row. The public port keeps the correction name for ABI stability.
  */
class ParamBuffer(p: AccelParams) extends Module {
  require(p.dim % 2 == 0, "parameter rows contain exactly two lane records")
  require(
    512 % p.accumulatorBits == 0 &&
      isPow2(p.correctionEntriesPerRow),
    "one Parameter row must contain a power-of-two number of corrections"
  )
  require(
    p.postShiftAmountBits == 5,
    "the ISA parameter ABI allocates five bits to each left shift"
  )

  val io = IO(new Bundle {
    val write = Flipped(Decoupled(new ParamWrite(p)))
    val readRequest =
      Flipped(Decoupled(new ParamRead(p)))
    val readResponse = Decoupled(new ParamRowResponse(p))
    val correctionReadRequest =
      Flipped(Decoupled(new CorrectionReq(p)))
    val correctionReadResponse =
      Decoupled(UInt(512.W))
  })

  private val memory = SyncReadMem(p.parameterRows, UInt(512.W))
  private val request =
    Reg(new ParamRead(p))
  private val correctionAddress =
    Reg(UInt(p.parameterAddressBits.W))
  private val correctionValue =
    Reg(UInt(512.W))
  private val rowIndexBits =
    log2Ceil(p.parameterRowsPerBlock max 2)
  private val issueIndex = RegInit(0.U(rowIndexBits.W))
  // Includes both rows already buffered and the one-cycle synchronous-memory
  // returns not yet enqueued. Two reservations are enough to sustain one raw
  // row per cycle while remaining safe under downstream backpressure.
  private val rowOutstanding = RegInit(0.U(2.W))
  // Use register slots rather than an asynchronously-read Queue RAM. The
  // payload is 520 bits wide, so a LUTRAM queue would add a wide read mux and
  // concentrate routing precisely on the parameter path being shortened.
  private val rowBuffer = Module(
    new TwoEntryBuffer(new ParamRowResponse(p))
  )

  private val Seq(
    idle,
    readingRows,
    drainingRows,
    issueCorrection,
    waitCorrection,
    sendingCorrection
  ) = Enum(6)
  private val state = RegInit(idle)

  io.write.ready := true.B
  when(io.write.fire) {
    memory.write(io.write.bits.address, io.write.bits.data)
  }

  io.readRequest.ready := state === idle
  io.correctionReadRequest.ready :=
    state === idle && !io.readRequest.valid
  when(io.readRequest.fire) {
    val blockEndExclusive =
      io.readRequest.bits.baseRow +&
        ((io.readRequest.bits.block +& 1.U) *
          p.parameterRowsPerBlock.U)
    assert(
      io.readRequest.bits.block < p.maxOutputBlocks.U,
      "parameter block index exceeds the configured output-block maximum"
    )
    assert(
      blockEndExclusive <= p.parameterRows.U,
      "parameter block exceeds Parameter Buffer capacity"
    )
    request := io.readRequest.bits
    issueIndex := 0.U
    rowOutstanding := 0.U
    state := readingRows
  }
  when(io.correctionReadRequest.fire) {
    correctionAddress := io.correctionReadRequest.bits.address
    state := issueCorrection
  }

  private val blockReadAddressWide =
    request.baseRow +
      request.block * p.parameterRowsPerBlock.U +
      issueIndex
  private val rowConsumed = io.readResponse.fire
  private val canIssueRow =
    rowOutstanding < 2.U || rowConsumed
  private val issueRow = state === readingRows && canIssueRow
  private val readEnable = issueRow || state === issueCorrection
  private val readAddress = Mux(
    state === issueCorrection,
    correctionAddress,
    blockReadAddressWide(p.parameterAddressBits - 1, 0)
  )
  private val readData = memory.read(readAddress, readEnable)
  private val returnedValid = RegNext(readEnable, false.B)
  private val returnedCorrection =
    RegNext(state === issueCorrection, false.B)
  private val returnedIndex = RegEnable(issueIndex, issueRow)

  when(issueRow) {
    when(issueIndex === (p.parameterRowsPerBlock - 1).U) {
      state := drainingRows
    }.otherwise {
      issueIndex := issueIndex + 1.U
    }
  }
  when(state === issueCorrection) {
    state := waitCorrection
  }

  rowBuffer.io.enq.valid := returnedValid && !returnedCorrection
  rowBuffer.io.enq.bits.block := request.block
  rowBuffer.io.enq.bits.lanePair := returnedIndex
  rowBuffer.io.enq.bits.data := readData
  rowBuffer.io.enq.bits.last :=
    returnedIndex === (p.parameterRowsPerBlock - 1).U
  when(rowBuffer.io.enq.valid) {
    assert(
      rowBuffer.io.enq.ready,
      "reserved parameter-row response buffer must have capacity"
    )
  }

  when(returnedValid && returnedCorrection) {
    correctionValue := readData
    state := sendingCorrection
  }

  io.readResponse <> rowBuffer.io.deq
  when(io.readResponse.fire && io.readResponse.bits.last) {
    state := idle
  }

  when(issueRow =/= rowConsumed) {
    when(issueRow) {
      rowOutstanding := rowOutstanding + 1.U
    }.otherwise {
      rowOutstanding := rowOutstanding - 1.U
    }
  }

  io.correctionReadResponse.valid := state === sendingCorrection
  io.correctionReadResponse.bits := correctionValue
  when(io.correctionReadResponse.fire) {
    state := idle
  }
}

object GenParamBuffer extends App {
  private val targetDir =
    args.headOption.getOrElse("build/generated-rtl")
  ChiselStage.emitSystemVerilogFile(
    new ParamBuffer(AccelParams(dim = 4, maxOutputBlocks = 4)),
    args = Array("--target-dir", targetDir),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
  )
}
