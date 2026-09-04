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

/** Select one packed per-pixel binary correction from a Parameter row. */
class CorrectionReq(p: AccelParams) extends Bundle {
  val address = UInt(p.parameterAddressBits.W)
}

/** Both interpretations are returned; arrayMode selects the consumer. */
class DecodedParams(p: AccelParams) extends Bundle {
  val multiBit = new ConvParamWrite(p)
  val binary = new BinParamWrite(p)
}

/** DMA-writable raw Parameter Buffer with decoded and packed-table reads.
  *
  * Each 512-bit row stores two 256-bit lane records. A default 16-lane output
  * block therefore consumes eight rows. Multi-bit and binary layers use two
  * frozen record layouts over the same raw storage. A separate row range may
  * pack 16 per-pixel signed-int32 binary corrections into each 512-bit row.
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
    val readResponse = Decoupled(new DecodedParams(p))
    val correctionReadRequest =
      Flipped(Decoupled(new CorrectionReq(p)))
    val correctionReadResponse =
      Decoupled(UInt(512.W))
  })

  private val memory = SyncReadMem(p.parameterRows, UInt(512.W))
  private val rows = Reg(
    Vec(p.parameterRowsPerBlock, UInt(512.W))
  )
  private val request =
    Reg(new ParamRead(p))
  private val correctionAddress =
    Reg(UInt(p.parameterAddressBits.W))
  private val correctionValue =
    Reg(UInt(512.W))
  private val rowIndexBits =
    log2Ceil(p.parameterRowsPerBlock max 2)
  private val issueIndex = RegInit(0.U(rowIndexBits.W))

  private val Seq(
    idle,
    reading,
    draining,
    sending,
    readingCorrection,
    drainingCorrection,
    sendingCorrection
  ) = Enum(7)
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
    state := reading
  }
  when(io.correctionReadRequest.fire) {
    correctionAddress := io.correctionReadRequest.bits.address
    state := readingCorrection
  }

  private val blockReadAddressWide =
    request.baseRow +
      request.block * p.parameterRowsPerBlock.U +
      issueIndex
  private val readEnable =
    state === reading || state === readingCorrection
  private val readAddress = Mux(
    state === readingCorrection,
    correctionAddress,
    blockReadAddressWide(p.parameterAddressBits - 1, 0)
  )
  private val readData = memory.read(readAddress, readEnable)
  private val returnedValid = RegNext(readEnable, false.B)
  private val returnedIndex = RegEnable(issueIndex, readEnable)
  private val returnedCorrection =
    RegNext(state === readingCorrection, false.B)

  when(readEnable) {
    when(state === readingCorrection) {
      state := drainingCorrection
    }.otherwise {
      when(issueIndex === (p.parameterRowsPerBlock - 1).U) {
        state := draining
      }.otherwise {
        issueIndex := issueIndex + 1.U
      }
    }
  }

  when(returnedValid && returnedCorrection) {
    correctionValue := readData
    state := sendingCorrection
  }.elsewhen(returnedValid) {
    rows(returnedIndex) := readData
    when(returnedIndex === (p.parameterRowsPerBlock - 1).U) {
      state := sending
    }
  }

  private val decoded =
    WireDefault(0.U.asTypeOf(new DecodedParams(p)))
  decoded.multiBit.block := request.block
  decoded.binary.block := request.block

  for (lane <- 0 until p.dim) {
    val row = rows(lane / 2)
    val record =
      if (lane % 2 == 0) row(255, 0) else row(511, 256)

    // Multi-bit lane record.
    decoded.multiBit.bias(lane) := record(31, 0).asSInt
    decoded.multiBit.post(lane).positiveShift :=
      record(39, 32).asSInt
    decoded.multiBit.post(lane).negativeCoeff1 :=
      record(41, 40).asSInt
    decoded.multiBit.post(lane).negativeCoeff2 :=
      record(43, 42).asSInt
    decoded.multiBit.post(lane).negativeLeftShift1 :=
      record(48, 44)
    decoded.multiBit.post(lane).negativeLeftShift2 :=
      record(53, 49)
    decoded.multiBit.post(lane).negativeCommonShift :=
      record(61, 54).asSInt
    decoded.multiBit.post(lane).qMin :=
      record(93, 62).asSInt
    decoded.multiBit.post(lane).qMax :=
      record(125, 94).asSInt
    decoded.multiBit.binaryThreshold(lane) :=
      record(157, 126).asSInt

    // Binary lane record.
    decoded.binary.post(lane).threshold :=
      record(31, 0).asSInt
    decoded.binary.post(lane).positiveCoeff2 :=
      record(33, 32).asSInt
    decoded.binary.post(lane).positiveLeftShift1 :=
      record(38, 34)
    decoded.binary.post(lane).positiveLeftShift2 :=
      record(43, 39)
    decoded.binary.post(lane).positiveCommonShift :=
      record(51, 44).asSInt
    decoded.binary.post(lane).positiveBias :=
      record(83, 52).asSInt
    decoded.binary.post(lane).negativeCoeff1 :=
      record(85, 84).asSInt
    decoded.binary.post(lane).negativeCoeff2 :=
      record(87, 86).asSInt
    decoded.binary.post(lane).negativeLeftShift1 :=
      record(92, 88)
    decoded.binary.post(lane).negativeLeftShift2 :=
      record(97, 93)
    decoded.binary.post(lane).negativeCommonShift :=
      record(105, 98).asSInt
    decoded.binary.post(lane).negativeBias :=
      record(137, 106).asSInt
    decoded.binary.post(lane).qMin :=
      record(169, 138).asSInt
    decoded.binary.post(lane).qMax :=
      record(201, 170).asSInt
    decoded.binary.outputSignThreshold(lane) :=
      record(233, 202).asSInt
  }

  io.readResponse.valid := state === sending
  io.readResponse.bits := decoded
  when(io.readResponse.fire) {
    state := idle
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
