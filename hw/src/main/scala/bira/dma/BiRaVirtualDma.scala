package bira

// Virtual-to-physical DMA fragmentation bridges.

import chisel3._
import chisel3.util._

/** Exact-address request to the Rocket-facing address translator.
  *
  * `translationStatus` remains opaque in the standalone core. The later
  * FrontendTLB adapter converts it to Rocket's privilege/status fields.
  */
class BiRaTranslationRequest extends Bundle {
  val virtualAddress = UInt(64.W)
  val isWrite = Bool()
  val translationStatus = UInt(64.W)
}

class BiRaTranslationResponse extends Bundle {
  val physicalAddress = UInt(64.W)
  val errorCode = UInt(8.W)
}

/** Fixed-width physical bus beat used below the virtual DMA bridges. */
class BiRaPhysicalReadRequest extends Bundle {
  val physicalAddress = UInt(64.W)
}

class BiRaPhysicalReadResponse(p: BiRaParams) extends Bundle {
  val data = UInt((p.dmaBeatBytes * 8).W)
  val errorCode = UInt(8.W)
}

class BiRaPhysicalWriteRequest(p: BiRaParams) extends Bundle {
  val physicalAddress = UInt(64.W)
  val data = UInt((p.dmaBeatBytes * 8).W)
  val mask = UInt(p.dmaBeatBytes.W)
}

class BiRaPhysicalWriteResponse extends Bundle {
  val errorCode = UInt(8.W)
}

/** Common fragment calculations.
  *
  * Every physical request is one aligned fixed-width beat. A fragment never
  * crosses either that beat or a virtual page.
  */
private object BiRaVirtualDmaHelpers {
  def minimum(a: UInt, b: UInt): UInt = Mux(a < b, a, b)

  def fragmentBytes(
    virtualAddress: UInt,
    physicalAddress: UInt,
    remaining: UInt,
    p: BiRaParams
  ): UInt = {
    val virtualPageOffset =
      virtualAddress(p.pageOffsetBits - 1, 0)
    val physicalBeatOffset =
      physicalAddress(p.dmaBeatOffsetBits - 1, 0)
    val pageRemaining =
      p.pageBytes.U - virtualPageOffset
    val beatRemaining =
      p.dmaBeatBytes.U - physicalBeatOffset
    minimum(remaining, minimum(pageRemaining, beatRemaining))
  }

  def alignedBeatAddress(address: UInt, p: BiRaParams): UInt =
    Cat(
      address(63, p.dmaBeatOffsetBits),
      0.U(p.dmaBeatOffsetBits.W)
    )

  def pageOffsetsMatch(
    virtualAddress: UInt,
    physicalAddress: UInt,
    p: BiRaParams
  ): Bool =
    virtualAddress(p.pageOffsetBits - 1, 0) ===
      physicalAddress(p.pageOffsetBits - 1, 0)

  def lowByteEnable(bytes: UInt): UInt = {
    val mask = Wire(Vec(64, Bool()))
    for (byte <- 0 until 64) {
      mask(byte) := byte.U < bytes
    }
    mask.asUInt
  }

  def lowDataMask(bytes: UInt): UInt = {
    val mask = Wire(Vec(64, UInt(8.W)))
    for (byte <- 0 until 64) {
      mask(byte) := Mux(byte.U < bytes, "hff".U, 0.U)
    }
    mask.asUInt
  }
}

/** Virtual row-read bridge.
  *
  * It translates each current virtual fragment, performs aligned physical
  * beat reads, extracts only requested bytes, and assembles one 512-bit row.
  * The translator naturally sees a new request when the row crosses a page.
  */
class BiRaVirtualReadDma(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new BiRaExternalReadRequest))
    val response = Decoupled(new BiRaExternalReadResponse)

    val translationRequest = Decoupled(new BiRaTranslationRequest)
    val translationResponse =
      Flipped(Decoupled(new BiRaTranslationResponse))

    val physicalRequest = Decoupled(new BiRaPhysicalReadRequest)
    val physicalResponse =
      Flipped(Decoupled(new BiRaPhysicalReadResponse(p)))
  })

  private val Seq(
    idle,
    translate,
    waitTranslation,
    requestPhysical,
    waitPhysical,
    respond
  ) = Enum(6)
  private val state = RegInit(idle)

  private val requestReg = Reg(new BiRaExternalReadRequest)
  private val currentVirtualAddress = Reg(UInt(64.W))
  private val physicalBeatAddress = Reg(UInt(64.W))
  private val remainingBytes = Reg(UInt(7.W))
  private val outputByteOffset = Reg(UInt(7.W))
  private val beatByteOffset =
    Reg(UInt(p.dmaBeatOffsetBits.W))
  private val fragmentByteCount = Reg(UInt(7.W))
  private val assembledData = RegInit(0.U(512.W))
  private val responseError = RegInit(BiRaError.none.U(8.W))

  io.request.ready := state === idle
  when(io.request.fire) {
    requestReg := io.request.bits
    currentVirtualAddress := io.request.bits.virtualAddress
    remainingBytes := io.request.bits.bytes
    outputByteOffset := 0.U
    assembledData := 0.U
    responseError := Mux(
      io.request.bits.bytes === 0.U ||
        io.request.bits.bytes > 64.U,
      BiRaError.internalProtocol.U,
      BiRaError.none.U
    )
    state := Mux(
      io.request.bits.bytes === 0.U ||
        io.request.bits.bytes > 64.U,
      respond,
      translate
    )
  }

  io.translationRequest.valid := state === translate
  io.translationRequest.bits.virtualAddress :=
    currentVirtualAddress
  io.translationRequest.bits.isWrite := false.B
  io.translationRequest.bits.translationStatus :=
    requestReg.translationStatus
  when(io.translationRequest.fire) {
    state := waitTranslation
  }

  io.translationResponse.ready := state === waitTranslation
  when(io.translationResponse.fire) {
    when(io.translationResponse.bits.errorCode =/= BiRaError.none.U) {
      responseError := io.translationResponse.bits.errorCode
      state := respond
    }.elsewhen(
      !BiRaVirtualDmaHelpers.pageOffsetsMatch(
        currentVirtualAddress,
        io.translationResponse.bits.physicalAddress,
        p
      )
    ) {
      responseError := BiRaError.internalProtocol.U
      state := respond
    }.otherwise {
      val fragment = BiRaVirtualDmaHelpers.fragmentBytes(
        currentVirtualAddress,
        io.translationResponse.bits.physicalAddress,
        remainingBytes,
        p
      )
      physicalBeatAddress :=
        BiRaVirtualDmaHelpers.alignedBeatAddress(
          io.translationResponse.bits.physicalAddress,
          p
        )
      beatByteOffset :=
        io.translationResponse.bits.physicalAddress(
          p.dmaBeatOffsetBits - 1,
          0
        )
      fragmentByteCount := fragment
      state := requestPhysical
    }
  }

  io.physicalRequest.valid := state === requestPhysical
  io.physicalRequest.bits.physicalAddress :=
    physicalBeatAddress
  when(io.physicalRequest.fire) {
    state := waitPhysical
  }

  io.physicalResponse.ready := state === waitPhysical
  when(io.physicalResponse.fire) {
    when(io.physicalResponse.bits.errorCode =/= BiRaError.none.U) {
      responseError := io.physicalResponse.bits.errorCode
      state := respond
    }.otherwise {
      val shiftedBeat =
        io.physicalResponse.bits.data >>
          (beatByteOffset << 3)
      val fragmentMask =
        BiRaVirtualDmaHelpers.lowDataMask(fragmentByteCount)
      val selected =
        shiftedBeat.pad(512) & fragmentMask
      val merged =
        assembledData |
          (selected << (outputByteOffset << 3))(511, 0)
      assembledData := merged

      when(remainingBytes === fragmentByteCount) {
        responseError := BiRaError.none.U
        state := respond
      }.otherwise {
        currentVirtualAddress :=
          currentVirtualAddress + fragmentByteCount
        remainingBytes :=
          remainingBytes - fragmentByteCount
        outputByteOffset :=
          outputByteOffset + fragmentByteCount
        state := translate
      }
    }
  }

  io.response.valid := state === respond
  io.response.bits.data := assembledData
  io.response.bits.errorCode := responseError
  when(io.response.fire) {
    state := idle
  }
}

/** Virtual row-write bridge using aligned masked physical beats. */
class BiRaVirtualWriteDma(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new BiRaExternalWriteRequest))
    val response = Decoupled(new BiRaExternalWriteResponse)

    val translationRequest = Decoupled(new BiRaTranslationRequest)
    val translationResponse =
      Flipped(Decoupled(new BiRaTranslationResponse))

    val physicalRequest =
      Decoupled(new BiRaPhysicalWriteRequest(p))
    val physicalResponse =
      Flipped(Decoupled(new BiRaPhysicalWriteResponse))
  })

  private val Seq(
    idle,
    translate,
    waitTranslation,
    requestPhysical,
    waitPhysical,
    respond
  ) = Enum(6)
  private val state = RegInit(idle)

  private val requestReg = Reg(new BiRaExternalWriteRequest)
  private val currentVirtualAddress = Reg(UInt(64.W))
  private val physicalBeatAddress = Reg(UInt(64.W))
  private val remainingBytes = Reg(UInt(7.W))
  private val inputByteOffset = Reg(UInt(7.W))
  private val beatByteOffset =
    Reg(UInt(p.dmaBeatOffsetBits.W))
  private val fragmentByteCount = Reg(UInt(7.W))
  private val responseError = RegInit(BiRaError.none.U(8.W))

  io.request.ready := state === idle
  when(io.request.fire) {
    requestReg := io.request.bits
    currentVirtualAddress := io.request.bits.virtualAddress
    remainingBytes := io.request.bits.bytes
    inputByteOffset := 0.U
    responseError := Mux(
      io.request.bits.bytes === 0.U ||
        io.request.bits.bytes > 64.U,
      BiRaError.internalProtocol.U,
      BiRaError.none.U
    )
    state := Mux(
      io.request.bits.bytes === 0.U ||
        io.request.bits.bytes > 64.U,
      respond,
      translate
    )
  }

  io.translationRequest.valid := state === translate
  io.translationRequest.bits.virtualAddress :=
    currentVirtualAddress
  io.translationRequest.bits.isWrite := true.B
  io.translationRequest.bits.translationStatus :=
    requestReg.translationStatus
  when(io.translationRequest.fire) {
    state := waitTranslation
  }

  io.translationResponse.ready := state === waitTranslation
  when(io.translationResponse.fire) {
    when(io.translationResponse.bits.errorCode =/= BiRaError.none.U) {
      responseError := io.translationResponse.bits.errorCode
      state := respond
    }.elsewhen(
      !BiRaVirtualDmaHelpers.pageOffsetsMatch(
        currentVirtualAddress,
        io.translationResponse.bits.physicalAddress,
        p
      )
    ) {
      responseError := BiRaError.internalProtocol.U
      state := respond
    }.otherwise {
      val fragment = BiRaVirtualDmaHelpers.fragmentBytes(
        currentVirtualAddress,
        io.translationResponse.bits.physicalAddress,
        remainingBytes,
        p
      )
      physicalBeatAddress :=
        BiRaVirtualDmaHelpers.alignedBeatAddress(
          io.translationResponse.bits.physicalAddress,
          p
        )
      beatByteOffset :=
        io.translationResponse.bits.physicalAddress(
          p.dmaBeatOffsetBits - 1,
          0
        )
      fragmentByteCount := fragment
      state := requestPhysical
    }
  }

  private val sourceShifted =
    requestReg.data >> (inputByteOffset << 3)
  private val fragmentMask =
    BiRaVirtualDmaHelpers.lowDataMask(fragmentByteCount)
  private val fragmentByteEnable =
    BiRaVirtualDmaHelpers.lowByteEnable(fragmentByteCount)
  private val selectedSource =
    sourceShifted & fragmentMask
  private val positionedData =
    selectedSource << (beatByteOffset << 3)
  private val positionedMask =
    fragmentByteEnable(p.dmaBeatBytes - 1, 0) <<
      beatByteOffset

  io.physicalRequest.valid := state === requestPhysical
  io.physicalRequest.bits.physicalAddress :=
    physicalBeatAddress
  io.physicalRequest.bits.data :=
    positionedData(p.dmaBeatBytes * 8 - 1, 0)
  io.physicalRequest.bits.mask :=
    positionedMask(p.dmaBeatBytes - 1, 0)
  when(io.physicalRequest.fire) {
    state := waitPhysical
  }

  io.physicalResponse.ready := state === waitPhysical
  when(io.physicalResponse.fire) {
    when(io.physicalResponse.bits.errorCode =/= BiRaError.none.U) {
      responseError := io.physicalResponse.bits.errorCode
      state := respond
    }.elsewhen(remainingBytes === fragmentByteCount) {
      responseError := BiRaError.none.U
      state := respond
    }.otherwise {
      currentVirtualAddress :=
        currentVirtualAddress + fragmentByteCount
      remainingBytes :=
        remainingBytes - fragmentByteCount
      inputByteOffset :=
        inputByteOffset + fragmentByteCount
      state := translate
    }
  }

  io.response.valid := state === respond
  io.response.bits.errorCode := responseError
  when(io.response.fire) {
    state := idle
  }
}
