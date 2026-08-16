package bira

// Compute-layer bilinear residual unit.

import chisel3._
import chisel3.util._

/** Integer bilinear interpolation for the final image-level residual.
  *
  * The arithmetic matches `bilinear_pixel()` in
  * `sw/models/bfsrcnn/reference/bfsrcnn_reference.c`:
  *
  *   source = (2 * output + 1 - scale) / (2 * scale)
  *
  * Coordinates use floor division and are clamped to the image boundary.
  * The scale must be a power of two. Four source samples are read for each
  * live output lane; repeated border samples are intentionally not cached in
  * this first implementation.
  *
  * The arithmetic is split across two registered stages after the reads:
  * horizontal weighting, followed by vertical weighting/rounding.
  */
class BilinearInterp(p: BiRaParams) extends Module {
  private val sampleCount = 4
  private val sampleIndexBits = log2Ceil(sampleCount)
  private val laneIndexBits = log2Ceil(p.dim)
  private val coordinateBits = p.imageDimensionBits + p.shuffleLogBits + 3
  private val denominatorBits = log2Ceil(2 * p.maxShuffleScale + 1)
  private val horizontalBits =
    p.activationBits + denominatorBits + 1
  private val numeratorBits =
    p.activationBits + 2 * denominatorBits + 3

  val io = IO(new Bundle {
    val request =
      Flipped(Decoupled(new BilinearInterpolationRequest(p)))
    val response =
      Decoupled(Vec(p.dim, UInt(p.activationBits.W)))

    val readRequest = Decoupled(UInt(p.fullAddressBits.W))
    val readResponse =
      Flipped(Valid(Vec(p.dim, UInt(p.activationBits.W))))
  })

  private val Seq(
    idle,
    prepareLane,
    issueSampleRead,
    waitForSampleRead,
    horizontalStage,
    verticalStage,
    sendResponse
  ) = Enum(7)

  private val state = RegInit(idle)
  private val requestReg =
    RegInit(0.U.asTypeOf(new BilinearInterpolationRequest(p)))
  private val lane = RegInit(0.U(laneIndexBits.W))
  private val sampleIndex = RegInit(0.U(sampleIndexBits.W))
  private val samples =
    RegInit(VecInit.fill(sampleCount)(0.U(p.activationBits.W)))
  private val result =
    RegInit(VecInit.fill(p.dim)(0.U(p.activationBits.W)))

  private val horizontalTop = Reg(UInt(horizontalBits.W))
  private val horizontalBottom = Reg(UInt(horizontalBits.W))
  private val verticalWeight0 = Reg(UInt(denominatorBits.W))
  private val verticalWeight1 = Reg(UInt(denominatorBits.W))
  private val roundingBias = Reg(UInt(numeratorBits.W))
  private val resultShift = Reg(UInt((p.shuffleLogBits + 2).W))

  private val outputIndex =
    requestReg.outputStartPixel + lane
  private val laneIsLive =
    outputIndex < requestReg.outputPixelCount
  private val outputY = outputIndex / requestReg.outputWidth
  private val outputX = outputIndex % requestReg.outputWidth

  private val scaleWide =
    (1.U(coordinateBits.W) << requestReg.scaleLog2)(
      coordinateBits - 1,
      0
    )
  private val denominatorShift =
    requestReg.scaleLog2 + 1.U
  private val denominator =
    (1.U(denominatorBits.W) << denominatorShift)(
      denominatorBits - 1,
      0
    )

  private val yNumerator = Wire(SInt(coordinateBits.W))
  private val xNumerator = Wire(SInt(coordinateBits.W))
  yNumerator :=
    (outputY << 1).zext.pad(coordinateBits) +
      1.S(coordinateBits.W) -
      scaleWide.zext
  xNumerator :=
    (outputX << 1).zext.pad(coordinateBits) +
      1.S(coordinateBits.W) -
      scaleWide.zext

  // Arithmetic right shift implements floor division by a positive power of
  // two for two's-complement signed values, including the negative border.
  private val sourceY0Unclamped =
    yNumerator >> denominatorShift
  private val sourceX0Unclamped =
    xNumerator >> denominatorShift
  private val sourceY1Unclamped =
    sourceY0Unclamped + 1.S
  private val sourceX1Unclamped =
    sourceX0Unclamped + 1.S

  private def clampCoordinate(
    coordinate: SInt,
    limit: UInt
  ): UInt = {
    Mux(
      coordinate < 0.S,
      0.U,
      Mux(
        coordinate >= limit.zext,
        limit - 1.U,
        coordinate.asUInt
      )
    )
  }

  private val sourceY0 =
    clampCoordinate(sourceY0Unclamped, requestReg.inputHeight)
  private val sourceY1 =
    clampCoordinate(sourceY1Unclamped, requestReg.inputHeight)
  private val sourceX0 =
    clampCoordinate(sourceX0Unclamped, requestReg.inputWidth)
  private val sourceX1 =
    clampCoordinate(sourceX1Unclamped, requestReg.inputWidth)

  private val yRemainder =
    (yNumerator -
      (sourceY0Unclamped << denominatorShift)).asUInt
  private val xRemainder =
    (xNumerator -
      (sourceX0Unclamped << denominatorShift)).asUInt
  private val weightY1 =
    yRemainder(denominatorBits - 1, 0)
  private val weightX1 =
    xRemainder(denominatorBits - 1, 0)
  private val weightY0 = denominator - weightY1
  private val weightX0 = denominator - weightX1

  private val sampleY = Mux(
    sampleIndex >= 2.U,
    sourceY1,
    sourceY0
  )
  private val sampleX = Mux(
    sampleIndex(0),
    sourceX1,
    sourceX0
  )
  private val sampleLinearIndex =
    sampleY * requestReg.inputWidth + sampleX
  private val sampleRow =
    requestReg.inputBase +
      (sampleLinearIndex >> p.laneIndexBits)
  private val sampleLane =
    sampleLinearIndex(p.laneIndexBits - 1, 0)

  io.request.ready := state === idle
  io.response.valid := state === sendResponse
  io.response.bits := result
  io.readRequest.valid := state === issueSampleRead
  io.readRequest.bits := sampleRow

  when(io.request.fire) {
    assert(io.request.bits.inputHeight > 0.U)
    assert(io.request.bits.inputWidth > 0.U)
    assert(io.request.bits.outputWidth > 0.U)
    assert(io.request.bits.outputPixelCount > 0.U)
    assert(io.request.bits.scaleLog2 > 0.U)
    assert(
      io.request.bits.scaleLog2 <=
        log2Ceil(p.maxShuffleScale).U
    )

    requestReg := io.request.bits
    lane := 0.U
    sampleIndex := 0.U
    result := VecInit.fill(p.dim)(0.U)
    state := prepareLane
  }

  switch(state) {
    is(prepareLane) {
      when(laneIsLive) {
        sampleIndex := 0.U
        state := issueSampleRead
      }.otherwise {
        result(lane) := 0.U
        when(lane === (p.dim - 1).U) {
          state := sendResponse
        }.otherwise {
          lane := lane + 1.U
        }
      }
    }

    is(issueSampleRead) {
      when(io.readRequest.fire) {
        state := waitForSampleRead
      }
    }

    is(waitForSampleRead) {
      when(io.readResponse.valid) {
        samples(sampleIndex) := io.readResponse.bits(sampleLane)
        when(sampleIndex === (sampleCount - 1).U) {
          state := horizontalStage
        }.otherwise {
          sampleIndex := sampleIndex + 1.U
          state := issueSampleRead
        }
      }
    }

    is(horizontalStage) {
      horizontalTop :=
        samples(0) * weightX0 + samples(1) * weightX1
      horizontalBottom :=
        samples(2) * weightX0 + samples(3) * weightX1
      verticalWeight0 := weightY0
      verticalWeight1 := weightY1

      // The divisor is (2*scale)^2. Add half before shifting to
      // implement the same positive rounding used by the C reference.
      val squaredShift = denominatorShift << 1
      roundingBias :=
        1.U(numeratorBits.W) << (squaredShift - 1.U)
      resultShift := squaredShift
      state := verticalStage
    }

    is(verticalStage) {
      val numerator = Wire(UInt(numeratorBits.W))
      numerator :=
        horizontalTop * verticalWeight0 +
          horizontalBottom * verticalWeight1 +
          roundingBias
      val interpolated = numerator >> resultShift
      result(lane) := Mux(
        interpolated > ((1 << p.activationBits) - 1).U,
        ((1 << p.activationBits) - 1).U,
        interpolated(p.activationBits - 1, 0)
      )

      when(lane === (p.dim - 1).U) {
        state := sendResponse
      }.otherwise {
        lane := lane + 1.U
        state := prepareLane
      }
    }

    is(sendResponse) {
      when(io.response.fire) {
        state := idle
      }
    }
  }
}
