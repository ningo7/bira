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
class BilinearInterp(p: AccelParams) extends Module {
  private val sampleCount = 4
  private val sampleIndexBits = log2Ceil(sampleCount)
  private val sampleCountBits = log2Ceil(sampleCount + 1)
  private val laneIndexBits = log2Ceil(p.dim)
  private val sampleLinearBits = 2 * p.imageDimensionBits + 1
  private val coordinateBits = p.imageDimensionBits + p.shuffleLogBits + 3
  private val denominatorBits = log2Ceil(2 * p.maxShuffleScale + 1)
  private val horizontalBits =
    p.activationBits + denominatorBits + 1
  private val numeratorBits =
    p.activationBits + 2 * denominatorBits + 3

  val io = IO(new Bundle {
    val request =
      Flipped(Decoupled(new InterpReq(p)))
    val response =
      Decoupled(Vec(p.dim, UInt(p.activationBits.W)))

    val readRequest = Decoupled(UInt(p.fullAddressBits.W))
    val readResponse =
      Flipped(Valid(Vec(p.dim, UInt(p.activationBits.W))))
  })

  private val Seq(
    idle,
    prepareLane,
    prepareSampleAddresses,
    streamSampleReads,
    horizontalStage,
    verticalStage,
    sendResponse
  ) = Enum(7)

  private val state = RegInit(idle)
  private val requestReg =
    RegInit(0.U.asTypeOf(new InterpReq(p)))
  private val lane = RegInit(0.U(laneIndexBits.W))
  private val sampleIssueCount = RegInit(0.U(sampleCountBits.W))
  private val responseSampleIndex = Reg(UInt(sampleIndexBits.W))
  private val responseSampleLane = Reg(UInt(laneIndexBits.W))
  private val sampleRows =
    Reg(Vec(sampleCount, UInt(p.fullAddressBits.W)))
  private val sampleLanes =
    Reg(Vec(sampleCount, UInt(laneIndexBits.W)))
  private val samples =
    RegInit(VecInit.fill(sampleCount)(0.U(p.activationBits.W)))
  private val result =
    RegInit(VecInit.fill(p.dim)(0.U(p.activationBits.W)))
  private val outputX = RegInit(0.U(p.imageDimensionBits.W))
  private val outputY = RegInit(0.U(p.imageDimensionBits.W))
  private val sourceY0Reg = Reg(UInt(p.imageDimensionBits.W))
  private val sourceY1Reg = Reg(UInt(p.imageDimensionBits.W))
  private val sourceX0Reg = Reg(UInt(p.imageDimensionBits.W))
  private val sourceX1Reg = Reg(UInt(p.imageDimensionBits.W))
  private val weightY0Reg = Reg(UInt(denominatorBits.W))
  private val weightY1Reg = Reg(UInt(denominatorBits.W))
  private val weightX0Reg = Reg(UInt(denominatorBits.W))
  private val weightX1Reg = Reg(UInt(denominatorBits.W))

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

  private def advanceOutputCoordinate(): Unit = {
    when(outputX === requestReg.outputWidth - 1.U) {
      outputX := 0.U
      outputY := outputY + 1.U
    }.otherwise {
      outputX := outputX + 1.U
    }
  }

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

  io.request.ready := state === idle
  io.response.valid := state === sendResponse
  io.response.bits := result
  io.readRequest.valid :=
    state === streamSampleReads && sampleIssueCount < sampleCount.U
  io.readRequest.bits := sampleRows(0)

  when(io.readRequest.fire) {
    responseSampleIndex :=
      sampleIssueCount(sampleIndexBits - 1, 0)
    responseSampleLane := sampleLanes(0)
    for (sample <- 0 until sampleCount - 1) {
      sampleRows(sample) := sampleRows(sample + 1)
      sampleLanes(sample) := sampleLanes(sample + 1)
    }
    sampleIssueCount := sampleIssueCount + 1.U
  }

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
    outputX := io.request.bits.outputStartX
    outputY := io.request.bits.outputStartY
    sampleIssueCount := 0.U
    result := VecInit.fill(p.dim)(0.U)
    state := prepareLane
  }

  switch(state) {
    is(prepareLane) {
      when(laneIsLive) {
        sourceY0Reg := sourceY0
        sourceY1Reg := sourceY1
        sourceX0Reg := sourceX0
        sourceX1Reg := sourceX1
        weightY0Reg := weightY0
        weightY1Reg := weightY1
        weightX0Reg := weightX0
        weightX1Reg := weightX1
        state := prepareSampleAddresses
      }.otherwise {
        result(lane) := 0.U
        when(lane === (p.dim - 1).U) {
          state := sendResponse
        }.otherwise {
          lane := lane + 1.U
          advanceOutputCoordinate()
        }
      }
    }

    is(prepareSampleAddresses) {
      // The four bilinear samples form a 2x2 rectangle.  Calculate only the
      // top-left linear index with a multiplier; the remaining three differ
      // by either zero/one column or zero/one input row.  Registering the
      // packed row/lane here prevents coordinate arithmetic from crossing the
      // shared-SPAD arbiter and reaching a BRAM address pin in one cycle.
      val topLeft = Wire(UInt(sampleLinearBits.W))
      val xStep = Wire(UInt(sampleLinearBits.W))
      val rowStep = Wire(UInt(sampleLinearBits.W))
      val linearSamples = Wire(Vec(sampleCount, UInt(sampleLinearBits.W)))
      topLeft :=
        sourceY0Reg * requestReg.inputWidth + sourceX0Reg
      xStep := sourceX1Reg - sourceX0Reg
      rowStep := Mux(
        sourceY1Reg === sourceY0Reg,
        0.U,
        requestReg.inputWidth
      )
      linearSamples(0) := topLeft
      linearSamples(1) := topLeft + xStep
      linearSamples(2) := topLeft + rowStep
      linearSamples(3) := topLeft + rowStep + xStep
      for (sample <- 0 until sampleCount) {
        sampleRows(sample) :=
          requestReg.inputBase +
            (linearSamples(sample) >> p.laneIndexBits)
        sampleLanes(sample) :=
          linearSamples(sample)(laneIndexBits - 1, 0)
      }
      sampleIssueCount := 0.U
      state := streamSampleReads
    }

    is(streamSampleReads) {
      when(io.readResponse.valid) {
        samples(responseSampleIndex) :=
          io.readResponse.bits(responseSampleLane)
        when(responseSampleIndex === (sampleCount - 1).U) {
          state := horizontalStage
        }
      }
    }

    is(horizontalStage) {
      horizontalTop :=
        samples(0) * weightX0Reg + samples(1) * weightX1Reg
      horizontalBottom :=
        samples(2) * weightX0Reg + samples(3) * weightX1Reg
      verticalWeight0 := weightY0Reg
      verticalWeight1 := weightY1Reg

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
        advanceOutputCoordinate()
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
