package bira

// Binary post-processing pipeline.

import chisel3._

/** Two-stage fused scale/RPReLU, residual add, and state saturation.
  *
  * This module implements the integer reference model exactly:
  *
  *   dot = 2 * popcount + correction  // correction is compiler-supplied -N
  *   delta = dot >= threshold ? positive(dot) : negative(dot)
  *   state = clamp(residual + delta, qMin, qMax)
  *
  * Signed right shifts round the magnitude to nearest and then restore sign,
  * matching Python `shift_integer` and the C hardware reference.
  */
class BinPostProc(p: BiRaParams) extends Module {
  private val workBits = p.accumulatorBits * 2

  val io = IO(new Bundle {
    val inputValid = Input(Bool())
    val accumulator =
      Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val correction = Input(SInt(p.accumulatorBits.W))
    val residual =
      Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val parameters =
      Input(Vec(p.dim, new BinaryPostProcessParameters(p)))
    val outputValid = Output(Bool())
    val output =
      Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
  })

  private def clippedAmount(amount: UInt): UInt =
    Mux(
      amount > p.maxPostProcessLeftShift.U,
      p.maxPostProcessLeftShift.U,
      amount
    )

  private def shiftInteger(value: SInt, shift: SInt): SInt = {
    val extended = value.pad(workBits)
    val shiftIsLeft = !shift(shift.getWidth - 1)
    val leftAmount = clippedAmount(shift.asUInt)
    val rightAmount = clippedAmount((-shift).asUInt)

    val shiftedLeft =
      (extended << leftAmount)(workBits - 1, 0).asSInt
    val magnitude = Mux(
      extended < 0.S,
      ((~extended.asUInt) + 1.U)(workBits - 1, 0),
      extended.asUInt
    )
    val rounding = Mux(
      rightAmount === 0.U,
      0.U(workBits.W),
      (1.U(workBits.W) << (rightAmount - 1.U))(
        workBits - 1,
        0
      )
    )
    val roundedMagnitude =
      ((magnitude +& rounding) >> rightAmount)(
        workBits - 1,
        0
      )
    val shiftedRight = Mux(
      extended < 0.S,
      -roundedMagnitude.asSInt,
      roundedMagnitude.asSInt
    )
    Mux(shiftIsLeft, shiftedLeft, shiftedRight)
  }

  private def applyCoefficient(
    value: SInt,
    coefficient: SInt
  ): SInt =
    Mux(
      coefficient === 1.S,
      value,
      Mux(
        coefficient === (-1).S,
        -value,
        0.S(workBits.W)
      )
    )

  private def affine(
    value: SInt,
    coefficient1: SInt,
    coefficient2: SInt,
    leftShift1: UInt,
    leftShift2: UInt,
    commonShift: SInt,
    bias: SInt
  ): SInt = {
    val extended = value.pad(workBits)
    val term1 = applyCoefficient(
      shiftInteger(extended, leftShift1.zext),
      coefficient1
    )
    val term2 = applyCoefficient(
      shiftInteger(extended, leftShift2.zext),
      coefficient2
    )
    val scaled = shiftInteger(
      (term1 + term2).asSInt,
      commonShift
    )
    (scaled + bias.pad(workBits)).asSInt
  }

  private val stage1Delta =
    Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Residual =
    Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Minimum =
    Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Maximum =
    Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Valid = RegInit(false.B)

  stage1Valid := io.inputValid
  for (lane <- 0 until p.dim) {
    val value = (
      io.accumulator(lane) + io.correction
    )(p.accumulatorBits - 1, 0).asSInt
    val parameters = io.parameters(lane)

    val positive = affine(
      value,
      1.S(2.W),
      parameters.positiveCoeff2,
      parameters.positiveLeftShift1,
      parameters.positiveLeftShift2,
      parameters.positiveCommonShift,
      parameters.positiveBias
    )
    val negative = affine(
      value,
      parameters.negativeCoeff1,
      parameters.negativeCoeff2,
      parameters.negativeLeftShift1,
      parameters.negativeLeftShift2,
      parameters.negativeCommonShift,
      parameters.negativeBias
    )
    val delta = Mux(
      value >= parameters.threshold,
      positive,
      negative
    )

    when(io.inputValid) {
      stage1Delta(lane) :=
        delta(workBits - 1, 0).asSInt
      stage1Residual(lane) :=
        io.residual(lane).pad(workBits)
      stage1Minimum(lane) := parameters.qMin.pad(workBits)
      stage1Maximum(lane) := parameters.qMax.pad(workBits)
    }
  }

  private val outputRegister =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val outputValidRegister = RegInit(false.B)
  outputValidRegister := stage1Valid
  for (lane <- 0 until p.dim) {
    val withResidual =
      stage1Delta(lane) + stage1Residual(lane)
    val clamped = Mux(
      withResidual < stage1Minimum(lane),
      stage1Minimum(lane),
      Mux(
        withResidual > stage1Maximum(lane),
        stage1Maximum(lane),
        withResidual
      )
    )
    when(stage1Valid) {
      outputRegister(lane) :=
        clamped(p.accumulatorBits - 1, 0).asSInt
    }
  }

  io.output := outputRegister
  io.outputValid := outputValidRegister
}
