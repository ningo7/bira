package bira

// Integer post-processing pipeline.

import chisel3._

/** Two-stage vector PReLU/requantization unit.
  *
  * Signed right shifts use round-to-nearest on the magnitude and then restore
  * the sign. This is intentionally not an arithmetic `>>`, whose rounding for
  * negative values would differ from the software model.
  */
class IntPostProc(p: BiRaParams) extends Module {
  private val workBits = p.accumulatorBits * 2

  val io = IO(new Bundle {
    val inputValid = Input(Bool())
    val input = Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val parameters = Input(Vec(p.dim, new PostProcessParameters(p)))
    val outputValid = Output(Bool())
    val output = Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
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

    val leftWide = extended << leftAmount
    val shiftedLeft = leftWide(workBits - 1, 0).asSInt

    val magnitude = Mux(
      extended < 0.S,
      ((~extended.asUInt) + 1.U)(workBits - 1, 0),
      extended.asUInt
    )
    val roundingBit = Mux(
      rightAmount === 0.U,
      0.U(workBits.W),
      (1.U(workBits.W) << (rightAmount - 1.U))(workBits - 1, 0)
    )
    val roundedMagnitude =
      ((magnitude +& roundingBit) >> rightAmount)(workBits - 1, 0)
    val shiftedRight = Mux(
      extended < 0.S,
      -roundedMagnitude.asSInt,
      roundedMagnitude.asSInt
    )

    Mux(shiftIsLeft, shiftedLeft, shiftedRight)
  }

  private def applyCoefficient(value: SInt, coefficient: SInt): SInt =
    Mux(
      coefficient === 1.S,
      value,
      Mux(coefficient === (-1).S, -value, 0.S(workBits.W))
    )

  private val stage1Value =
    Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Minimum =
    Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Maximum =
    Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Valid = RegInit(false.B)

  stage1Valid := io.inputValid
  for (lane <- 0 until p.dim) {
    val value = io.input(lane).pad(workBits)
    val parameters = io.parameters(lane)

    val positive = shiftInteger(value, parameters.positiveShift)

    val negativeTerm1 = shiftInteger(
      value,
      parameters.negativeLeftShift1.zext
    )
    val negativeTerm2 = shiftInteger(
      value,
      parameters.negativeLeftShift2.zext
    )
    val negativeSum =
      applyCoefficient(negativeTerm1, parameters.negativeCoeff1) +
        applyCoefficient(negativeTerm2, parameters.negativeCoeff2)
    val negative = shiftInteger(
      negativeSum.asSInt,
      parameters.negativeCommonShift
    )

    val selected = Mux(value >= 0.S, positive, negative)
    when(io.inputValid) {
      stage1Value(lane) :=
        selected(workBits - 1, 0).asSInt
      stage1Minimum(lane) := parameters.qMin.pad(workBits)
      stage1Maximum(lane) := parameters.qMax.pad(workBits)
    }
  }

  private val outputRegister =
    Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val outputValidRegister = RegInit(false.B)
  outputValidRegister := stage1Valid
  for (lane <- 0 until p.dim) {
    val clamped = Mux(
      stage1Value(lane) < stage1Minimum(lane),
      stage1Minimum(lane),
      Mux(
        stage1Value(lane) > stage1Maximum(lane),
        stage1Maximum(lane),
        stage1Value(lane)
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
