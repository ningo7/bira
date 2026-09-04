package bira

// Binary post-processing pipeline.

import chisel3._

object BinPostProc {
  val latency = 7
}

/** Seven-stage branch-selected binary affine/RPReLU and residual pipeline.
  *
  *   dot   = 2 * popcount + correction
  *   delta = selectedAffine(dot)
  *   state = clamp(residual + delta, qMin, qMax)
  *
  * Threshold selection precedes the affine datapath, so only one branch is
  * evaluated per lane. Registers isolate correction, the parallel term
  * shifts, coefficient application, term addition, the common shift, bias,
  * and residual saturation. The pipeline accepts one vector every cycle.
  */
class BinPostProc(p: AccelParams) extends Module {
  private val workBits = p.accumulatorBits * 2

  val io = IO(new Bundle {
    val inputValid = Input(Bool())
    val accumulator = Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val correction = Input(SInt(p.accumulatorBits.W))
    val residual = Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val parameters = Input(Vec(p.dim, new BinPostParams(p)))
    val outputValid = Output(Bool())
    val output = Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
  })

  private def clippedAmount(amount: UInt): UInt = {
    val bounded = Mux(
      amount > p.maxPostProcessLeftShift.U,
      p.maxPostProcessLeftShift.U,
      amount
    )
    // Keep the physical barrel-shifter selector at the bounded width. Leaving
    // the original shiftBits width here makes FIRRTL conservatively build a
    // workBits + 2^shiftBits - 1 result before truncation (319 bits for the
    // default 64-bit work value and 8-bit encoded shift).
    bounded(p.postShiftAmountBits - 1, 0)
  }

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
      (1.U(workBits.W) << (rightAmount - 1.U))(workBits - 1, 0)
    )
    val roundedMagnitude =
      ((magnitude +& rounding) >> rightAmount)(workBits - 1, 0)
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

  // P0: isolate correction selection and the accumulator addition.
  private val stage0Valid = RegNext(io.inputValid, false.B)
  private val stage0Dot = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage0Residual = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage0Parameters = Reg(Vec(p.dim, new BinPostParams(p)))
  for (lane <- 0 until p.dim) {
    val dot =
      (io.accumulator(lane) + io.correction)(p.accumulatorBits - 1, 0).asSInt
    when(io.inputValid) {
      stage0Dot(lane) := dot
      stage0Residual(lane) := io.residual(lane)
      stage0Parameters(lane) := io.parameters(lane)
    }
  }

  // P1: select the active RPReLU branch and evaluate its two shifts in
  // parallel. Coefficient application is deliberately deferred to P2 so a
  // dynamic rounded shift never feeds another wide negation in one cycle.
  private val stage1Valid = RegNext(stage0Valid, false.B)
  private val stage1Shifted1 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Shifted2 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Coefficient1 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage1Coefficient2 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage1CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  private val stage1Bias = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Residual = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Minimum = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Maximum = Reg(Vec(p.dim, SInt(workBits.W)))
  for (lane <- 0 until p.dim) {
    val value = stage0Dot(lane).pad(workBits)
    val params = stage0Parameters(lane)
    val dot = stage0Dot(lane)
    val positive = dot >= params.threshold
    val coefficient1 = Mux(positive, 1.S(2.W), params.negativeCoeff1)
    val coefficient2 = Mux(
      positive,
      params.positiveCoeff2,
      params.negativeCoeff2
    )
    val leftShift1 = Mux(
      positive,
      params.positiveLeftShift1,
      params.negativeLeftShift1
    )
    val leftShift2 = Mux(
      positive,
      params.positiveLeftShift2,
      params.negativeLeftShift2
    )
    val commonShift = Mux(
      positive,
      params.positiveCommonShift,
      params.negativeCommonShift
    )
    val bias = Mux(positive, params.positiveBias, params.negativeBias)
    when(stage0Valid) {
      stage1Shifted1(lane) := shiftInteger(value, leftShift1.zext)
      stage1Shifted2(lane) := shiftInteger(value, leftShift2.zext)
      stage1Coefficient1(lane) := coefficient1
      stage1Coefficient2(lane) := coefficient2
      stage1CommonShift(lane) := commonShift
      stage1Bias(lane) := bias.pad(workBits)
      stage1Residual(lane) := stage0Residual(lane).pad(workBits)
      stage1Minimum(lane) := params.qMin.pad(workBits)
      stage1Maximum(lane) := params.qMax.pad(workBits)
    }
  }

  // P2: apply the selected {-1, 0, +1} coefficients. This is a mux/negate
  // stage and no longer shares a cycle with either dynamic shift.
  private val stage2Valid = RegNext(stage1Valid, false.B)
  private val stage2Term1 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2Term2 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  private val stage2Bias = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2Residual = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2Minimum = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2Maximum = Reg(Vec(p.dim, SInt(workBits.W)))
  for (lane <- 0 until p.dim) {
    when(stage1Valid) {
      stage2Term1(lane) :=
        applyCoefficient(stage1Shifted1(lane), stage1Coefficient1(lane))
      stage2Term2(lane) :=
        applyCoefficient(stage1Shifted2(lane), stage1Coefficient2(lane))
      stage2CommonShift(lane) := stage1CommonShift(lane)
      stage2Bias(lane) := stage1Bias(lane)
      stage2Residual(lane) := stage1Residual(lane)
      stage2Minimum(lane) := stage1Minimum(lane)
      stage2Maximum(lane) := stage1Maximum(lane)
    }
  }

  // P3: add the two affine terms before the common dynamic shift.
  private val stage3Valid = RegNext(stage2Valid, false.B)
  private val stage3Sum = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage3CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  private val stage3Bias = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage3Residual = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage3Minimum = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage3Maximum = Reg(Vec(p.dim, SInt(workBits.W)))
  for (lane <- 0 until p.dim) {
    when(stage2Valid) {
      stage3Sum(lane) := (stage2Term1(lane) + stage2Term2(lane)).asSInt
      stage3CommonShift(lane) := stage2CommonShift(lane)
      stage3Bias(lane) := stage2Bias(lane)
      stage3Residual(lane) := stage2Residual(lane)
      stage3Minimum(lane) := stage2Minimum(lane)
      stage3Maximum(lane) := stage2Maximum(lane)
    }
  }

  // P4: perform only the common dynamic shift and symmetric rounding. The
  // following 64-bit bias addition has its own carry-chain stage.
  private val stage4Valid = RegNext(stage3Valid, false.B)
  private val stage4Shifted = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage4Bias = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage4Residual = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage4Minimum = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage4Maximum = Reg(Vec(p.dim, SInt(workBits.W)))
  for (lane <- 0 until p.dim) {
    when(stage3Valid) {
      stage4Shifted(lane) :=
        shiftInteger(stage3Sum(lane), stage3CommonShift(lane))
      stage4Bias(lane) := stage3Bias(lane)
      stage4Residual(lane) := stage3Residual(lane)
      stage4Minimum(lane) := stage3Minimum(lane)
      stage4Maximum(lane) := stage3Maximum(lane)
    }
  }

  // P5: add the selected branch bias after the common shift.
  private val stage5Valid = RegNext(stage4Valid, false.B)
  private val stage5Delta = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage5Residual = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage5Minimum = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage5Maximum = Reg(Vec(p.dim, SInt(workBits.W)))
  for (lane <- 0 until p.dim) {
    when(stage4Valid) {
      stage5Delta(lane) :=
        (stage4Shifted(lane) + stage4Bias(lane))(workBits - 1, 0).asSInt
      stage5Residual(lane) := stage4Residual(lane)
      stage5Minimum(lane) := stage4Minimum(lane)
      stage5Maximum(lane) := stage4Maximum(lane)
    }
  }

  // P6: residual state update, saturation, and narrowing.
  private val outputRegister = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val outputValidRegister = RegNext(stage5Valid, false.B)
  for (lane <- 0 until p.dim) {
    val state = stage5Delta(lane) + stage5Residual(lane)
    val clamped = Mux(
      state < stage5Minimum(lane),
      stage5Minimum(lane),
      Mux(stage5Maximum(lane) < state, stage5Maximum(lane), state)
    )
    when(stage5Valid) {
      outputRegister(lane) := clamped(p.accumulatorBits - 1, 0).asSInt
    }
  }

  io.output := outputRegister
  io.outputValid := outputValidRegister
}
