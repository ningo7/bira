package bira

// Binary post-processing pipeline.

import chisel3._

object BinPostProc {
  val latency = 9
}

/** Nine-stage branch-selected binary affine/RPReLU and residual pipeline.
  *
  *   dot   = accumulator = 2 * popcount - N
  *   delta = selectedAffine(dot)
  *   state = clamp(residual + delta, qMin, qMax)
  *
  * Threshold selection precedes the affine datapath, so only one branch is
  * evaluated per lane. Dynamic rounded shifts are split into preparation and
  * completion stages so their carry chains and barrel shifters never share a
  * cycle. The pipeline accepts one vector every cycle.
  */
class BinPostProc(p: AccelParams) extends Module {
  private val workBits = p.accumulatorBits * 2

  val io = IO(new Bundle {
    val inputValid = Input(Bool())
    val accumulator = Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
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

  private def prepareShift(
    value: SInt,
    shift: SInt,
    rightBits: Int = workBits
  ): (SInt, SInt, UInt, Bool) = {
    val extended = value.pad(workBits)
    val rightExtended = value.pad(rightBits)
    val shiftIsLeft = !shift(shift.getWidth - 1)
    val leftAmount = clippedAmount(shift.asUInt)
    val rightAmount = clippedAmount((-shift).asUInt)
    val shiftedLeft =
      (extended << leftAmount)(workBits - 1, 0).asSInt
    val rounding = Mux(
      rightAmount === 0.U,
      0.U(rightBits.W),
      (1.U(rightBits.W) << (rightAmount - 1.U))(rightBits - 1, 0)
    )
    // Symmetric round-to-nearest can be expressed without absolute-value
    // and sign-restoration carry chains. For a negative value, adding
    // rounding-1 before an arithmetic shift is equivalent to negating the
    // rounded magnitude.
    val negativeAdjustment = Mux(
      rightAmount === 0.U,
      0.U(rightBits.W),
      rounding - 1.U
    )
    val adjustment = Mux(
      rightExtended < 0.S,
      negativeAdjustment,
      rounding
    )
    val adjustedNarrow =
      (rightExtended + adjustment.zext)(rightBits - 1, 0).asSInt
    (shiftedLeft, adjustedNarrow, rightAmount, shiftIsLeft)
  }

  private def finishShift(
    shiftedLeft: SInt,
    adjustedRight: SInt,
    rightAmount: UInt,
    shiftIsLeft: Bool
  ): SInt = {
    val shiftedRight = adjustedRight.pad(workBits) >> rightAmount
    Mux(
      shiftIsLeft,
      shiftedLeft,
      shiftedRight(workBits - 1, 0).asSInt
    )
  }

  private def applyCoefficient(value: SInt, coefficient: SInt): SInt =
    Mux(
      coefficient === 1.S,
      value,
      Mux(coefficient === (-1).S, -value, 0.S(workBits.W))
    )

  /** Explicit staged delay used for wide post-processing sidebands.
    *
    * Each lane has stable physical stage boundaries and no shared valid-based
    * clock enable. Downstream synthesis may still choose FF or SRL storage,
    * but it no longer has to route a high-fanout enable across the datapath.
    */
  private def registerDelay[T <: Data](
    input: T,
    cycles: Int,
    name: String
  ): T = {
    require(cycles > 0)
    var delayed = input
    for (stage <- 0 until cycles) {
      val next = Reg(chiselTypeOf(input))
      next.suggestName(s"${name}_$stage")
      next := delayed
      dontTouch(next)
      delayed = next
    }
    delayed
  }

  // P0: isolate the initialized Accumulator output from branch selection.
  // Binary correction (-N) is already present as the Accumulator bias.
  private val stage0Valid = RegNext(io.inputValid, false.B)
  private val stage0Dot = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage0Residual = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage0Parameters = Reg(Vec(p.dim, new BinPostParams(p)))
  for (lane <- 0 until p.dim) {
    // Validity is tracked separately. Datapath registers intentionally update
    // every cycle so valid never becomes a high-fanout clock-enable network.
    stage0Residual(lane) := io.residual(lane)
    stage0Dot(lane) := io.accumulator(lane)
    stage0Parameters(lane) := io.parameters(lane)
  }

  // P1: select the active branch and prepare its two rounded shifts. The
  // carry-chain adjustment and left barrel shift are registered before the
  // independent right barrel shift in P2.
  private val stage1Valid = RegNext(stage0Valid, false.B)
  private val stage1Left1 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage1Left2 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val initialRightBits = p.accumulatorBits + 1
  private val stage1Adjusted1 =
    Reg(Vec(p.dim, SInt(initialRightBits.W)))
  private val stage1Adjusted2 =
    Reg(Vec(p.dim, SInt(initialRightBits.W)))
  private val stage1RightAmount1 = Reg(Vec(p.dim, UInt(p.postShiftAmountBits.W)))
  private val stage1RightAmount2 = Reg(Vec(p.dim, UInt(p.postShiftAmountBits.W)))
  private val stage1IsLeft1 = Reg(Vec(p.dim, Bool()))
  private val stage1IsLeft2 = Reg(Vec(p.dim, Bool()))
  private val stage1Coefficient1 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage1Coefficient2 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage1CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  // Capture branch-selected sidebands beside the P1 selection logic before
  // entering their compact delay chains.  Driving an inferred SRL directly
  // from this selection cone lets placement separate the carry/mux logic and
  // the SRL input by a large distance, making routing dominate the path.
  private val stage1Bias = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage1Minimum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage1Maximum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val selectedBias = Wire(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val selectedMinimum = Wire(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val selectedMaximum = Wire(Vec(p.dim, SInt(p.accumulatorBits.W)))
  for (lane <- 0 until p.dim) {
    val value = stage0Dot(lane)
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
    selectedBias(lane) := bias
    selectedMinimum(lane) := params.qMin
    selectedMaximum(lane) := params.qMax
    // A 32-bit dot plus at most a 2^30 rounding term fits exactly in 33
    // signed bits. The remaining high bits are sign extension, not arithmetic.
    val shift1 = prepareShift(value, leftShift1.zext, initialRightBits)
    val shift2 = prepareShift(value, leftShift2.zext, initialRightBits)
    // Invalid-cycle values are ignored by the independent valid pipeline.
    stage1Bias(lane) := selectedBias(lane)
    stage1Minimum(lane) := selectedMinimum(lane)
    stage1Maximum(lane) := selectedMaximum(lane)
    stage1Left1(lane) := shift1._1
    stage1Adjusted1(lane) := shift1._2
    stage1RightAmount1(lane) := shift1._3
    stage1IsLeft1(lane) := shift1._4
    stage1Left2(lane) := shift2._1
    stage1Adjusted2(lane) := shift2._2
    stage1RightAmount2(lane) := shift2._3
    stage1IsLeft2(lane) := shift2._4
    stage1Coefficient1(lane) := coefficient1
    stage1Coefficient2(lane) := coefficient2
    stage1CommonShift(lane) := commonShift
  }

  // Delay each lane independently without a shared valid-based clock enable.
  private val stage6Bias = VecInit.tabulate(p.dim) { lane =>
    registerDelay(stage1Bias(lane), 5, s"bias_${lane}")
  }
  private val stage7Residual = VecInit.tabulate(p.dim) { lane =>
    registerDelay(stage0Residual(lane), 7, s"residual_${lane}")
  }
  private val stage7Minimum = VecInit.tabulate(p.dim) { lane =>
    registerDelay(stage1Minimum(lane), 6, s"minimum_${lane}")
  }
  private val stage7Maximum = VecInit.tabulate(p.dim) { lane =>
    registerDelay(stage1Maximum(lane), 6, s"maximum_${lane}")
  }

  // P2: finish both branch shifts with only a barrel shift and result mux.
  private val stage2Valid = RegNext(stage1Valid, false.B)
  private val stage2Shifted1 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2Shifted2 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2Coefficient1 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage2Coefficient2 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage2CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  for (lane <- 0 until p.dim) {
    stage2Shifted1(lane) := finishShift(
      stage1Left1(lane),
      stage1Adjusted1(lane),
      stage1RightAmount1(lane),
      stage1IsLeft1(lane)
    )
    stage2Shifted2(lane) := finishShift(
      stage1Left2(lane),
      stage1Adjusted2(lane),
      stage1RightAmount2(lane),
      stage1IsLeft2(lane)
    )
    stage2Coefficient1(lane) := stage1Coefficient1(lane)
    stage2Coefficient2(lane) := stage1Coefficient2(lane)
    stage2CommonShift(lane) := stage1CommonShift(lane)
  }

  // P3: apply the selected {-1, 0, +1} coefficients.
  private val stage3Valid = RegNext(stage2Valid, false.B)
  private val stage3Term1 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage3Term2 = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage3CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  for (lane <- 0 until p.dim) {
    stage3Term1(lane) :=
      applyCoefficient(stage2Shifted1(lane), stage2Coefficient1(lane))
    stage3Term2(lane) :=
      applyCoefficient(stage2Shifted2(lane), stage2Coefficient2(lane))
    stage3CommonShift(lane) := stage2CommonShift(lane)
  }

  // P4: add the two affine terms before the common dynamic shift.
  private val stage4Valid = RegNext(stage3Valid, false.B)
  private val stage4Sum = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage4CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  for (lane <- 0 until p.dim) {
    stage4Sum(lane) := (stage3Term1(lane) + stage3Term2(lane)).asSInt
    stage4CommonShift(lane) := stage3CommonShift(lane)
  }

  // P5: prepare the common rounded shift.
  private val stage5Valid = RegNext(stage4Valid, false.B)
  private val stage5Left = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage5Adjusted = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage5RightAmount = Reg(Vec(p.dim, UInt(p.postShiftAmountBits.W)))
  private val stage5IsLeft = Reg(Vec(p.dim, Bool()))
  for (lane <- 0 until p.dim) {
    val shifted = prepareShift(stage4Sum(lane), stage4CommonShift(lane))
    stage5Left(lane) := shifted._1
    stage5Adjusted(lane) := shifted._2
    stage5RightAmount(lane) := shifted._3
    stage5IsLeft(lane) := shifted._4
  }

  // P6: finish the common shift.
  private val stage6Valid = RegNext(stage5Valid, false.B)
  private val stage6Shifted = Reg(Vec(p.dim, SInt(workBits.W)))
  for (lane <- 0 until p.dim) {
    stage6Shifted(lane) := finishShift(
      stage5Left(lane),
      stage5Adjusted(lane),
      stage5RightAmount(lane),
      stage5IsLeft(lane)
    )
  }

  // P7: add the selected branch bias.
  private val stage7Valid = RegNext(stage6Valid, false.B)
  private val stage7Delta = Reg(Vec(p.dim, SInt(workBits.W)))
  for (lane <- 0 until p.dim) {
    stage7Delta(lane) :=
      (stage6Shifted(lane) + stage6Bias(lane).pad(workBits))(
        workBits - 1,
        0
      ).asSInt
  }

  // P8: residual state update, saturation, and narrowing.
  private val outputRegister = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val outputValidRegister = RegNext(stage7Valid, false.B)
  for (lane <- 0 until p.dim) {
    val residual = stage7Residual(lane).pad(workBits)
    val minimum = stage7Minimum(lane).pad(workBits)
    val maximum = stage7Maximum(lane).pad(workBits)
    val state = stage7Delta(lane) + residual
    val clamped = Mux(
      state < minimum,
      minimum,
      Mux(maximum < state, maximum, state)
    )
    outputRegister(lane) := clamped(p.accumulatorBits - 1, 0).asSInt
  }

  io.output := outputRegister
  io.outputValid := outputValidRegister
}
