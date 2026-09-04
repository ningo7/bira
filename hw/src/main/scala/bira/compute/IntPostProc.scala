package bira

// Integer post-processing pipeline.

import chisel3._

/** Fully pipelined branch-selected vector PReLU/requantization datapath.
  *
  * Each stage accepts a new vector every cycle.  Wide variable shifts and
  * carry chains are kept in separate stages so the parameter-bank mux, two
  * affine terms, common rounded shift, and saturation never form one path.
  */
class IntPostProc(p: AccelParams) extends Module {
  private val workBits = p.accumulatorBits * 2
  private val amountBits = p.postShiftAmountBits

  val io = IO(new Bundle {
    val inputValid = Input(Bool())
    val input = Input(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val parameters = Input(Vec(p.dim, new PostProcessParameters(p)))
    val outputValid = Output(Bool())
    val output = Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
  })

  /** Return a physically bounded shift amount.
    *
    * Keeping this result at `postShiftAmountBits` is important: retaining the
    * eight-bit signed-shift width makes FIRRTL construct a 0..255 barrel
    * shifter even though the architectural maximum is only 31.
    */
  private def clippedAmount(amount: UInt): UInt = {
    val result = Wire(UInt(amountBits.W))
    result := Mux(
      amount > p.maxPostProcessLeftShift.U,
      p.maxPostProcessLeftShift.U,
      amount(amountBits - 1, 0)
    )
    result
  }

  // P0: resolve the sign-dependent parameter branch and register the narrow
  // controls.  This isolates the parameter-bank selection mux from shifts.
  private val stage0Valid = RegNext(io.inputValid, false.B)
  private val stage0Value = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage0Coefficient1 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage0Coefficient2 = Reg(Vec(p.dim, SInt(2.W)))
  private val stage0LeftShift1 = Reg(Vec(p.dim, UInt(amountBits.W)))
  private val stage0LeftShift2 = Reg(Vec(p.dim, UInt(amountBits.W)))
  private val stage0CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  private val stage0Minimum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage0Maximum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  for (lane <- 0 until p.dim) {
    val positive = io.input(lane) >= 0.S
    val params = io.parameters(lane)
    when(io.inputValid) {
      stage0Value(lane) := io.input(lane)
      stage0Coefficient1(lane) :=
        Mux(positive, 1.S(2.W), params.negativeCoeff1)
      stage0Coefficient2(lane) :=
        Mux(positive, 0.S(2.W), params.negativeCoeff2)
      stage0LeftShift1(lane) :=
        Mux(positive, 0.U, params.negativeLeftShift1)
      stage0LeftShift2(lane) :=
        Mux(positive, 0.U, params.negativeLeftShift2)
      stage0CommonShift(lane) :=
        Mux(positive, params.positiveShift, params.negativeCommonShift)
      stage0Minimum(lane) := params.qMin
      stage0Maximum(lane) := params.qMax
    }
  }

  // P1: perform the two bounded left shifts.  Negative coefficients use
  // one's complement here; their +1 corrections are folded into one adder in
  // P2, avoiding a negate carry chain before the affine addition.
  private val stage1Valid = RegNext(stage0Valid, false.B)
  private val stage1Operand1 = Reg(Vec(p.dim, UInt(workBits.W)))
  private val stage1Operand2 = Reg(Vec(p.dim, UInt(workBits.W)))
  private val stage1Correction = Reg(Vec(p.dim, UInt(2.W)))
  private val stage1CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  private val stage1Minimum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage1Maximum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  for (lane <- 0 until p.dim) {
    val value = stage0Value(lane).pad(workBits)
    val shifted1 =
      (value << stage0LeftShift1(lane))(workBits - 1, 0)
    val shifted2 =
      (value << stage0LeftShift2(lane))(workBits - 1, 0)
    val coefficient1 = stage0Coefficient1(lane)
    val coefficient2 = stage0Coefficient2(lane)
    val negative1 = coefficient1 === (-1).S
    val negative2 = coefficient2 === (-1).S
    val operand1 = Mux(
      coefficient1 === 1.S,
      shifted1,
      Mux(negative1, ~shifted1, 0.U(workBits.W))
    )
    val operand2 = Mux(
      coefficient2 === 1.S,
      shifted2,
      Mux(negative2, ~shifted2, 0.U(workBits.W))
    )
    when(stage0Valid) {
      stage1Operand1(lane) := operand1
      stage1Operand2(lane) := operand2
      stage1Correction(lane) := negative1.asUInt +& negative2.asUInt
      stage1CommonShift(lane) := stage0CommonShift(lane)
      stage1Minimum(lane) := stage0Minimum(lane)
      stage1Maximum(lane) := stage0Maximum(lane)
    }
  }

  // P2: one affine carry chain shared by both signed {-1, 0, 1} terms.
  private val stage2Valid = RegNext(stage1Valid, false.B)
  private val stage2Value = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage2CommonShift = Reg(Vec(p.dim, SInt(p.shiftBits.W)))
  private val stage2Minimum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage2Maximum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  for (lane <- 0 until p.dim) {
    val affine =
      stage1Operand1(lane) + stage1Operand2(lane) +
        stage1Correction(lane).pad(workBits)
    when(stage1Valid) {
      stage2Value(lane) := affine(workBits - 1, 0).asSInt
      stage2CommonShift(lane) := stage1CommonShift(lane)
      stage2Minimum(lane) := stage1Minimum(lane)
      stage2Maximum(lane) := stage1Maximum(lane)
    }
  }

  // P3: prepare symmetric round-to-nearest for a right shift.  For a
  // negative two's-complement value the exact adjustment is rounding - 1;
  // this replaces abs/add/negate with a single carry chain.
  private val stage3Valid = RegNext(stage2Valid, false.B)
  private val stage3Value = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage3ShiftLeft = Reg(Vec(p.dim, Bool()))
  private val stage3Amount = Reg(Vec(p.dim, UInt(amountBits.W)))
  private val stage3Minimum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage3Maximum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  for (lane <- 0 until p.dim) {
    val commonShift = stage2CommonShift(lane)
    val shiftLeft = !commonShift(p.shiftBits - 1)
    val leftAmount = clippedAmount(commonShift.asUInt)
    val rightAmount = clippedAmount((-commonShift).asUInt)
    val rounding = Mux(
      rightAmount === 0.U,
      0.U(workBits.W),
      (1.U(workBits.W) << (rightAmount - 1.U))(workBits - 1, 0)
    )
    val adjustment = Mux(
      rightAmount === 0.U,
      0.U(workBits.W),
      Mux(stage2Value(lane) < 0.S, rounding - 1.U, rounding)
    )
    val rounded =
      (stage2Value(lane) + adjustment.zext)(workBits - 1, 0).asSInt
    when(stage2Valid) {
      stage3Value(lane) := Mux(shiftLeft, stage2Value(lane), rounded)
      stage3ShiftLeft(lane) := shiftLeft
      stage3Amount(lane) := Mux(shiftLeft, leftAmount, rightAmount)
      stage3Minimum(lane) := stage2Minimum(lane)
      stage3Maximum(lane) := stage2Maximum(lane)
    }
  }

  // P4: the common bounded barrel shift is now independent of rounding.
  private val stage4Valid = RegNext(stage3Valid, false.B)
  private val stage4Value = Reg(Vec(p.dim, SInt(workBits.W)))
  private val stage4Minimum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val stage4Maximum = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  for (lane <- 0 until p.dim) {
    val shiftedLeft =
      (stage3Value(lane) << stage3Amount(lane))(workBits - 1, 0).asSInt
    val shiftedRight = stage3Value(lane) >> stage3Amount(lane)
    when(stage3Valid) {
      stage4Value(lane) :=
        Mux(stage3ShiftLeft(lane), shiftedLeft, shiftedRight)
      stage4Minimum(lane) := stage3Minimum(lane)
      stage4Maximum(lane) := stage3Maximum(lane)
    }
  }

  // P5: signed saturation and narrowing.
  private val outputRegister = Reg(Vec(p.dim, SInt(p.accumulatorBits.W)))
  private val outputValidRegister = RegNext(stage4Valid, false.B)
  for (lane <- 0 until p.dim) {
    val minimum = stage4Minimum(lane).pad(workBits)
    val maximum = stage4Maximum(lane).pad(workBits)
    val clamped = Mux(
      stage4Value(lane) < minimum,
      minimum,
      Mux(stage4Value(lane) > maximum, maximum, stage4Value(lane))
    )
    when(stage4Valid) {
      outputRegister(lane) := clamped(p.accumulatorBits - 1, 0).asSInt
    }
  }

  io.output := outputRegister
  io.outputValid := outputValidRegister
}
