package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Unit tests for fused binary residual post-processing. */
class BinPostProcSpec
    extends AnyFreeSpec
    with Matchers {
  "fused binary branches must add residual and saturate signed state" in {
    val p = AccelParams(
      dim = 4,
      fullBanks = 3,
      binaryBanks = 2,
      accumulatorBanks = 2,
      bankRows = 16,
      maxImageHeight = 2,
      maxImageWidth = 2,
      maxInputChannels = 16,
      maxOutputBlocks = 1
    )

    simulate(new BinPostProc(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.inputValid.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      // The binary correction is already folded into the Accumulator values.
      val accumulator = Seq(4, -9, 96, -104)
      val residual = Seq(3, 7, 80, -80)

      for (lane <- 0 until p.dim) {
        dut.io.accumulator(lane).poke(accumulator(lane).S)
        dut.io.residual(lane).poke(residual(lane).S)
        val parameters = dut.io.parameters(lane)
        parameters.threshold.poke(0.S)
        parameters.positiveCoeff2.poke(1.S)
        parameters.positiveLeftShift1.poke(0.U)
        parameters.positiveLeftShift2.poke(1.U)
        parameters.positiveCommonShift.poke((-1).S)
        parameters.positiveBias.poke(1.S)
        parameters.negativeCoeff1.poke((-1).S)
        parameters.negativeCoeff2.poke(0.S)
        parameters.negativeLeftShift1.poke(0.U)
        parameters.negativeLeftShift2.poke(0.U)
        parameters.negativeCommonShift.poke(0.S)
        parameters.negativeBias.poke((-2).S)
        parameters.qMin.poke((-128).S)
        parameters.qMax.poke(127.S)
      }

      // positive: round(3*x/2)+1; negative: -x-2; then residual add
      dut.io.inputValid.poke(true.B)
      dut.clock.step()
      dut.io.inputValid.poke(false.B)
      while (!dut.io.outputValid.peek().litToBoolean) {
        dut.clock.step()
      }
      dut.io.outputValid.expect(true.B)
      Seq(10, 14, 127, 22).zipWithIndex.foreach {
        case (expected, lane) =>
          dut.io.output(lane).expect(expected.S)
      }
    }
  }
}
