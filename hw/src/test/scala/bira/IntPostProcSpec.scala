package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Unit tests for multi-bit post-processing. */
class IntPostProcSpec extends AnyFreeSpec with Matchers {
  "PReLU must match symmetric rounded shifts and signed saturation" in {
    val p = BiRaParams(
      dim = 4,
      bankRows = 16,
      maxImageHeight = 2,
      maxImageWidth = 2,
      maxOutputBlocks = 1
    )

    simulate(new IntPostProc(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.inputValid.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      val inputs = Seq(7, -7, 100, -100)
      for (lane <- 0 until p.dim) {
        dut.io.input(lane).poke(inputs(lane).S)
        dut.io.parameters(lane).positiveShift.poke(0.S)
        dut.io.parameters(lane).negativeCoeff1.poke(1.S)
        dut.io.parameters(lane).negativeCoeff2.poke(0.S)
        dut.io.parameters(lane).negativeLeftShift1.poke(0.U)
        dut.io.parameters(lane).negativeLeftShift2.poke(0.U)
        dut.io.parameters(lane).negativeCommonShift.poke(0.S)
        dut.io.parameters(lane).qMin.poke((-32).S)
        dut.io.parameters(lane).qMax.poke(31.S)
      }

      dut.io.parameters(0).positiveShift.poke(1.S)
      dut.io.parameters(1).negativeCommonShift.poke((-1).S)

      dut.io.inputValid.poke(true.B)
      dut.clock.step()
      dut.io.inputValid.poke(false.B)
      dut.clock.step()
      dut.io.outputValid.expect(true.B)
      val expected = Seq(14, -4, 31, -32)
      for (lane <- 0 until p.dim) {
        dut.io.output(lane).expect(expected(lane).S)
      }
    }
  }
}
