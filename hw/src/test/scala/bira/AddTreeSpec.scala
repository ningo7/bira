package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Unit tests for the shared pipelined reduction tree. */
class AddTreeSpec extends AnyFreeSpec with Matchers {
  "each operand count must enter at its matching binary-tree level" in {
    simulate(new AddTree(32)) { dut =>
      dut.reset.poke(true.B)
      dut.io.inputValid.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      for (count <- Seq(1, 2, 4, 8, 16)) {
        for (index <- 0 until 16) {
          val value = if (index < count) index - 7 else 1000 + index
          dut.io.inputs(index).poke(value.S)
        }
        dut.io.inputCount.poke(count.U)
        dut.io.inputValid.poke(true.B)
        dut.clock.step()
        dut.io.inputValid.poke(false.B)
        dut.clock.step()
        dut.io.outputValid.expect(true.B)
        dut.io.output.expect((0 until count).map(_ - 7).sum.S)
      }

      // The two stages must accept a new independent reduction every cycle.
      dut.clock.step()
      val backToBackValues = Seq(3, -5, 11)
      backToBackValues.zipWithIndex.foreach {
        case (value, transaction) =>
          dut.io.inputCount.poke(4.U)
          for (index <- 0 until 16) {
            dut.io.inputs(index).poke(
              (if (index < 4) value else 1000).S
            )
          }
          dut.io.inputValid.poke(true.B)
          dut.clock.step()
          if (transaction > 0) {
            dut.io.outputValid.expect(true.B)
            dut.io.output.expect(
              (backToBackValues(transaction - 1) * 4).S
            )
          }
      }
      dut.io.inputValid.poke(false.B)
      dut.clock.step()
      dut.io.outputValid.expect(true.B)
      dut.io.output.expect((backToBackValues.last * 4).S)
    }
  }
}
