package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Verifies the pipelined 1R1W accumulator contract. */
class VecAccumSpec extends AnyFreeSpec with Matchers {
  "independent rows must accept one accumulate request every cycle" in {
    val p = AccelParams(
      dim = 4,
      accumulatorBanks = 1,
      bankRows = 16,
      maxImageHeight = 2,
      maxImageWidth = 2,
      maxOutputBlocks = 1
    )

    simulate(new VecAccum(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.request.valid.poke(false.B)
      dut.io.response.ready.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      def drive(operation: AccumulatorOperation.Type, address: Int, base: Int): Unit = {
        dut.io.request.valid.poke(true.B)
        dut.io.request.bits.operation.poke(operation)
        dut.io.request.bits.address.poke(address.U)
        dut.io.request.bits.shift.poke(0.U)
        dut.io.request.bits.negate.poke(false.B)
        dut.io.request.bits.completesOutput.poke(false.B)
        dut.io.request.bits.block.poke(0.U)
        dut.io.request.bits.pixel.poke(0.U)
        dut.io.request.bits.outputX.poke(0.U)
        dut.io.request.bits.outputY.poke(0.U)
        for (lane <- 0 until p.dim) {
          dut.io.request.bits.data(lane).poke((base + lane).S)
        }
        dut.io.request.ready.expect(true.B)
        dut.clock.step()
      }

      // Initialize four distinct rows on consecutive cycles.
      for (row <- 0 until 4) {
        drive(AccumulatorOperation.write, row, 10 * row)
      }
      dut.io.request.valid.poke(false.B)
      dut.clock.step(3)

      var received = 0
      def collectResponse(): Unit = {
        if (dut.io.response.valid.peek().litToBoolean) {
          val row = received
          for (lane <- 0 until p.dim) {
            dut.io.response.bits.data(lane).expect(
              (100 + 20 * row + 2 * lane).S
            )
          }
          dut.io.response.bits.outputX.expect((row & 3).U)
          dut.io.response.bits.outputY.expect(((row + 1) & 3).U)
          received += 1
        }
      }

      // The previous writeback and the next read may overlap through the
      // independent ports. Consecutive rows are the normal convolution order.
      for (row <- 0 until 4) {
        dut.io.request.valid.poke(true.B)
        dut.io.request.bits.operation.poke(AccumulatorOperation.add)
        dut.io.request.bits.address.poke(row.U)
        dut.io.request.bits.shift.poke(0.U)
        dut.io.request.bits.negate.poke(false.B)
        dut.io.request.bits.completesOutput.poke(true.B)
        dut.io.request.bits.block.poke(0.U)
        dut.io.request.bits.pixel.poke(row.U)
        dut.io.request.bits.outputX.poke((row & 3).U)
        dut.io.request.bits.outputY.poke(((row + 1) & 3).U)
        for (lane <- 0 until p.dim) {
          dut.io.request.bits.data(lane).poke((100 + 10 * row + lane).S)
        }
        dut.io.request.ready.expect(true.B)
        collectResponse()
        dut.clock.step()
      }
      dut.io.request.valid.poke(false.B)

      var cycles = 0
      while (received < 4 && cycles < 12) {
        collectResponse()
        dut.clock.step()
        cycles += 1
      }
      received mustBe 4

      // A true same-row RMW dependency stalls for exactly the writeback
      // cycle, then proceeds with the updated value.
      def driveSameRowAdd(value: Int, expectReady: Boolean): Unit = {
        dut.io.request.valid.poke(true.B)
        dut.io.request.bits.operation.poke(AccumulatorOperation.add)
        dut.io.request.bits.address.poke(0.U)
        dut.io.request.bits.shift.poke(0.U)
        dut.io.request.bits.negate.poke(false.B)
        dut.io.request.bits.completesOutput.poke(false.B)
        dut.io.request.bits.block.poke(0.U)
        dut.io.request.bits.pixel.poke(0.U)
        for (lane <- 0 until p.dim) {
          dut.io.request.bits.data(lane).poke(value.S)
        }
        dut.io.request.ready.expect(expectReady.B)
        dut.clock.step()
      }

      driveSameRowAdd(1, expectReady = true)
      driveSameRowAdd(2, expectReady = false)
      driveSameRowAdd(2, expectReady = true)

      dut.io.request.bits.operation.poke(AccumulatorOperation.read)
      dut.io.request.bits.completesOutput.poke(true.B)
      dut.io.request.bits.pixel.poke(3.U)
      dut.io.request.bits.outputX.poke(3.U)
      dut.io.request.bits.outputY.poke(0.U)
      dut.io.request.ready.expect(false.B)
      dut.clock.step()
      dut.io.request.ready.expect(true.B)
      dut.clock.step()
      dut.io.request.valid.poke(false.B)

      var sawReadback = false
      cycles = 0
      while (!sawReadback && cycles < 8) {
        if (
          dut.io.response.valid.peek().litToBoolean &&
          dut.io.response.bits.completesOutput.peek().litToBoolean
        ) {
          dut.io.response.bits.pixel.expect(3.U)
          for (lane <- 0 until p.dim) {
            dut.io.response.bits.data(lane).expect((103 + 2 * lane).S)
          }
          sawReadback = true
        }
        dut.clock.step()
        cycles += 1
      }
      sawReadback mustBe true
    }
  }
}
