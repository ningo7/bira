package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Checks the frozen 256-bit lane-record ABI in both interpretations. */
class BiRaParameterBufferSpec extends AnyFreeSpec with Matchers {
  private def bits(value: BigInt, width: Int): BigInt =
    value & ((BigInt(1) << width) - 1)

  private def field(value: BigInt, low: Int, width: Int): BigInt =
    bits(value, width) << low

  private def packMulti(lane: Int): BigInt = {
    val bias = if (lane % 2 == 0) -100 - lane else 100 + lane
    val positiveShift = -8 + lane
    val qMin = -1000 - lane
    val qMax = 2000 + lane
    val threshold = -300 + lane

    field(bias, 0, 32) |
      field(positiveShift, 32, 8) |
      field(-1, 40, 2) |
      field(1, 42, 2) |
      field(3 + lane, 44, 5) |
      field(4 + lane, 49, 5) |
      field(-12 + lane, 54, 8) |
      field(qMin, 62, 32) |
      field(qMax, 94, 32) |
      field(threshold, 126, 32)
  }

  private def packBinary(lane: Int): BigInt = {
    field(-40 + lane, 0, 32) |
      field(-1, 32, 2) |
      field(2 + lane, 34, 5) |
      field(3 + lane, 39, 5) |
      field(-9 + lane, 44, 8) |
      field(500 + lane, 52, 32) |
      field(1, 84, 2) |
      field(-1, 86, 2) |
      field(4 + lane, 88, 5) |
      field(5 + lane, 93, 5) |
      field(-7 + lane, 98, 8) |
      field(-600 - lane, 106, 32) |
      field(-700 - lane, 138, 32) |
      field(800 + lane, 170, 32) |
      field(900 + lane, 202, 32)
  }

  "raw DMA rows decode into multi-bit and binary parameter blocks" in {
    val p = BiRaParams(
      dim = 4,
      fullBanks = 4,
      binaryBanks = 2,
      accumulatorBanks = 2,
      bankRows = 16,
      parameterRows = 32,
      maxImageHeight = 4,
      maxImageWidth = 4,
      maxInputChannels = 16,
      maxOutputBlocks = 4
    )

    simulate(new BiRaParameterBuffer(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.write.valid.poke(false.B)
      dut.io.readRequest.valid.poke(false.B)
      dut.io.readResponse.ready.poke(false.B)
      dut.io.correctionReadRequest.valid.poke(false.B)
      dut.io.correctionReadResponse.ready.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      def writeBlock(baseRow: Int, records: Seq[BigInt]): Unit = {
        for (row <- 0 until p.parameterRowsPerBlock) {
          val packed = records(2 * row) | (records(2 * row + 1) << 256)
          dut.io.write.bits.address.poke((baseRow + row).U)
          dut.io.write.bits.data.poke(packed.U)
          dut.io.write.valid.poke(true.B)
          dut.io.write.ready.expect(true.B)
          dut.clock.step()
        }
        dut.io.write.valid.poke(false.B)
      }

      def requestBlock(baseRow: Int, block: Int): Unit = {
        dut.io.readRequest.bits.baseRow.poke(baseRow.U)
        dut.io.readRequest.bits.block.poke(block.U)
        dut.io.readRequest.valid.poke(true.B)
        dut.io.readRequest.ready.expect(true.B)
        dut.clock.step()
        dut.io.readRequest.valid.poke(false.B)

        var cycles = 0
        while (!dut.io.readResponse.valid.peek().litToBoolean && cycles < 20) {
          dut.clock.step()
          cycles += 1
        }
        dut.io.readResponse.valid.expect(true.B)
      }

      writeBlock(baseRow = 0, (0 until p.dim).map(packMulti))
      requestBlock(baseRow = 0, block = 0)
      for (lane <- 0 until p.dim) {
        val expectedBias = if (lane % 2 == 0) -100 - lane else 100 + lane
        dut.io.readResponse.bits.multiBit.bias(lane)
          .expect(expectedBias.S)
        dut.io.readResponse.bits.multiBit.post(lane).positiveShift
          .expect((-8 + lane).S)
        dut.io.readResponse.bits.multiBit.post(lane).negativeCoeff1
          .expect((-1).S)
        dut.io.readResponse.bits.multiBit.post(lane).negativeCoeff2
          .expect(1.S)
        dut.io.readResponse.bits.multiBit.post(lane).qMin
          .expect((-1000 - lane).S)
        dut.io.readResponse.bits.multiBit.post(lane).qMax
          .expect((2000 + lane).S)
        dut.io.readResponse.bits.multiBit.binaryThreshold(lane)
          .expect((-300 + lane).S)
      }
      dut.io.readResponse.ready.poke(true.B)
      dut.clock.step()
      dut.io.readResponse.ready.poke(false.B)

      writeBlock(baseRow = 8, (0 until p.dim).map(packBinary))
      for (lane <- 0 until p.dim) {
        ((packBinary(lane) >> 86) & 3) mustBe 3
      }
      requestBlock(baseRow = 8, block = 0)
      for (lane <- 0 until p.dim) {
        val post = dut.io.readResponse.bits.binary.post(lane)
        withClue(s"binary lane $lane: ") {
          post.threshold.expect((-40 + lane).S)
          post.positiveCoeff2.expect((-1).S)
          post.positiveBias.expect((500 + lane).S)
          post.negativeCoeff1.expect(1.S)
          post.negativeCoeff2.expect((-1).S)
          post.negativeBias.expect((-600 - lane).S)
          post.qMin.expect((-700 - lane).S)
          post.qMax.expect((800 + lane).S)
          dut.io.readResponse.bits.binary.outputSignThreshold(lane)
            .expect((900 + lane).S)
        }
      }
      dut.io.readResponse.ready.poke(true.B)
      dut.clock.step()
      dut.io.readResponse.ready.poke(false.B)

      val correctionRow = (0 until p.correctionEntriesPerRow).foldLeft(
        BigInt(0)
      ) {
        case (row, entry) =>
          row | (
            bits(-100 - entry, p.accumulatorBits) <<
              (entry * p.accumulatorBits)
          )
      }
      dut.io.write.bits.address.poke(20.U)
      dut.io.write.bits.data.poke(correctionRow.U)
      dut.io.write.valid.poke(true.B)
      dut.clock.step()
      dut.io.write.valid.poke(false.B)
      dut.io.correctionReadRequest.bits.address.poke(20.U)
      dut.io.correctionReadRequest.valid.poke(true.B)
      dut.io.correctionReadRequest.ready.expect(true.B)
      dut.clock.step()
      dut.io.correctionReadRequest.valid.poke(false.B)
      dut.io.correctionReadResponse.ready.poke(true.B)
      while (!dut.io.correctionReadResponse.valid.peek().litToBoolean) {
        dut.clock.step()
      }
      dut.io.correctionReadResponse.bits.expect(correctionRow.U)
      dut.clock.step()
    }
  }
}
