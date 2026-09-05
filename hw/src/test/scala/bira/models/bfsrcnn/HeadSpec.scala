package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class HeadSpec extends AnyFreeSpec with Matchers {
  private def signedByte(value: BigInt): Int = {
    val raw = value.toInt & 0xff
    if (raw >= 128) raw - 256 else raw
  }

  "standalone control must execute a padded uint8 x int16 head convolution" in {
    val p = AccelParams(
      dim = 4,
      fullBanks = 5,
      binaryBanks = 2,
      accumulatorBanks = 1,
      bankRows = 32,
      maxImageHeight = 2,
      maxImageWidth = 3,
      maxOutputBlocks = 1
    )

    val height = 2
    val width = 3
    val input = Seq(
      1, 2, 3,
      4, 5, 6
    )
    val weights = Seq.tabulate(p.kernelElements, p.dim) {
      case (tap, lane) => ((tap + 2 * lane) % 5) - 2
    }
    val bias = Seq(1, -2, 3, -4)

    val expected = Seq.tabulate(height * width, p.dim) {
      case (pixel, lane) =>
        val y = pixel / width
        val x = pixel % width
        var accumulator = bias(lane)
        for {
          ky <- 0 until p.kernelSize
          kx <- 0 until p.kernelSize
        } {
          val iy = y + ky - p.kernelSize / 2
          val ix = x + kx - p.kernelSize / 2
          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            val tap = ky * p.kernelSize + kx
            accumulator += input(iy * width + ix) * weights(tap)(lane)
          }
        }
        accumulator.max(-128).min(127)
    }

    val inputBase = 0
    val weightLowBase = p.bankRows
    val weightHighBase = 2 * p.bankRows
    val outputBase = 3 * p.bankRows

    simulate(new Core(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.command.valid.poke(false.B)
      dut.io.binaryCommand.valid.poke(false.B)
      dut.io.parameterWrite.valid.poke(false.B)
      dut.io.binaryParameterWrite.valid.poke(false.B)
      dut.io.accumulatorWrite.valid.poke(false.B)
      dut.io.fullWrite.valid.poke(false.B)
      dut.io.fullReadRequest.valid.poke(false.B)
      dut.io.binaryWrite.valid.poke(false.B)
      dut.io.binaryReadRequest.valid.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      def writeFullRow(address: Int, values: Seq[Int]): Unit = {
        require(values.length == p.dim)
        dut.io.fullWrite.bits.address.poke(address.U)
        for (lane <- 0 until p.dim) {
          dut.io.fullWrite.bits.data(lane).poke((values(lane) & 0xff).U)
        }
        dut.io.fullWrite.valid.poke(true.B)
        while (!dut.io.fullWrite.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.fullWrite.valid.poke(false.B)
      }

      input.grouped(p.dim).zipWithIndex.foreach {
        case (values, row) =>
          writeFullRow(inputBase + row, values.padTo(p.dim, 0))
      }

      for (tap <- 0 until p.kernelElements) {
        writeFullRow(
          weightLowBase + tap,
          weights(tap).map(_ & 0xff)
        )
        writeFullRow(
          weightHighBase + tap,
          weights(tap).map(value => (value >> 8) & 0xff)
        )
      }

      dut.io.parameterWrite.bits.block.poke(0.U)
      for (lanePair <- 0 until p.parameterRowsPerBlock) {
        dut.io.parameterWrite.bits.lanePair.poke(lanePair.U)
        for (laneInRow <- 0 until 2) {
          val lane = lanePair * 2 + laneInRow
          dut.io.parameterWrite.bits.bias(laneInRow).poke(bias(lane).S)
          val post = dut.io.parameterWrite.bits.post(laneInRow)
          post.positiveShift.poke(0.S)
          post.negativeCoeff1.poke(1.S)
          post.negativeCoeff2.poke(0.S)
          post.negativeLeftShift1.poke(0.U)
          post.negativeLeftShift2.poke(0.U)
          post.negativeCommonShift.poke(0.S)
          post.qMin.poke((-128).S)
          post.qMax.poke(127.S)
          dut.io.parameterWrite.bits.binaryThreshold(laneInRow).poke(0.S)
        }
        dut.io.parameterWrite.valid.poke(true.B)
        while (!dut.io.parameterWrite.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.parameterWrite.valid.poke(false.B)
      }

      dut.io.command.bits.inputBase.poke(inputBase.U)
      dut.io.command.bits.weightLowBase.poke(weightLowBase.U)
      dut.io.command.bits.weightHighBase.poke(weightHighBase.U)
      dut.io.command.bits.outputBase.poke(outputBase.U)
      dut.io.command.bits.binaryOutputBase.poke(0.U)
      dut.io.command.bits.accumulatorBase.poke(0.U)
      dut.io.command.bits.residualBase.poke(0.U)
      dut.io.command.bits.inputHeight.poke(height.U)
      dut.io.command.bits.inputWidth.poke(width.U)
      dut.io.command.bits.outputHeight.poke(height.U)
      dut.io.command.bits.outputWidth.poke(width.U)
      dut.io.command.bits.inputChannels.poke(1.U)
      dut.io.command.bits.outputBlocks.poke(1.U)
      dut.io.command.bits.kernelHeight.poke(p.kernelSize.U)
      dut.io.command.bits.kernelWidth.poke(p.kernelSize.U)
      dut.io.command.bits.paddingY.poke((p.kernelSize / 2).U)
      dut.io.command.bits.paddingX.poke((p.kernelSize / 2).U)
      dut.io.command.bits.weightPrecision.poke(16.U)
      dut.io.command.bits.activationPrecision.poke(8.U)
      dut.io.command.bits.inputSigned.poke(false.B)
      dut.io.command.bits.depthwise.poke(false.B)
      dut.io.command.bits.columnReduce.poke(false.B)
      dut.io.command.bits.bilinearResidual.poke(false.B)
      dut.io.command.bits.shufflePack2.poke(false.B)
      dut.io.command.bits.shuffleLog2.poke(0.U)
      dut.io.command.bits.writeFullOutput.poke(true.B)
      dut.io.command.bits.writeBinaryOutput.poke(false.B)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peek().litToBoolean) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      var cycles = 0
      while (!dut.io.status.done.peek().litToBoolean && cycles < 4000) {
        dut.clock.step()
        cycles += 1
      }
      withClue(s"head layer timed out after $cycles cycles") {
        dut.io.status.done.peek().litToBoolean mustBe true
      }
      dut.io.status.busy.expect(false.B)

      def readFullRow(address: Int): Seq[Int] = {
        dut.io.fullReadRequest.bits.poke(address.U)
        dut.io.fullReadRequest.valid.poke(true.B)
        while (!dut.io.fullReadRequest.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.fullReadRequest.valid.poke(false.B)
        var waitCycles = 0
        while (!dut.io.fullReadResponse.valid.peek().litToBoolean &&
          waitCycles < 4) {
          dut.clock.step()
          waitCycles += 1
        }
        dut.io.fullReadResponse.valid.peek().litToBoolean mustBe true
        (0 until p.dim).map { lane =>
          signedByte(dut.io.fullReadResponse.bits(lane).peek().litValue)
        }
      }

      for (pixel <- 0 until height * width) {
        readFullRow(outputBase + pixel) mustBe expected(pixel)
      }
    }
  }
}
