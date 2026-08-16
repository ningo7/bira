package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class BfsrcnnExpandShuffleSpec extends AnyFreeSpec with Matchers {
  private def signedByte(value: BigInt): Int = {
    val raw = value.toInt & 0xff
    if (raw >= 128) raw - 256 else raw
  }

  private def shiftInteger(value: Int, shift: Int): Int = {
    if (shift >= 0) {
      value << shift
    } else {
      val amount = -shift
      val magnitude = math.abs(value)
      val rounded =
        (magnitude + (1 << (amount - 1))) >> amount
      if (value < 0) -rounded else rounded
    }
  }

  "expand must compute two adjacent shuffled pixels per array use" in {
    val p = BiRaParams(
      dim = 16,
      fullBanks = 4,
      binaryBanks = 1,
      accumulatorBanks = 1,
      bankRows = 128,
      maxImageHeight = 1,
      maxImageWidth = 2,
      maxInputChannels = 16,
      maxOutputBlocks = 8
    )
    val height = 1
    val width = 2
    val scale = 4
    val inputChannels = 16
    val shuffledChannels = 8
    val expandedChannels =
      shuffledChannels * scale * scale
    val pairBlocks = scale * scale / 2

    val input = Seq.tabulate(height * width, inputChannels) {
      case (pixel, channel) =>
        ((pixel * 9 + channel * 5) % 17) - 8
    }
    val weights = Seq.tabulate(expandedChannels, inputChannels) {
      case (outputChannel, inputChannel) =>
        ((outputChannel * 3 + inputChannel * 5) % 7) - 3
    }
    val bias = Seq.tabulate(expandedChannels) { outputChannel =>
      (outputChannel % 5) - 2
    }
    val negativeShift =
      Seq.tabulate(expandedChannels) { outputChannel =>
        if ((outputChannel % 3) == 0) -1 else 0
      }

    val expanded = Seq.tabulate(
      height * width,
      expandedChannels
    ) {
      case (pixel, outputChannel) =>
        val accumulator = bias(outputChannel) +
          (0 until inputChannels).map { inputChannel =>
            input(pixel)(inputChannel) *
              weights(outputChannel)(inputChannel)
          }.sum
        val shifted = if (accumulator >= 0) {
          accumulator
        } else {
          shiftInteger(
            accumulator,
            negativeShift(outputChannel)
          )
        }
        shifted.max(-128).min(127)
    }

    val shuffledHeight = height * scale
    val shuffledWidth = width * scale
    val shuffled = Seq.tabulate(
      shuffledHeight * shuffledWidth,
      shuffledChannels
    ) {
      case (highPixel, channel) =>
        val highY = highPixel / shuffledWidth
        val highX = highPixel % shuffledWidth
        val lowY = highY / scale
        val lowX = highX / scale
        val subPixel =
          (highY % scale) * scale + highX % scale
        val expandedChannel =
          channel * scale * scale + subPixel
        expanded(lowY * width + lowX)(expandedChannel)
    }

    val inputBase = 0
    val outputBase = p.bankRows
    val weightLowBase = 2 * p.bankRows

    simulate(new BiRaCore(p)) { dut =>
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
          dut.io.fullWrite.bits.data(lane)
            .poke((values(lane) & 0xff).U)
        }
        dut.io.fullWrite.valid.poke(true.B)
        while (!dut.io.fullWrite.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.fullWrite.valid.poke(false.B)
      }

      input.zipWithIndex.foreach {
        case (values, pixel) =>
          writeFullRow(inputBase + pixel, values)
      }

      for {
        pairBlock <- 0 until pairBlocks
        inputChannel <- 0 until inputChannels
      } {
        val row = pairBlock * inputChannels + inputChannel
        val laneWeights = Seq.tabulate(p.dim) { lane =>
          val channel = lane % shuffledChannels
          val subPixel =
            pairBlock * 2 +
              (if (lane < shuffledChannels) 0 else 1)
          val outputChannel =
            channel * scale * scale + subPixel
          weights(outputChannel)(inputChannel)
        }
        writeFullRow(
          weightLowBase + row,
          laneWeights.map(_ & 0xff)
        )
      }

      for (pairBlock <- 0 until pairBlocks) {
        dut.io.parameterWrite.bits.block.poke(pairBlock.U)
        for (lane <- 0 until p.dim) {
          val channel = lane % shuffledChannels
          val subPixel =
            pairBlock * 2 +
              (if (lane < shuffledChannels) 0 else 1)
          val outputChannel =
            channel * scale * scale + subPixel
          dut.io.parameterWrite.bits.bias(lane)
            .poke(bias(outputChannel).S)
          val post = dut.io.parameterWrite.bits.post(lane)
          post.positiveShift.poke(0.S)
          post.negativeCoeff1.poke(1.S)
          post.negativeCoeff2.poke(0.S)
          post.negativeLeftShift1.poke(0.U)
          post.negativeLeftShift2.poke(0.U)
          post.negativeCommonShift
            .poke(negativeShift(outputChannel).S)
          post.qMin.poke((-128).S)
          post.qMax.poke(127.S)
          dut.io.parameterWrite.bits.binaryThreshold(lane)
            .poke(0.S)
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
      dut.io.command.bits.weightHighBase.poke(0.U)
      dut.io.command.bits.outputBase.poke(outputBase.U)
      dut.io.command.bits.binaryOutputBase.poke(0.U)
      dut.io.command.bits.accumulatorBase.poke(0.U)
      dut.io.command.bits.residualBase.poke(0.U)
      dut.io.command.bits.inputHeight.poke(height.U)
      dut.io.command.bits.inputWidth.poke(width.U)
      dut.io.command.bits.outputHeight.poke(height.U)
      dut.io.command.bits.outputWidth.poke(width.U)
      dut.io.command.bits.inputChannels.poke(inputChannels.U)
      dut.io.command.bits.outputBlocks.poke(pairBlocks.U)
      dut.io.command.bits.kernelHeight.poke(1.U)
      dut.io.command.bits.kernelWidth.poke(1.U)
      dut.io.command.bits.paddingY.poke(0.U)
      dut.io.command.bits.paddingX.poke(0.U)
      dut.io.command.bits.weightPrecision.poke(8.U)
      dut.io.command.bits.activationPrecision.poke(8.U)
      dut.io.command.bits.inputSigned.poke(true.B)
      dut.io.command.bits.depthwise.poke(false.B)
      dut.io.command.bits.columnReduce.poke(false.B)
      dut.io.command.bits.bilinearResidual.poke(false.B)
      dut.io.command.bits.shufflePack2.poke(true.B)
      dut.io.command.bits.shuffleLog2.poke(2.U)
      dut.io.command.bits.writeFullOutput.poke(true.B)
      dut.io.command.bits.writeBinaryOutput.poke(false.B)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peek().litToBoolean) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      var cycles = 0
      while (!dut.io.status.done.peek().litToBoolean &&
        cycles < 50000) {
        dut.clock.step()
        cycles += 1
      }
      withClue(s"expand/shuffle timed out after $cycles cycles") {
        dut.io.status.done.peek().litToBoolean mustBe true
      }

      def readFullRow(address: Int): Seq[Int] = {
        dut.io.fullReadRequest.bits.poke(address.U)
        dut.io.fullReadRequest.valid.poke(true.B)
        while (!dut.io.fullReadRequest.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.fullReadRequest.valid.poke(false.B)
        while (!dut.io.fullReadResponse.valid.peek().litToBoolean) {
          dut.clock.step()
        }
        (0 until p.dim).map { lane =>
          signedByte(
            dut.io.fullReadResponse.bits(lane).peek().litValue
          )
        }
      }

      for (row <- 0 until shuffled.size / 2) {
        val expected =
          shuffled(row * 2) ++ shuffled(row * 2 + 1)
        readFullRow(outputBase + row) mustBe expected
      }
    }
  }
}
