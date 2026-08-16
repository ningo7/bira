package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class BfsrcnnShrink23Spec extends AnyFreeSpec with Matchers {
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
      val rounded = (magnitude + (1 << (amount - 1))) >> amount
      if (value < 0) -rounded else rounded
    }
  }

  "generic depthwise mode must execute shrink2 A8 W16 3x3 convolution" in {
    val p = BiRaParams(
      dim = 16,
      fullBanks = 6,
      binaryBanks = 2,
      accumulatorBanks = 2,
      bankRows = 128,
      maxImageHeight = 2,
      maxImageWidth = 3,
      maxInputChannels = 32,
      maxOutputBlocks = 2
    )
    val height = 2
    val width = 3
    val channels = 32
    val outputBlocks = channels / p.dim

    val input = Seq.tabulate(height * width, channels) {
      case (pixel, channel) => (pixel * 13 + channel * 5) % 29
    }
    val weights = Seq.tabulate(channels, p.kernelElements) {
      case (channel, tap) => ((channel * 3 + tap * 5) % 9) - 4
    }
    val bias = Seq.tabulate(channels)(channel => (channel % 7) - 3)
    val shifts = Seq.tabulate(channels)(channel =>
      if ((channel & 1) == 0) 0 else -1
    )

    val expected = Seq.tabulate(height * width, channels) {
      case (pixel, channel) =>
        val y = pixel / width
        val x = pixel % width
        var accumulator = bias(channel)
        for {
          ky <- 0 until 3
          kx <- 0 until 3
        } {
          val iy = y + ky - 1
          val ix = x + kx - 1
          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            accumulator += input(iy * width + ix)(channel) *
              weights(channel)(ky * 3 + kx)
          }
        }
        shiftInteger(accumulator.max(0), shifts(channel)).max(0).min(255)
    }

    val inputBase = 0
    val weightLowBase = p.bankRows
    val weightHighBase = 2 * p.bankRows
    val outputBase = 3 * p.bankRows

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

      input.flatten.grouped(p.dim).zipWithIndex.foreach {
        case (row, index) => writeFullRow(inputBase + index, row)
      }

      for {
        block <- 0 until outputBlocks
        tap <- 0 until p.kernelElements
      } {
        val values = Seq.tabulate(p.dim) { lane =>
          weights(block * p.dim + lane)(tap)
        }
        val row = block * p.kernelElements + tap
        writeFullRow(weightLowBase + row, values.map(_ & 0xff))
        writeFullRow(
          weightHighBase + row,
          values.map(value => (value >> 8) & 0xff)
        )
      }

      for (block <- 0 until outputBlocks) {
        dut.io.parameterWrite.bits.block.poke(block.U)
        for (lane <- 0 until p.dim) {
          val channel = block * p.dim + lane
          dut.io.parameterWrite.bits.bias(lane).poke(bias(channel).S)
          val post = dut.io.parameterWrite.bits.post(lane)
          post.positiveShift.poke(shifts(channel).S)
          post.negativeCoeff1.poke(0.S)
          post.negativeCoeff2.poke(0.S)
          post.negativeLeftShift1.poke(0.U)
          post.negativeLeftShift2.poke(0.U)
          post.negativeCommonShift.poke(0.S)
          post.qMin.poke(0.S)
          post.qMax.poke(255.S)
          dut.io.parameterWrite.bits.binaryThreshold(lane).poke(0.S)
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
      dut.io.command.bits.inputChannels.poke(channels.U)
      dut.io.command.bits.outputBlocks.poke(outputBlocks.U)
      dut.io.command.bits.kernelHeight.poke(3.U)
      dut.io.command.bits.kernelWidth.poke(3.U)
      dut.io.command.bits.paddingY.poke(1.U)
      dut.io.command.bits.paddingX.poke(1.U)
      dut.io.command.bits.weightPrecision.poke(16.U)
      dut.io.command.bits.activationPrecision.poke(8.U)
      dut.io.command.bits.inputSigned.poke(false.B)
      dut.io.command.bits.depthwise.poke(true.B)
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
      while (!dut.io.status.done.peek().litToBoolean && cycles < 12000) {
        dut.clock.step()
        cycles += 1
      }
      withClue(s"shrink2 timed out after $cycles cycles") {
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
        (0 until p.dim).map(lane =>
          dut.io.fullReadResponse.bits(lane).peek().litValue.toInt
        )
      }

      for {
        pixel <- 0 until height * width
        block <- 0 until outputBlocks
      } {
        readFullRow(outputBase + pixel * outputBlocks + block) mustBe
          expected(pixel).slice(block * p.dim, (block + 1) * p.dim)
      }
    }
  }

  "generic dense mode must execute shrink3 and write full plus thresholded binary state" in {
    val p = BiRaParams(
      dim = 16,
      fullBanks = 6,
      binaryBanks = 2,
      accumulatorBanks = 2,
      bankRows = 128,
      maxImageHeight = 2,
      maxImageWidth = 2,
      maxInputChannels = 32,
      maxOutputBlocks = 1
    )
    val height = 2
    val width = 2
    val inputChannels = 32
    val outputChannels = 16

    val input = Seq.tabulate(height * width, inputChannels) {
      case (pixel, channel) => (pixel * 7 + channel * 3) % 23
    }
    val weights = Seq.tabulate(outputChannels, inputChannels) {
      case (outputChannel, inputChannel) =>
        ((outputChannel * 5 + inputChannel * 3) % 11) - 5
    }
    val bias = Seq.tabulate(outputChannels)(channel => (channel % 7) - 3)
    val shifts = Seq.tabulate(outputChannels)(channel =>
      if ((channel & 1) == 0) 0 else -1
    )
    val thresholds = Seq.tabulate(outputChannels)(channel =>
      (channel % 9) - 4
    )

    val expected = Seq.tabulate(height * width, outputChannels) {
      case (pixel, outputChannel) =>
        val accumulator = bias(outputChannel) +
          (0 until inputChannels).map { inputChannel =>
            input(pixel)(inputChannel) *
              weights(outputChannel)(inputChannel)
          }.sum
        shiftInteger(accumulator, shifts(outputChannel)).max(-128).min(127)
    }

    val inputBase = 0
    val weightLowBase = p.bankRows
    val outputBase = 3 * p.bankRows
    val binaryOutputBase = 0

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

      input.flatten.grouped(p.dim).zipWithIndex.foreach {
        case (row, index) => writeFullRow(inputBase + index, row)
      }

      for (inputChannel <- 0 until inputChannels) {
        val values = Seq.tabulate(p.dim)(lane =>
          weights(lane)(inputChannel)
        )
        writeFullRow(
          weightLowBase + inputChannel,
          values.map(_ & 0xff)
        )
      }

      dut.io.parameterWrite.bits.block.poke(0.U)
      for (lane <- 0 until p.dim) {
        dut.io.parameterWrite.bits.bias(lane).poke(bias(lane).S)
        val post = dut.io.parameterWrite.bits.post(lane)
        post.positiveShift.poke(shifts(lane).S)
        post.negativeCoeff1.poke(1.S)
        post.negativeCoeff2.poke(0.S)
        post.negativeLeftShift1.poke(0.U)
        post.negativeLeftShift2.poke(0.U)
        post.negativeCommonShift.poke(shifts(lane).S)
        post.qMin.poke((-128).S)
        post.qMax.poke(127.S)
        dut.io.parameterWrite.bits.binaryThreshold(lane)
          .poke(thresholds(lane).S)
      }
      dut.io.parameterWrite.valid.poke(true.B)
      while (!dut.io.parameterWrite.ready.peek().litToBoolean) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.parameterWrite.valid.poke(false.B)

      dut.io.command.bits.inputBase.poke(inputBase.U)
      dut.io.command.bits.weightLowBase.poke(weightLowBase.U)
      dut.io.command.bits.weightHighBase.poke(0.U)
      dut.io.command.bits.outputBase.poke(outputBase.U)
      dut.io.command.bits.binaryOutputBase.poke(binaryOutputBase.U)
      dut.io.command.bits.accumulatorBase.poke(0.U)
      dut.io.command.bits.residualBase.poke(0.U)
      dut.io.command.bits.inputHeight.poke(height.U)
      dut.io.command.bits.inputWidth.poke(width.U)
      dut.io.command.bits.outputHeight.poke(height.U)
      dut.io.command.bits.outputWidth.poke(width.U)
      dut.io.command.bits.inputChannels.poke(inputChannels.U)
      dut.io.command.bits.outputBlocks.poke(1.U)
      dut.io.command.bits.kernelHeight.poke(1.U)
      dut.io.command.bits.kernelWidth.poke(1.U)
      dut.io.command.bits.paddingY.poke(0.U)
      dut.io.command.bits.paddingX.poke(0.U)
      dut.io.command.bits.weightPrecision.poke(8.U)
      dut.io.command.bits.activationPrecision.poke(8.U)
      dut.io.command.bits.inputSigned.poke(false.B)
      dut.io.command.bits.depthwise.poke(false.B)
      dut.io.command.bits.columnReduce.poke(false.B)
      dut.io.command.bits.bilinearResidual.poke(false.B)
      dut.io.command.bits.shufflePack2.poke(false.B)
      dut.io.command.bits.shuffleLog2.poke(0.U)
      dut.io.command.bits.writeFullOutput.poke(true.B)
      dut.io.command.bits.writeBinaryOutput.poke(true.B)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peek().litToBoolean) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      var cycles = 0
      while (!dut.io.status.done.peek().litToBoolean && cycles < 12000) {
        dut.clock.step()
        cycles += 1
      }
      withClue(s"shrink3 timed out after $cycles cycles") {
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
        (0 until p.dim).map(lane =>
          signedByte(dut.io.fullReadResponse.bits(lane).peek().litValue)
        )
      }

      def readBinaryRow(address: Int): Seq[Int] = {
        dut.io.binaryReadRequest.bits.poke(address.U)
        dut.io.binaryReadRequest.valid.poke(true.B)
        while (!dut.io.binaryReadRequest.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.binaryReadRequest.valid.poke(false.B)
        while (!dut.io.binaryReadResponse.valid.peek().litToBoolean) {
          dut.clock.step()
        }
        (0 until p.dim).map(lane =>
          dut.io.binaryReadResponse.bits(lane).peek().litValue.toInt
        )
      }

      for (pixel <- 0 until height * width) {
        readFullRow(outputBase + pixel) mustBe expected(pixel)
        readBinaryRow(binaryOutputBase + pixel) mustBe
          Seq.tabulate(outputChannels)(lane =>
            if (expected(pixel)(lane) >= thresholds(lane)) 1 else 0
          )
      }
    }
  }
}
