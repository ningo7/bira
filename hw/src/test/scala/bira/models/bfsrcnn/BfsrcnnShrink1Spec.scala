package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class BfsrcnnShrink1Spec extends AnyFreeSpec with Matchers {
  "generic convolution control must execute signed-A8 W4 48-to-32 pointwise convolution" in {
    val p = BiRaParams(
      dim = 16,
      fullBanks = 5,
      binaryBanks = 2,
      accumulatorBanks = 2,
      bankRows = 128,
      maxImageHeight = 2,
      maxImageWidth = 2,
      maxInputChannels = 48,
      maxOutputBlocks = 2
    )

    val height = 2
    val width = 2
    val inputChannels = 48
    val outputChannels = 32
    val outputBlocks = outputChannels / p.dim
    val operandsPerCycle = 4
    val inputGroups = inputChannels / operandsPerCycle

    val input = Seq.tabulate(height * width, inputChannels) {
      case (pixel, channel) => ((pixel * 11 + channel * 7) % 31) - 15
    }
    val weights = Seq.tabulate(outputChannels, inputChannels) {
      case (outputChannel, inputChannel) =>
        ((outputChannel * 3 + inputChannel * 5) % 16) - 8
    }
    val bias = Seq.tabulate(outputChannels) { channel =>
      (channel % 9) - 4
    }
    val requantShift = Seq.tabulate(outputChannels) { channel =>
      if ((channel & 1) == 0) 0 else -1
    }

    def shiftInteger(value: Int, shift: Int): Int = {
      if (shift >= 0) {
        value << shift
      } else {
        val amount = -shift
        val magnitude = math.abs(value)
        val rounded = (magnitude + (1 << (amount - 1))) >> amount
        if (value < 0) -rounded else rounded
      }
    }

    val expected = Seq.tabulate(height * width, outputChannels) {
      case (pixel, outputChannel) =>
        val accumulator = bias(outputChannel) +
          (0 until inputChannels).map { inputChannel =>
            input(pixel)(inputChannel) *
              weights(outputChannel)(inputChannel)
          }.sum
        shiftInteger(accumulator.max(0), requantShift(outputChannel))
          .max(0)
          .min(255)
    }

    val inputBase = 0
    val weightBase = p.bankRows
    val unusedWeightHighBase = 2 * p.bankRows
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

      input.flatten.grouped(p.dim).zipWithIndex.foreach {
        case (values, row) =>
          writeFullRow(inputBase + row, values)
      }

      for {
        outputBlock <- 0 until outputBlocks
        inputGroup <- 0 until inputGroups
        operand <- 0 until operandsPerCycle
      } {
        val row =
          (outputBlock * inputGroups + inputGroup) *
            operandsPerCycle + operand
        val values = Seq.tabulate(p.dim) { outputLane =>
          val outputChannel = outputBlock * p.dim + outputLane
          val inputChannel = inputGroup * operandsPerCycle + operand
          weights(outputChannel)(inputChannel)
        }
        writeFullRow(weightBase + row, values)
      }

      for (outputBlock <- 0 until outputBlocks) {
        dut.io.parameterWrite.bits.block.poke(outputBlock.U)
        for (lane <- 0 until p.dim) {
          val outputChannel = outputBlock * p.dim + lane
          dut.io.parameterWrite.bits.bias(lane)
            .poke(bias(outputChannel).S)
          val post = dut.io.parameterWrite.bits.post(lane)
          post.positiveShift.poke(requantShift(outputChannel).S)
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
      dut.io.command.bits.weightLowBase.poke(weightBase.U)
      dut.io.command.bits.weightHighBase.poke(unusedWeightHighBase.U)
      dut.io.command.bits.outputBase.poke(outputBase.U)
      dut.io.command.bits.binaryOutputBase.poke(0.U)
      dut.io.command.bits.accumulatorBase.poke(0.U)
      dut.io.command.bits.residualBase.poke(0.U)
      dut.io.command.bits.inputHeight.poke(height.U)
      dut.io.command.bits.inputWidth.poke(width.U)
      dut.io.command.bits.outputHeight.poke(height.U)
      dut.io.command.bits.outputWidth.poke(width.U)
      dut.io.command.bits.inputChannels.poke(inputChannels.U)
      dut.io.command.bits.outputBlocks.poke(outputBlocks.U)
      dut.io.command.bits.kernelHeight.poke(1.U)
      dut.io.command.bits.kernelWidth.poke(1.U)
      dut.io.command.bits.paddingY.poke(0.U)
      dut.io.command.bits.paddingX.poke(0.U)
      dut.io.command.bits.weightPrecision.poke(4.U)
      dut.io.command.bits.activationPrecision.poke(8.U)
      dut.io.command.bits.inputSigned.poke(true.B)
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
      while (!dut.io.status.done.peek().litToBoolean && cycles < 12000) {
        dut.clock.step()
        cycles += 1
      }
      withClue(s"shrink1 timed out after $cycles cycles") {
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
        var waitCycles = 0
        while (!dut.io.fullReadResponse.valid.peek().litToBoolean &&
          waitCycles < 4) {
          dut.clock.step()
          waitCycles += 1
        }
        dut.io.fullReadResponse.valid.peek().litToBoolean mustBe true
        (0 until p.dim).map { lane =>
          dut.io.fullReadResponse.bits(lane).peek().litValue.toInt
        }
      }

      for {
        pixel <- 0 until height * width
        outputBlock <- 0 until outputBlocks
      } {
        val expectedRow = expected(pixel).slice(
          outputBlock * p.dim,
          (outputBlock + 1) * p.dim
        )
        readFullRow(
          outputBase + pixel * outputBlocks + outputBlock
        ) mustBe expectedRow
      }
    }
  }
}
