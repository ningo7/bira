package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** End-to-end test for the final 8-input/1-output W16 convolution mode. */
class BfsrcnnFinalLayerSpec extends AnyFreeSpec with Matchers {
  private def floorDivide(numerator: Int, denominator: Int): Int =
    Math.floorDiv(numerator, denominator)

  private def bilinearPixel(
    input: Seq[Int],
    height: Int,
    width: Int,
    outputY: Int,
    outputX: Int,
    scale: Int
  ): Int = {
    val denominator = 2 * scale
    val divisor = denominator * denominator
    val yNumerator = 2 * outputY + 1 - scale
    val xNumerator = 2 * outputX + 1 - scale
    val y0Unclamped = floorDivide(yNumerator, denominator)
    val x0Unclamped = floorDivide(xNumerator, denominator)
    val wy1 = yNumerator - y0Unclamped * denominator
    val wx1 = xNumerator - x0Unclamped * denominator
    val wy0 = denominator - wy1
    val wx0 = denominator - wx1
    val y0 = y0Unclamped.max(0).min(height - 1)
    val y1 = (y0Unclamped + 1).max(0).min(height - 1)
    val x0 = x0Unclamped.max(0).min(width - 1)
    val x1 = (x0Unclamped + 1).max(0).min(width - 1)
    val numerator =
      input(y0 * width + x0) * wy0 * wx0 +
        input(y0 * width + x1) * wy0 * wx1 +
        input(y1 * width + x0) * wy1 * wx0 +
        input(y1 * width + x1) * wy1 * wx1
    (numerator + divisor / 2) / divisor
  }

  private def shiftInteger(value: Int, shift: Int): Int = {
    if (shift >= 0) {
      value << shift
    } else {
      val amount = -shift
      val magnitude = math.abs(value.toLong)
      val rounded = (magnitude + (1L << (amount - 1))) >> amount
      (if (value < 0) -rounded else rounded).toInt
    }
  }

  "final mode must reduce eight columns and pack sixteen output pixels per row" in {
    val p = BiRaParams(
      dim = 16,
      fullBanks = 5,
      binaryBanks = 2,
      accumulatorBanks = 2,
      bankRows = 128,
      maxImageHeight = 8,
      maxImageWidth = 12,
      maxInputChannels = 16,
      maxOutputBlocks = 1
    )

    val lowHeight = 2
    val lowWidth = 3
    val scale = 4
    val height = lowHeight * scale
    val width = lowWidth * scale
    val channels = 8
    val pixels = height * width
    val input = Seq.tabulate(pixels, channels) {
      case (pixel, channel) =>
        ((pixel * 7 + channel * 5) % 31) - 15
    }
    val weights = Seq.tabulate(9, channels) {
      case (tap, channel) =>
        ((tap * 3 + channel * 2) % 9) - 4
    }
    val bias = -7
    val correctionShift = -1
    val lowResolutionInput =
      Seq.tabulate(lowHeight * lowWidth)(pixel =>
        40 + (pixel * 29) % 181
      )

    val expected = Seq.tabulate(pixels) { pixel =>
      val y = pixel / width
      val x = pixel % width
      var accumulator = bias
      for {
        ky <- 0 until 3
        kx <- 0 until 3
      } {
        val iy = y + ky - 1
        val ix = x + kx - 1
        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
          val tap = ky * 3 + kx
          for (channel <- 0 until channels) {
            accumulator +=
              input(iy * width + ix)(channel) * weights(tap)(channel)
          }
        }
      }
      val residual = bilinearPixel(
        lowResolutionInput,
        lowHeight,
        lowWidth,
        y,
        x,
        scale
      )
      (residual + shiftInteger(accumulator, correctionShift))
        .max(0)
        .min(255)
    }

    val inputBase = 0
    val weightLowBase = p.bankRows
    val weightHighBase = 2 * p.bankRows
    val residualBase = 3 * p.bankRows
    val outputBase = 4 * p.bankRows

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

      input.grouped(2).zipWithIndex.foreach {
        case (pixelPair, row) =>
          writeFullRow(
            inputBase + row,
            pixelPair.flatten.padTo(p.dim, 0)
          )
      }
      for (tap <- 0 until 9) {
        val paddedWeights = weights(tap).padTo(p.dim, 0)
        writeFullRow(
          weightLowBase + tap,
          paddedWeights.map(_ & 0xff)
        )
        writeFullRow(
          weightHighBase + tap,
          paddedWeights.map(value => (value >> 8) & 0xff)
        )
      }
      lowResolutionInput.grouped(p.dim).zipWithIndex.foreach {
        case (values, row) =>
          writeFullRow(
            residualBase + row,
            values.padTo(p.dim, 0)
          )
      }

      dut.io.parameterWrite.bits.block.poke(0.U)
      for (lane <- 0 until p.dim) {
        dut.io.parameterWrite.bits.bias(lane).poke(
          (if (lane == 0) bias else 0).S
        )
        val post = dut.io.parameterWrite.bits.post(lane)
        post.positiveShift.poke(
          (if (lane == 0) correctionShift else 0).S
        )
        post.negativeCoeff1.poke(1.S)
        post.negativeCoeff2.poke(0.S)
        post.negativeLeftShift1.poke(0.U)
        post.negativeLeftShift2.poke(0.U)
        post.negativeCommonShift.poke(
          (if (lane == 0) correctionShift else 0).S
        )
        post.qMin.poke(Int.MinValue.S)
        post.qMax.poke(Int.MaxValue.S)
        dut.io.parameterWrite.bits.binaryThreshold(lane).poke(0.S)
      }
      dut.io.parameterWrite.valid.poke(true.B)
      while (!dut.io.parameterWrite.ready.peek().litToBoolean) {
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.parameterWrite.valid.poke(false.B)

      dut.io.command.bits.inputBase.poke(inputBase.U)
      dut.io.command.bits.weightLowBase.poke(weightLowBase.U)
      dut.io.command.bits.weightHighBase.poke(weightHighBase.U)
      dut.io.command.bits.outputBase.poke(outputBase.U)
      dut.io.command.bits.binaryOutputBase.poke(0.U)
      dut.io.command.bits.accumulatorBase.poke(0.U)
      dut.io.command.bits.residualBase.poke(residualBase.U)
      dut.io.command.bits.inputHeight.poke(height.U)
      dut.io.command.bits.inputWidth.poke(width.U)
      dut.io.command.bits.outputHeight.poke(height.U)
      dut.io.command.bits.outputWidth.poke(width.U)
      dut.io.command.bits.inputChannels.poke(channels.U)
      dut.io.command.bits.outputBlocks.poke(1.U)
      dut.io.command.bits.kernelHeight.poke(3.U)
      dut.io.command.bits.kernelWidth.poke(3.U)
      dut.io.command.bits.paddingY.poke(1.U)
      dut.io.command.bits.paddingX.poke(1.U)
      dut.io.command.bits.weightPrecision.poke(16.U)
      dut.io.command.bits.activationPrecision.poke(8.U)
      dut.io.command.bits.inputSigned.poke(true.B)
      dut.io.command.bits.depthwise.poke(false.B)
      dut.io.command.bits.columnReduce.poke(true.B)
      dut.io.command.bits.bilinearResidual.poke(true.B)
      dut.io.command.bits.shufflePack2.poke(false.B)
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
      while (!dut.io.status.done.peek().litToBoolean && cycles < 300000) {
        dut.clock.step()
        cycles += 1
      }
      withClue(s"final layer timed out after $cycles cycles") {
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
          dut.io.fullReadResponse.bits(lane).peek().litValue.toInt
        }
      }

      val outputRows = (pixels + p.dim - 1) / p.dim
      val actual = (0 until outputRows).flatMap { row =>
        readFullRow(outputBase + row)
      }
      actual.take(pixels) mustBe expected
      actual.drop(pixels).foreach(_ mustBe 0)
    }
  }
}
