package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class BinaryLayersSpec extends AnyFreeSpec with Matchers {
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

  private def fusedDelta(value: Int): Int = {
    if (value >= 0) {
      shiftInteger(value + (value << 1), -1) + 1
    } else {
      -value - 2
    }
  }

  "two binary layers must execute XNOR, static -N, fused post, residual, and sign" in {
    val p = AccelParams(
      dim = 16,
      fullBanks = 4,
      binaryBanks = 4,
      accumulatorBanks = 1,
      bankRows = 256,
      maxImageHeight = 2,
      maxImageWidth = 9,
      maxInputChannels = 16,
      maxOutputBlocks = 1
    )
    val height = 2
    // Eighteen pixels cross the 16-entry Parameter Buffer packing boundary.
    val width = 9
    val pixels = height * width
    val kernel = 3

    val initialState = Seq.tabulate(pixels, p.dim) {
      case (pixel, channel) =>
        ((pixel * 19 + channel * 7) % 61) - 30
    }
    val initialThresholds = Seq.tabulate(p.dim) {
      channel => (channel % 7) - 3
    }
    val layerSignThresholds = Seq.tabulate(2, p.dim) {
      case (layer, channel) =>
        ((layer + 1) * 3 + channel * 5) % 17 - 8
    }
    val weights = Seq.tabulate(2, p.dim, kernel * kernel, p.dim) {
      case (layer, outputChannel, tap, inputChannel) =>
        ((layer * 11 + outputChannel * 5 + tap * 3 +
          inputChannel * 7) & 3) != 0
    }

    def signState(
      state: Seq[Seq[Int]],
      thresholds: Seq[Int]
    ): Seq[Seq[Boolean]] =
      state.map(row =>
        row.zip(thresholds).map {
          case (value, threshold) => value >= threshold
        }
      )

    def runGoldenLayer(
      layer: Int,
      residual: Seq[Seq[Int]],
      binary: Seq[Seq[Boolean]]
    ): Seq[Seq[Int]] = {
      Seq.tabulate(pixels, p.dim) {
        case (pixel, outputChannel) =>
          val y = pixel / width
          val x = pixel % width
          var accumulator = 0
          for {
            ky <- 0 until kernel
            kx <- 0 until kernel
          } {
            val iy = y + ky - 1
            val ix = x + kx - 1
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
              val tap = ky * kernel + kx
              for (inputChannel <- 0 until p.dim) {
                val activation =
                  if (binary(iy * width + ix)(inputChannel)) 1 else -1
                val weight =
                  if (weights(layer)(outputChannel)(tap)(inputChannel)) 1
                  else -1
                accumulator += activation * weight
              }
            }
          }
          (residual(pixel)(outputChannel) + fusedDelta(accumulator))
            .max(-128)
            .min(127)
      }
    }

    val binary0 = signState(initialState, initialThresholds)
    val state1 = runGoldenLayer(0, initialState, binary0)
    val binary1 = signState(state1, layerSignThresholds(0))
    val state2 = runGoldenLayer(1, state1, binary1)

    val initialFullStateBase = 0
    val layer1FullStateBase = p.bankRows
    val layer2FullStateBase = 2 * p.bankRows
    val binaryInputBase = 0
    val binaryLayer1Base = 3 * p.bankRows
    val layer0WeightBase = p.bankRows
    val layer1WeightBase = 2 * p.bankRows
    val correctionBase = 0
    val accumulatorBase = 0

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
      dut.io.correctionReadRequest.ready.poke(true.B)
      dut.io.correctionReadResponse.valid.poke(false.B)
      dut.io.correctionReadResponse.bits.poke(0.U)
      dut.clock.step()
      dut.reset.poke(false.B)

      def writeFullRow(address: Int, values: Seq[Int]): Unit = {
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

      def writeBinaryRow(
        address: Int,
        values: Seq[Boolean]
      ): Unit = {
        dut.io.binaryWrite.bits.address.poke(address.U)
        for (lane <- 0 until p.dim) {
          dut.io.binaryWrite.bits.data(lane)
            .poke(values(lane).B)
        }
        dut.io.binaryWrite.valid.poke(true.B)
        while (!dut.io.binaryWrite.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.binaryWrite.valid.poke(false.B)
      }

      initialState.zipWithIndex.foreach {
        case (row, pixel) =>
          writeFullRow(initialFullStateBase + pixel, row)
          writeBinaryRow(binaryInputBase + pixel, binary0(pixel))
      }

      for {
        layer <- 0 until 2
        outputChannel <- 0 until p.dim
        tap <- 0 until kernel * kernel
      } {
        val base =
          if (layer == 0) layer0WeightBase else layer1WeightBase
        val address =
          base + outputChannel * kernel * kernel + tap
        writeBinaryRow(
          address,
          weights(layer)(outputChannel)(tap)
        )
      }

      val corrections = Seq.tabulate(pixels) { pixel =>
        val y = pixel / width
        val x = pixel % width
        var validTaps = 0
        for {
          ky <- 0 until kernel
          kx <- 0 until kernel
        } {
          val iy = y + ky - 1
          val ix = x + kx - 1
          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            validTaps += 1
          }
        }
        -validTaps * p.dim
      }

      def writeParameters(signThresholds: Seq[Int]): Unit = {
        dut.io.binaryParameterWrite.bits.block.poke(0.U)
        for (lanePair <- 0 until p.parameterRowsPerBlock) {
          dut.io.binaryParameterWrite.bits.lanePair.poke(lanePair.U)
          for (laneInRow <- 0 until 2) {
            val lane = lanePair * 2 + laneInRow
            val parameters =
              dut.io.binaryParameterWrite.bits.post(laneInRow)
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
            dut.io.binaryParameterWrite.bits
              .outputSignThreshold(laneInRow)
              .poke(signThresholds(lane).S)
          }
          dut.io.binaryParameterWrite.valid.poke(true.B)
          while (!dut.io.binaryParameterWrite.ready.peek().litToBoolean) {
            dut.clock.step()
          }
          dut.clock.step()
          dut.io.binaryParameterWrite.valid.poke(false.B)
        }
      }

      def runLayer(
        inputBase: Int,
        weightBase: Int,
        residualBase: Int,
        fullOutputBase: Int,
        outputBase: Int,
        writeBinary: Boolean
      ): Unit = {
        dut.io.binaryCommand.bits.inputBase.poke(inputBase.U)
        dut.io.binaryCommand.bits.weightBase.poke(weightBase.U)
        dut.io.binaryCommand.bits.binaryOutputBase.poke(outputBase.U)
        dut.io.binaryCommand.bits.residualBase.poke(residualBase.U)
        dut.io.binaryCommand.bits.fullOutputBase.poke(fullOutputBase.U)
        dut.io.binaryCommand.bits.correctionBase.poke(correctionBase.U)
        dut.io.binaryCommand.bits.accumulatorBase.poke(accumulatorBase.U)
        dut.io.binaryCommand.bits.inputHeight.poke(height.U)
        dut.io.binaryCommand.bits.inputWidth.poke(width.U)
        dut.io.binaryCommand.bits.outputHeight.poke(height.U)
        dut.io.binaryCommand.bits.outputWidth.poke(width.U)
        dut.io.binaryCommand.bits.inputChannels.poke(p.dim.U)
        dut.io.binaryCommand.bits.outputBlocks.poke(1.U)
        dut.io.binaryCommand.bits.kernelHeight.poke(kernel.U)
        dut.io.binaryCommand.bits.kernelWidth.poke(kernel.U)
        dut.io.binaryCommand.bits.paddingY.poke(1.U)
        dut.io.binaryCommand.bits.paddingX.poke(1.U)
        dut.io.binaryCommand.bits.writeBinaryOutput
          .poke(writeBinary.B)
        dut.io.binaryCommand.valid.poke(true.B)
        while (!dut.io.binaryCommand.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.binaryCommand.valid.poke(false.B)

        var cycles = 0
        var correctionPending: Option[BigInt] = None
        while (!dut.io.status.done.peek().litToBoolean &&
          cycles < 30000) {
          correctionPending match {
            case Some(value) =>
              dut.io.correctionReadResponse.bits.poke(value.U)
              dut.io.correctionReadResponse.valid.poke(true.B)
            case None =>
              dut.io.correctionReadResponse.valid.poke(false.B)
          }
          val responseFire =
            correctionPending.nonEmpty &&
              dut.io.correctionReadResponse.ready.peek().litToBoolean
          val requestFire =
            dut.io.correctionReadRequest.valid.peek().litToBoolean
          val requestedCorrection =
            if (requestFire) {
              val row =
                dut.io.correctionReadRequest.bits.address.peek().litValue.toInt -
                  correctionBase
              Some(
                (0 until p.correctionEntriesPerRow).foldLeft(BigInt(0)) {
                  case (packed, entry) =>
                    val pixel = row * p.correctionEntriesPerRow + entry
                    val value =
                      if (pixel < corrections.length) corrections(pixel)
                      else 0
                    val encoded =
                      BigInt(value) & ((BigInt(1) << p.accumulatorBits) - 1)
                    packed | (encoded << (entry * p.accumulatorBits))
                }
              )
            } else {
              None
            }
          dut.clock.step()
          if (requestFire) {
            correctionPending = requestedCorrection
          } else if (responseFire) {
            correctionPending = None
          }
          cycles += 1
        }
        dut.io.correctionReadResponse.valid.poke(false.B)
        withClue(s"binary layer timed out after $cycles cycles") {
          dut.io.status.done.peek().litToBoolean mustBe true
        }
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
          signedByte(
            dut.io.fullReadResponse.bits(lane).peek().litValue
          )
        )
      }

      def readBinaryRow(address: Int): Seq[Boolean] = {
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
          dut.io.binaryReadResponse.bits(lane)
            .peek()
            .litValue != 0
        )
      }

      writeParameters(layerSignThresholds(0))
      runLayer(
        binaryInputBase,
        layer0WeightBase,
        initialFullStateBase,
        layer1FullStateBase,
        binaryLayer1Base,
        writeBinary = true
      )
      for (pixel <- 0 until pixels) {
        readFullRow(layer1FullStateBase + pixel) mustBe state1(pixel)
        readBinaryRow(binaryLayer1Base + pixel) mustBe binary1(pixel)
      }

      writeParameters(layerSignThresholds(1))
      runLayer(
        binaryLayer1Base,
        layer1WeightBase,
        layer1FullStateBase,
        layer2FullStateBase,
        outputBase = 64,
        writeBinary = false
      )
      for (pixel <- 0 until pixels) {
        readFullRow(layer2FullStateBase + pixel) mustBe state2(pixel)
      }
    }
  }
}
