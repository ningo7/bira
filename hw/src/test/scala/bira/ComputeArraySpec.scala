package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Native controller-shaped test wrapper around dispatch and ComputeArray. */
private class DispatchedComputeArray(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val binaryMode = Input(Bool())
    val columnReduceMode = Input(Bool())
    val inputValid = Input(Bool())

    val multiActivationBits =
      Input(Vec(p.maxWeightOperands, Vec(p.dim, Bool())))
    val multiWeights = Input(
      Vec(
        p.maxWeightOperands,
        Vec(p.dim, UInt(p.weightBits.W))
      )
    )
    val weightPrecision =
      Input(UInt(p.weightPrecisionBits.W))

    val binaryActivation = Input(UInt(p.dim.W))
    val binaryWeights =
      Input(Vec(p.dim, UInt(p.dim.W)))

    val outputValid = Output(Bool())
    val columnSums =
      Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
  })

  private val dispatch = Module(new ComputeArrayDispatch(p))
  private val array = Module(new ComputeArray(p))

  dispatch.io.binaryMode := io.binaryMode
  dispatch.io.multiInputValid := io.inputValid
  dispatch.io.multiActivationBits := io.multiActivationBits
  dispatch.io.multiWeights := io.multiWeights
  dispatch.io.weightPrecision := io.weightPrecision
  dispatch.io.columnReduceMode := io.columnReduceMode
  dispatch.io.binaryInputValid := io.inputValid
  dispatch.io.binaryActivation := io.binaryActivation
  dispatch.io.binaryWeights := io.binaryWeights

  array.io.input := dispatch.io.output
  io.outputValid := array.io.outputValid
  io.columnSums := array.io.columnSums
}

/** Unit tests for native-layout normalization before the physical array. */
class ComputeArrayDispatchSpec extends AnyFreeSpec with Matchers {
  "dispatch must produce one canonical bit-cell input for every precision" in {
    val p = AccelParams(dim = 16)
    val operandWeights =
      Seq(0x0001, 0x0002, 0x0004, 0x0008, 0x0001, 0x0002, 0x0000, 0x0001)
    val operandActivations =
      Seq(true, false, true, true, false, true, false, true)

    simulate(new ComputeArrayDispatch(p)) { dut =>
      dut.io.binaryMode.poke(false.B)
      dut.io.multiInputValid.poke(true.B)
      dut.io.columnReduceMode.poke(false.B)
      dut.io.binaryInputValid.poke(false.B)
      dut.io.binaryActivation.poke(0.U)
      dut.io.binaryWeights.foreach(_.poke(0.U))
      for {
        operand <- 0 until p.maxWeightOperands
        column <- 0 until p.dim
      } {
        dut.io.multiActivationBits(operand)(column)
          .poke((column == 0 && operandActivations(operand)).B)
        dut.io.multiWeights(operand)(column).poke(
          (if (column == 0) operandWeights(operand) else 0).U
        )
      }

      for (precision <- Seq(16, 8, 4, 2)) {
        val operandCount = p.weightBits / precision
        val groupMask = (BigInt(1) << precision) - 1
        val expectedActivations =
          (0 until operandCount).foldLeft(BigInt(0)) {
            case (packed, operand) =>
              if (operandActivations(operand)) {
                packed | (groupMask << (operand * precision))
              } else {
                packed
              }
          }
        val expectedWeights =
          (0 until operandCount).foldLeft(BigInt(0)) {
            case (packed, operand) =>
              packed |
                ((BigInt(operandWeights(operand)) & groupMask) <<
                  (operand * precision))
          }

        dut.io.weightPrecision.poke(precision.U)
        dut.io.output.valid.expect(true.B)
        dut.io.output.bits.binaryMode.expect(false.B)
        dut.io.output.bits.weightPrecision.expect(precision.U)
        dut.io.output.bits.activations(0)
          .expect(expectedActivations.U)
        dut.io.output.bits.weights(0).expect(expectedWeights.U)
      }

      val binaryActivation = 0xa55a
      val binaryWeight = 0x3cc3
      dut.io.binaryMode.poke(true.B)
      dut.io.binaryInputValid.poke(true.B)
      dut.io.binaryActivation.poke(binaryActivation.U)
      dut.io.binaryWeights(0).poke(binaryWeight.U)
      dut.io.output.valid.expect(false.B)
      dut.clock.step()
      dut.io.output.valid.expect(true.B)
      dut.io.output.bits.binaryMode.expect(true.B)
      dut.io.output.bits.weightPrecision.expect(0.U)
      dut.io.output.bits.activations(0)
        .expect(binaryActivation.U)
      dut.io.output.bits.weights(0).expect(binaryWeight.U)
    }
  }
}

/** Unit tests for dispatch plus the shared binary/multi-bit array. */
class ComputeArraySpec extends AnyFreeSpec with Matchers {
  private def runArray(
    dut: DispatchedComputeArray,
    binaryInput: Boolean = false
  ): Unit = {
    dut.io.inputValid.poke(true.B)
    dut.clock.step()
    dut.io.inputValid.poke(false.B)
    dut.clock.step()
    if (binaryInput) {
      dut.clock.step()
    }
    dut.io.outputValid.expect(true.B)
  }

  private def resetArray(dut: DispatchedComputeArray): Unit = {
    dut.reset.poke(true.B)
    dut.io.inputValid.poke(false.B)
    dut.io.columnReduceMode.poke(false.B)
    dut.clock.step()
    dut.reset.poke(false.B)
  }

  private def clearBinaryInputs(
    dut: DispatchedComputeArray,
    p: AccelParams
  ): Unit = {
    dut.io.binaryActivation.poke(0.U)
    dut.io.binaryWeights.foreach(_.poke(0.U))
  }

  private def clearMultiInputs(
    dut: DispatchedComputeArray,
    p: AccelParams
  ): Unit = {
    for {
      operand <- 0 until p.maxWeightOperands
      column <- 0 until p.dim
    } {
      dut.io.multiActivationBits(operand)(column).poke(false.B)
      dut.io.multiWeights(operand)(column).poke(0.U)
    }
  }

  "AND mode must reconstruct a selected signed W16 weight" in {
    val p = AccelParams(dim = 16)
    val weights = Seq(
      -32768, -12345, -257, -1,
      0, 1, 2, 7,
      127, 255, 256, 1024,
      12345, 16384, 30000, 32767
    )
    val mask = (BigInt(1) << p.weightBits) - 1

    simulate(new DispatchedComputeArray(p)) { dut =>
      resetArray(dut)
      dut.io.binaryMode.poke(false.B)
      clearBinaryInputs(dut, p)
      dut.io.weightPrecision.poke(16.U)
      for {
        operand <- 0 until p.maxWeightOperands
        lane <- 0 until p.dim
      } {
        val value = if (operand == 0) weights(lane) else 0
        dut.io.multiWeights(operand)(lane)
          .poke((BigInt(value) & mask).U)
        dut.io.multiActivationBits(operand)(lane).poke(false.B)
      }
      runArray(dut)
      dut.io.columnSums.foreach(_.expect(0.S))

      for (lane <- 0 until p.dim) {
        dut.io.multiActivationBits(0)(lane).poke(true.B)
      }
      runArray(dut)
      for (lane <- 0 until p.dim) {
        dut.io.columnSums(lane).expect(weights(lane).S)
      }
    }
  }

  "AND mode W4 must enter four products at the four-input tree level" in {
    val p = AccelParams(dim = 16)
    val weights = Seq.tabulate(4, p.dim) {
      case (operand, lane) => ((operand * 5 + lane * 3) % 16) - 8
    }
    val activationBits = Seq(true, false, true, true)

    simulate(new DispatchedComputeArray(p)) { dut =>
      resetArray(dut)
      dut.io.binaryMode.poke(false.B)
      clearBinaryInputs(dut, p)
      dut.io.weightPrecision.poke(4.U)
      for (operand <- 0 until p.maxWeightOperands) {
        for (lane <- 0 until p.dim) {
          dut.io.multiActivationBits(operand)(lane).poke(
            (operand < 4 && activationBits(operand)).B
          )
          val value = if (operand < 4) weights(operand)(lane) else 0
          dut.io.multiWeights(operand)(lane).poke((value & 0xf).U)
        }
      }

      runArray(dut)
      for (lane <- 0 until p.dim) {
        val expected = (0 until 4)
          .filter(activationBits)
          .map(weights(_)(lane))
          .sum
        dut.io.columnSums(lane).expect(expected.S)
      }
    }
  }

  "XNOR mode must use all 16 cells and the full popcount tree" in {
    val p = AccelParams(dim = 16)
    val activation = Integer.parseInt("1011001011100001", 2)
    val weights = Seq(
      activation,
      activation ^ 0xffff,
      0x0000,
      0xffff
    ) ++ Seq.tabulate(12)(index => (0x1234 * (index + 1)) & 0xffff)

    simulate(new DispatchedComputeArray(p)) { dut =>
      resetArray(dut)
      dut.io.binaryMode.poke(true.B)
      dut.io.weightPrecision.poke(0.U)
      clearMultiInputs(dut, p)
      dut.io.binaryActivation.poke(activation.U)
      for (column <- 0 until p.dim) {
        val weight = weights(column)
        val equal = Integer.bitCount((~(activation ^ weight)) & 0xffff)
        dut.io.binaryWeights(column).poke(weight.U)
      }
      runArray(dut, binaryInput = true)
      for (column <- 0 until p.dim) {
        val weight = weights(column)
        val equal = Integer.bitCount((~(activation ^ weight)) & 0xffff)
        dut.io.columnSums(column).expect(equal.S)
      }
    }
  }
}
