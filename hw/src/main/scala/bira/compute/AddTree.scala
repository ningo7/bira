package bira

// Compute-layer reduction primitive.

import chisel3._
import chisel3.util._

/** A shared, two-stage 16-input binary tree with selectable entry level.
  *
  * A smaller power-of-two operand set bypasses the unused lower levels:
  *
  *   16 operands -> 16-to-8-to-4-to-2-to-1
  *    8 operands ->       8-to-4-to-2-to-1
  *    4 operands ->            4-to-2-to-1
  *    2 operands ->                 2-to-1
  *    1 operand  -> direct
  *
  * This is the physical behavior wanted by variable-precision packing. For
  * example, four W4 products enter directly at the four-operand level.
  *
  * All entry levels have the same two-cycle latency. Stage 1 produces at most
  * four partial sums. Stage 2 reduces those partial sums to one result. A new
  * input may be accepted every cycle.
  */
class AddTree(width: Int) extends Module {
  private val inputCapacity = 16

  val io = IO(new Bundle {
    val inputValid = Input(Bool())
    val inputs = Input(Vec(inputCapacity, SInt(width.W)))
    val inputCount = Input(UInt(log2Ceil(inputCapacity + 1).W))
    val outputValid = Output(Bool())
    val output = Output(SInt(width.W))
  })

  private def add(left: SInt, right: SInt): SInt =
    (left + right)(width - 1, 0).asSInt

  private val fourOperandLevel = Wire(Vec(4, SInt(width.W)))
  for (index <- 0 until 4) {
    val fourInputSum = add(
      add(io.inputs(index * 4), io.inputs(index * 4 + 1)),
      add(io.inputs(index * 4 + 2), io.inputs(index * 4 + 3))
    )
    val twoInputSum =
      add(io.inputs(index * 2), io.inputs(index * 2 + 1))
    fourOperandLevel(index) := Mux(
      io.inputCount === 16.U,
      fourInputSum,
      Mux(io.inputCount === 8.U, twoInputSum, io.inputs(index))
    )
  }

  private val stage1Values =
    Reg(Vec(4, SInt(width.W)))
  private val stage1InputCount =
    Reg(UInt(log2Ceil(inputCapacity + 1).W))
  private val stage1Valid = RegInit(false.B)

  stage1Valid := io.inputValid
  when(io.inputValid) {
    stage1Values := fourOperandLevel
    stage1InputCount := io.inputCount
  }

  private val stage2Pair0 =
    add(stage1Values(0), stage1Values(1))
  private val stage2Pair1 =
    add(stage1Values(2), stage1Values(3))
  private val stage2Result = Mux(
    stage1InputCount === 0.U,
    0.S,
    Mux(
      stage1InputCount === 1.U,
      stage1Values(0),
      Mux(
        stage1InputCount === 2.U,
        stage2Pair0,
        add(stage2Pair0, stage2Pair1)
      )
    )
  )

  private val outputRegister = Reg(SInt(width.W))
  private val outputValidRegister = RegInit(false.B)
  outputValidRegister := stage1Valid
  when(stage1Valid) {
    outputRegister := stage2Result
  }

  io.output := outputRegister
  io.outputValid := outputValidRegister

  when(io.inputValid) {
    assert(
      io.inputCount === 0.U ||
        io.inputCount === 1.U ||
        io.inputCount === 2.U ||
        io.inputCount === 4.U ||
        io.inputCount === 8.U ||
        io.inputCount === 16.U,
      "adder-tree input count must be a power of two no greater than 16"
    )
  }
}
