package bira

// Shared binary/integer compute array.

import chisel3._
import chisel3.util._

/** Canonical operands presented to the physical compute tile.
  *
  * Dispatch expands either controller's native data layout into one pair of
  * bit vectors per physical column. Bit `cell` of the two vectors always
  * drives the same configurable bit cell, regardless of the source mode.
  */
class ComputeArrayInput(p: BiRaParams) extends Bundle {
  val activations = Vec(p.dim, UInt(p.weightBits.W))
  val weights = Vec(p.dim, UInt(p.weightBits.W))
  val binaryMode = Bool()
  val weightPrecision = UInt(p.weightPrecisionBits.W)
  val columnReduceMode = Bool()
}

/** Selects a controller and normalizes its native layout for ComputeArray. */
class ComputeArrayDispatch(p: BiRaParams) extends Module {
  val io = IO(new Bundle {
    val binaryMode = Input(Bool())

    val multiInputValid = Input(Bool())
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
    val columnReduceMode = Input(Bool())

    val binaryInputValid = Input(Bool())
    val binaryActivation = Input(UInt(p.dim.W))
    val binaryWeights =
      Input(Vec(p.dim, UInt(p.dim.W)))

    val output = Valid(new ComputeArrayInput(p))
  })

  io.output.valid := Mux(
    io.binaryMode,
    io.binaryInputValid,
    io.multiInputValid
  )
  io.output.bits.binaryMode := io.binaryMode
  io.output.bits.weightPrecision := Mux(
    io.binaryMode,
    0.U,
    io.weightPrecision
  )
  io.output.bits.columnReduceMode :=
    !io.binaryMode && io.columnReduceMode

  for (column <- 0 until p.dim) {
    val activationCells = Wire(Vec(p.weightBits, Bool()))
    val weightCells = Wire(Vec(p.weightBits, Bool()))

    for (cell <- 0 until p.weightBits) {
      val multiActivation = MuxLookup(
        io.weightPrecision,
        false.B
      )(
        Seq(
          16.U -> io.multiActivationBits(0)(column),
          8.U -> io.multiActivationBits(cell / 8)(column),
          4.U -> io.multiActivationBits(cell / 4)(column),
          2.U -> io.multiActivationBits(cell / 2)(column)
        )
      )
      val multiWeightBit = MuxLookup(
        io.weightPrecision,
        false.B
      )(
        Seq(
          16.U -> io.multiWeights(0)(column)(cell),
          8.U -> io.multiWeights(cell / 8)(column)(cell % 8),
          4.U -> io.multiWeights(cell / 4)(column)(cell % 4),
          2.U -> io.multiWeights(cell / 2)(column)(cell % 2)
        )
      )
      val binaryActivation =
        if (cell < p.dim) io.binaryActivation(cell)
        else false.B
      val binaryWeight =
        if (cell < p.dim) io.binaryWeights(column)(cell)
        else false.B

      activationCells(cell) := Mux(
        io.binaryMode,
        binaryActivation,
        multiActivation
      )
      weightCells(cell) := Mux(
        io.binaryMode,
        binaryWeight,
        multiWeightBit
      )
    }

    io.output.bits.activations(column) :=
      activationCells.asUInt
    io.output.bits.weights(column) := weightCells.asUInt
  }
}

/** One physical compute tile shared by packed-binary and multi-bit modes.
  *
  * Every column contains 16 configurable bit cells. A cell performs:
  *
  *   binary mode: XNOR(activation, weight)
  *   multi mode:  AND(activation, signed-weight bit)
  *
  * The column also contains one shared AddTree. In binary mode
  * all `dim` XNOR results enter the full popcount tree. In multi-bit mode, the
  * AND results in each 2/4/8/16-bit group are first interpreted with signed
  * two's-complement bit weights; the resulting 8/4/2/1 channel products enter
  * the matching level of the same tree.
  */
class ComputeArray(p: BiRaParams) extends Module {
  private val cellCount = p.weightBits
  require(cellCount == 16)
  require(p.dim <= cellCount)

  val io = IO(new Bundle {
    val input = Flipped(Valid(new ComputeArrayInput(p)))

    /** Multi-bit partial sum or binary equality popcount per column. */
    val outputValid = Output(Bool())
    val columnSums =
      Output(Vec(p.dim, SInt(p.accumulatorBits.W)))
    val columnReducedValid = Output(Bool())
    val columnReducedSum = Output(SInt(p.accumulatorBits.W))
  })

  private val operandCount = MuxLookup(
    io.input.bits.weightPrecision,
    0.U(p.weightOperandCountBits.W)
  )(
    Seq(
      16.U -> 1.U,
      8.U -> 2.U,
      4.U -> 4.U,
      2.U -> 8.U
    )
  )

  private def signedGroup(
    bits: Seq[Bool],
    width: Int
  ): SInt = {
    require(bits.length == width)
    Cat(bits.reverse).asSInt.pad(p.accumulatorBits)
  }

  for (column <- 0 until p.dim) {
    val cellResults = Wire(Vec(cellCount, Bool()))

    for (cell <- 0 until cellCount) {
      val activation = io.input.bits.activations(column)(cell)
      val weight = io.input.bits.weights(column)(cell)
      val bothOne = activation && weight
      val bothZero = !activation && !weight

      // AND is common to both modes; binary mode additionally counts 0 == 0.
      cellResults(cell) :=
        bothOne || (io.input.bits.binaryMode && bothZero)
    }

    val tree = Module(new AddTree(p.accumulatorBits))
    for (treeInput <- 0 until 16) {
      val multiProduct =
        if (treeInput < p.maxWeightOperands) {
          val product16 =
            if (treeInput == 0) {
              signedGroup(
                (0 until 16).map(cellResults(_)),
                16
              )
            } else {
              0.S(p.accumulatorBits.W)
            }
          val product8 =
            if (treeInput < 2) {
              signedGroup(
                (0 until 8).map(index =>
                  cellResults(treeInput * 8 + index)
                ),
                8
              )
            } else {
              0.S(p.accumulatorBits.W)
            }
          val product4 =
            if (treeInput < 4) {
              signedGroup(
                (0 until 4).map(index =>
                  cellResults(treeInput * 4 + index)
                ),
                4
              )
            } else {
              0.S(p.accumulatorBits.W)
            }
          val product2 = signedGroup(
            (0 until 2).map(index =>
              cellResults(treeInput * 2 + index)
            ),
            2
          )

          MuxLookup(
            io.input.bits.weightPrecision,
            0.S(p.accumulatorBits.W)
          )(
            Seq(
              16.U -> product16,
              8.U -> product8,
              4.U -> product4,
              2.U -> product2
            )
          )
        } else {
          0.S(p.accumulatorBits.W)
        }

      tree.io.inputs(treeInput) := Mux(
        io.input.bits.binaryMode,
        cellResults(treeInput).asUInt.zext,
        multiProduct
      )
    }
    tree.io.inputValid := io.input.valid
    tree.io.inputCount := Mux(
      io.input.bits.binaryMode,
      p.dim.U,
      operandCount
    )
    io.columnSums(column) := tree.io.output
    if (column == 0) {
      io.outputValid := tree.io.outputValid
    }
  }

  // Final-layer mode changes the meaning of columns: every active column owns
  // one input channel instead of one output channel. The normal per-column
  // trees still form W16 products; this tree performs the extra Cin reduction.
  private val columnTree = Module(new AddTree(p.accumulatorBits))
  for (column <- 0 until 16) {
    columnTree.io.inputs(column) :=
      (if (column < p.dim) io.columnSums(column)
       else 0.S(p.accumulatorBits.W))
  }
  columnTree.io.inputValid :=
    io.input.bits.columnReduceMode && io.outputValid
  columnTree.io.inputCount := (p.dim / 2).U
  io.columnReducedValid := columnTree.io.outputValid
  io.columnReducedSum := columnTree.io.output

  when(io.input.valid && !io.input.bits.binaryMode) {
    assert(
      io.input.bits.weightPrecision === 0.U ||
        io.input.bits.weightPrecision === 2.U ||
        io.input.bits.weightPrecision === 4.U ||
        io.input.bits.weightPrecision === 8.U ||
        io.input.bits.weightPrecision === 16.U,
      "multi-bit precision must be inactive or 2/4/8/16"
    )
  }
}
