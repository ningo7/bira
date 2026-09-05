package bira

// Common interfaces shared across hardware layers.

import chisel3._
import chisel3.util._

object AccumulatorOperation extends ChiselEnum {
  val write, add, read = Value
}

/** One-entry elastic pipeline register.
  *
  * Unlike a work queue this holds only the item crossing a pipeline boundary.
  * Simultaneous consume/replace is supported, so a ready downstream preserves
  * an initiation interval of one cycle.
  */
class ElasticRegister[T <: Data](gen: T) extends Module {
  val io = IO(new Bundle {
    val enq = Flipped(Decoupled(gen))
    val deq = Decoupled(gen)
  })

  private val full = RegInit(false.B)
  private val data = Reg(gen)

  io.enq.ready := !full || io.deq.ready
  io.deq.valid := full
  io.deq.bits := data

  when(io.enq.fire) {
    data := io.enq.bits
  }
  when(io.enq.fire =/= io.deq.fire) {
    full := io.enq.fire
  }
}

/** Two-entry non-transparent pipeline buffer implemented with registers.
  *
  * A generic two-entry Queue with a several-hundred-bit payload is commonly
  * mapped to asynchronously-read distributed RAM on Xilinx FPGAs. That costs
  * LUTs and leaves a wide memory read mux on the consumer path. These
  * controller boundaries only need two ordered skid slots, so fixed registers
  * are smaller and can be placed next to their producer and consumer.
  *
  * `enq.ready` depends only on registered occupancy, preserving the intended
  * combinational timing cut. At occupancy one, consume-and-replace happens in
  * one cycle, so steady-state initiation interval remains one.
  */
class TwoEntryBuffer[T <: Data](gen: T) extends Module {
  val io = IO(new Bundle {
    val enq = Flipped(Decoupled(gen))
    val deq = Decoupled(gen)
  })

  private val occupancy = RegInit(0.U(2.W))
  private val first = Reg(gen)
  private val second = Reg(gen)

  io.enq.ready := occupancy =/= 2.U
  io.deq.valid := occupancy =/= 0.U
  io.deq.bits := first

  private val enqueue = io.enq.fire
  private val dequeue = io.deq.fire

  when(enqueue && dequeue) {
    // Non-transparent full handling means this case occurs at occupancy one.
    first := io.enq.bits
  }.elsewhen(enqueue) {
    when(occupancy === 0.U) {
      first := io.enq.bits
    }.otherwise {
      second := io.enq.bits
    }
    occupancy := occupancy + 1.U
  }.elsewhen(dequeue) {
    when(occupancy === 2.U) {
      first := second
    }
    occupancy := occupancy - 1.U
  }
}

/** Temporary standalone command for a stride-one multi-bit convolution.
  *
  * Address units are vector rows, not bytes. A full-scratchpad row contains
  * `dim` 8-bit elements.
  *
  * Runtime fields describe the convolution rather than naming a network layer.
  * Output channels are padded to `dim`-lane blocks in scratchpad storage.
  */
class ConvolutionCommand(p: AccelParams) extends Bundle {
  val inputBase = UInt(p.fullAddressBits.W)
  val weightLowBase = UInt(p.fullAddressBits.W)
  val weightHighBase = UInt(p.fullAddressBits.W)
  val outputBase = UInt(p.fullAddressBits.W)
  val binaryOutputBase = UInt(p.binaryAddressBits.W)
  val accumulatorBase = UInt(p.accumulatorAddressBits.W)
  /** Packed original low-resolution uint8 image for final bilinear residual. */
  val residualBase = UInt(p.fullAddressBits.W)

  val inputHeight = UInt(p.imageDimensionBits.W)
  val inputWidth = UInt(p.imageDimensionBits.W)
  val outputHeight = UInt(p.imageDimensionBits.W)
  val outputWidth = UInt(p.imageDimensionBits.W)
  val inputChannels = UInt(p.channelCountBits.W)
  val outputBlocks = UInt(p.blockIndexBits.W)
  val kernelHeight = UInt(p.kernelDimensionBits.W)
  val kernelWidth = UInt(p.kernelDimensionBits.W)
  val paddingY = UInt(p.paddingBits.W)
  val paddingX = UInt(p.paddingBits.W)
  val weightPrecision = UInt(p.weightPrecisionBits.W)
  val activationPrecision = UInt(p.activationPrecisionBits.W)
  val inputSigned = Bool()
  val depthwise = Bool()
  /** Use the first dim/2 columns as parallel input channels, then sum columns. */
  val columnReduce = Bool()
  /** Generate an upscaled residual from the original low-resolution input. */
  val bilinearResidual = Bool()
  /** Pack two adjacent PixelShuffle output pixels into one dim-lane row. */
  val shufflePack2 = Bool()
  /** log2 of the PixelShuffle or bilinear-residual scale. */
  val shuffleLog2 = UInt(p.shuffleLogBits.W)
  val writeFullOutput = Bool()
  val writeBinaryOutput = Bool()
}

/** Request one vector of bilinearly upscaled uint8 residual pixels.
  *
  * The source is a packed single-channel low-resolution image in fullSpad:
  * one vector row contains `dim` consecutive pixels. `outputStartPixel`
  * identifies the first high-resolution pixel returned in the vector.
  */
class InterpReq(p: AccelParams) extends Bundle {
  val inputBase = UInt(p.fullAddressBits.W)
  val inputHeight = UInt(p.imageDimensionBits.W)
  val inputWidth = UInt(p.imageDimensionBits.W)
  val outputWidth = UInt(p.imageDimensionBits.W)
  val outputStartPixel = UInt(p.pixelIndexBits.W)
  val outputStartX = UInt(p.imageDimensionBits.W)
  val outputStartY = UInt(p.imageDimensionBits.W)
  val outputPixelCount = UInt((p.pixelIndexBits + 1).W)
  val scaleLog2 = UInt(p.shuffleLogBits.W)
}

/** Runtime command for a packed binary convolution.
  *
  * Binary activations use HWI vector rows: one row contains `dim` input
  * channels. Binary weights use output-channel-major OIHW packing: one
  * scratchpad row contains the `dim` input-channel bits for one output
  * channel, kernel tap, and input block.
  *
  * `correctionBase` points at compiler-preloaded Parameter Buffer rows. Each
  * row packs multiple per-pixel `-N` values used as the binary convolution
  * bias. The accumulator is initialized from that bias and then accumulates
  * `2 * popcount`, so post-processing receives `2 * popcount - N` directly.
  */
class BinaryConvolutionCommand(p: AccelParams) extends Bundle {
  val inputBase = UInt(p.binaryAddressBits.W)
  val weightBase = UInt(p.binaryAddressBits.W)
  val binaryOutputBase = UInt(p.binaryAddressBits.W)
  val residualBase = UInt(p.fullAddressBits.W)
  val fullOutputBase = UInt(p.fullAddressBits.W)
  val correctionBase = UInt(p.parameterAddressBits.W)
  val accumulatorBase = UInt(p.accumulatorAddressBits.W)

  val inputHeight = UInt(p.imageDimensionBits.W)
  val inputWidth = UInt(p.imageDimensionBits.W)
  val outputHeight = UInt(p.imageDimensionBits.W)
  val outputWidth = UInt(p.imageDimensionBits.W)
  val inputChannels = UInt(p.channelCountBits.W)
  val outputBlocks = UInt(p.blockIndexBits.W)
  val kernelHeight = UInt(p.kernelDimensionBits.W)
  val kernelWidth = UInt(p.kernelDimensionBits.W)
  val paddingY = UInt(p.paddingBits.W)
  val paddingX = UInt(p.paddingBits.W)
  val writeBinaryOutput = Bool()
}

class VectorWriteRequest(
  addressBits: Int,
  lanes: Int,
  elementBits: Int
) extends Bundle {
  val address = UInt(addressBits.W)
  val data = Vec(lanes, UInt(elementBits.W))
}

class ActivationFetchRequest(p: AccelParams) extends Bundle {
  val address = UInt(p.fullAddressBits.W)
  val lane = UInt(p.laneIndexBits.W)
}

class ActivationFetchResponse(p: AccelParams) extends Bundle {
  val activation = UInt(p.activationBits.W)
  val row = Vec(p.dim, UInt(p.activationBits.W))
}

class WeightFetchRequest(p: AccelParams) extends Bundle {
  val lowAddress = UInt(p.fullAddressBits.W)
  val highAddress = UInt(p.fullAddressBits.W)
  val highBytePresent = Bool()
}

class WeightFetchResponse(p: AccelParams) extends Bundle {
  val weights = Vec(p.dim, UInt(p.weightBits.W))
}

class AccumulatorRequest(p: AccelParams) extends Bundle {
  val operation = AccumulatorOperation()
  val address = UInt(p.accumulatorAddressBits.W)
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
  val shift = UInt(p.activationBitIndexBits.W)
  val negate = Bool()
  // Opaque controller metadata returned with the ordered response. Keeping
  // the tag beside the SRAM request removes the controller's one-outstanding
  // transaction state machine without adding a separate tag FIFO.
  val completesOutput = Bool()
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
  val outputX = UInt(p.imageDimensionBits.W)
  val outputY = UInt(p.imageDimensionBits.W)
}

class AccumulatorResponse(p: AccelParams) extends Bundle {
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
  val completesOutput = Bool()
  val block = UInt(p.blockIndexBits.W)
  val pixel = UInt(p.pixelIndexBits.W)
  val outputX = UInt(p.imageDimensionBits.W)
  val outputY = UInt(p.imageDimensionBits.W)
}

/** Per-channel parameters for the exact integer PReLU used by the reference C.
  *
  * Positive path:
  *   shiftInteger(x, positiveShift)
  * Negative path:
  *   shiftInteger(coeff1 * (x << left1) +
  *                coeff2 * (x << left2), commonShift)
  */
class PostProcessParameters(p: AccelParams) extends Bundle {
  val positiveShift = SInt(p.shiftBits.W)
  val negativeCoeff1 = SInt(2.W)
  val negativeCoeff2 = SInt(2.W)
  val negativeLeftShift1 = UInt(p.postShiftAmountBits.W)
  val negativeLeftShift2 = UInt(p.postShiftAmountBits.W)
  val negativeCommonShift = SInt(p.shiftBits.W)
  val qMin = SInt(p.accumulatorBits.W)
  val qMax = SInt(p.accumulatorBits.W)
}

/** Parameters for the two output lanes stored in one 512-bit ABI row. */
class ConvParamPairWrite(p: AccelParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val lanePair = UInt(p.parameterLanePairBits.W)
  val bias = Vec(2, SInt(p.accumulatorBits.W))
  val post = Vec(2, new PostProcessParameters(p))
  val binaryThreshold = Vec(2, SInt(p.accumulatorBits.W))
}

/** One vector row used to preload compiler-derived accumulator constants. */
class AccProgWrite(p: AccelParams) extends Bundle {
  val address = UInt(p.accumulatorAddressBits.W)
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
}

/** Fused binary-convolution scale/RPReLU parameters for one output lane.
  *
  * The accumulator-domain branch threshold and both affine branches match
  * IntegerBinaryBlock in
  * sw/models/bfsrcnn/quantization/integer_model.py. Each branch
  * computes:
  *
  *   shiftInteger(coeff1 * (x << left1) +
  *                coeff2 * (x << left2), commonShift) + bias
  *
  * Positive coeff1 is always +1 in the exported format.
  */
class BinPostParams(p: AccelParams) extends Bundle {
  val threshold = SInt(p.accumulatorBits.W)

  val positiveCoeff2 = SInt(2.W)
  val positiveLeftShift1 = UInt(p.postShiftAmountBits.W)
  val positiveLeftShift2 = UInt(p.postShiftAmountBits.W)
  val positiveCommonShift = SInt(p.shiftBits.W)
  val positiveBias = SInt(p.accumulatorBits.W)

  val negativeCoeff1 = SInt(2.W)
  val negativeCoeff2 = SInt(2.W)
  val negativeLeftShift1 = UInt(p.postShiftAmountBits.W)
  val negativeLeftShift2 = UInt(p.postShiftAmountBits.W)
  val negativeCommonShift = SInt(p.shiftBits.W)
  val negativeBias = SInt(p.accumulatorBits.W)

  val qMin = SInt(p.accumulatorBits.W)
  val qMax = SInt(p.accumulatorBits.W)
}

/** Binary parameters for the two lanes stored in one 512-bit ABI row. */
class BinParamPairWrite(p: AccelParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val lanePair = UInt(p.parameterLanePairBits.W)
  val post = Vec(2, new BinPostParams(p))
  /** Threshold used to prepare the next binary layer's input. */
  val outputSignThreshold =
    Vec(2, SInt(p.accumulatorBits.W))
}

class AcceleratorStatus extends Bundle {
  val busy = Bool()
  val done = Bool()
}
