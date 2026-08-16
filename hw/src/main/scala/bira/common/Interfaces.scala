package bira

// Common interfaces shared across hardware layers.

import chisel3._
import chisel3.util._

object AccumulatorOperation extends ChiselEnum {
  val write, add, read = Value
}

/** Temporary standalone command for a stride-one multi-bit convolution.
  *
  * Address units are vector rows, not bytes. A full-scratchpad row contains
  * `dim` 8-bit elements.
  *
  * Runtime fields describe the convolution rather than naming a network layer.
  * Output channels are padded to `dim`-lane blocks in scratchpad storage.
  */
class ConvolutionCommand(p: BiRaParams) extends Bundle {
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
class BilinearInterpolationRequest(p: BiRaParams) extends Bundle {
  val inputBase = UInt(p.fullAddressBits.W)
  val inputHeight = UInt(p.imageDimensionBits.W)
  val inputWidth = UInt(p.imageDimensionBits.W)
  val outputWidth = UInt(p.imageDimensionBits.W)
  val outputStartPixel = UInt(p.pixelIndexBits.W)
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
  * row packs multiple per-pixel `-N` values. The accumulator starts from zero
  * and holds `2 * popcount`; post-processing adds the selected `-N`.
  */
class BinaryConvolutionCommand(p: BiRaParams) extends Bundle {
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

class ActivationFetchRequest(p: BiRaParams) extends Bundle {
  val address = UInt(p.fullAddressBits.W)
  val lane = UInt(p.laneIndexBits.W)
}

class ActivationFetchResponse(p: BiRaParams) extends Bundle {
  val activation = UInt(p.activationBits.W)
  val row = Vec(p.dim, UInt(p.activationBits.W))
}

class WeightFetchRequest(p: BiRaParams) extends Bundle {
  val lowAddress = UInt(p.fullAddressBits.W)
  val highAddress = UInt(p.fullAddressBits.W)
  val highBytePresent = Bool()
}

class WeightFetchResponse(p: BiRaParams) extends Bundle {
  val weights = Vec(p.dim, UInt(p.weightBits.W))
}

class AccumulatorRequest(p: BiRaParams) extends Bundle {
  val operation = AccumulatorOperation()
  val address = UInt(p.accumulatorAddressBits.W)
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
  val shift = UInt(p.activationBitIndexBits.W)
  val negate = Bool()
}

class AccumulatorResponse(p: BiRaParams) extends Bundle {
  val data = Vec(p.dim, SInt(p.accumulatorBits.W))
}

/** Per-channel parameters for the exact integer PReLU used by the reference C.
  *
  * Positive path:
  *   shiftInteger(x, positiveShift)
  * Negative path:
  *   shiftInteger(coeff1 * (x << left1) +
  *                coeff2 * (x << left2), commonShift)
  */
class PostProcessParameters(p: BiRaParams) extends Bundle {
  val positiveShift = SInt(p.shiftBits.W)
  val negativeCoeff1 = SInt(2.W)
  val negativeCoeff2 = SInt(2.W)
  val negativeLeftShift1 = UInt(p.postShiftAmountBits.W)
  val negativeLeftShift2 = UInt(p.postShiftAmountBits.W)
  val negativeCommonShift = SInt(p.shiftBits.W)
  val qMin = SInt(p.accumulatorBits.W)
  val qMax = SInt(p.accumulatorBits.W)
}

class ConvolutionParameterWrite(p: BiRaParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val bias = Vec(p.dim, SInt(p.accumulatorBits.W))
  val post = Vec(p.dim, new PostProcessParameters(p))
  val binaryThreshold = Vec(p.dim, SInt(p.accumulatorBits.W))
}

/** One vector row used to preload compiler-derived accumulator constants. */
class AccumulatorProgrammingWrite(p: BiRaParams) extends Bundle {
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
class BinaryPostProcessParameters(p: BiRaParams) extends Bundle {
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

class BinaryConvolutionParameterWrite(p: BiRaParams) extends Bundle {
  val block = UInt(p.blockIndexBits.W)
  val post = Vec(p.dim, new BinaryPostProcessParameters(p))
  /** Threshold used to prepare the next binary layer's input. */
  val outputSignThreshold =
    Vec(p.dim, SInt(p.accumulatorBits.W))
}

class AcceleratorStatus extends Bundle {
  val busy = Bool()
  val done = Bool()
}
