package bira

// Common compile-time configuration.

import chisel3.util.log2Ceil

/** Compile-time parameters for the standalone BIRA accelerator core.
  *
  * The defaults are documented in docs/hardware.md. Runtime convolution
  * dimensions, precisions, signedness, and base addresses are carried by
  * ConvolutionCommand; structural maxima remain static.
  */
case class AccelParams(
  dim: Int = 16,
  activationBits: Int = 8,
  weightBits: Int = 16,
  accumulatorBits: Int = 32,
  fullBanks: Int = 10,
  binaryBanks: Int = 4,
  accumulatorBanks: Int = 1,
  bankRows: Int = 1024,
  parameterRows: Int = 512,
  // Includes the x4 high-resolution tail/final stages. Low-resolution
  // feature maps in the current model remain at most 32x32.
  maxImageHeight: Int = 128,
  maxImageWidth: Int = 128,
  maxInputChannels: Int = 64,
  maxOutputBlocks: Int = 8,
  kernelSize: Int = 3,
  shiftBits: Int = 8,
  maxPostProcessLeftShift: Int = 31,
  maxShuffleScale: Int = 4,
  nContexts: Int = 4,
  contextIdBits: Int = 3,
  rsEntries: Int = 8,
  loadQueueEntries: Int = 4,
  execQueueEntries: Int = 4,
  storeQueueEntries: Int = 4,
  dmaBeatBytes: Int = 16,
  pageBytes: Int = 4096
) {
  require(dim > 0 && isPowerOfTwo(dim),
    "the current packed-row address mapping requires a power-of-two dim")
  require(activationBits > 0)
  require(weightBits >= 2)
  require(weightBits == 16,
    "the configurable tile currently contains 16 weight bit-cells per column")
  require(accumulatorBits >= weightBits + activationBits)
  require(fullBanks >= 3, "head fetch requires independent activation/low/high regions")
  require(binaryBanks > 0)
  require(accumulatorBanks > 0)
  require(bankRows > 0 && isPowerOfTwo(bankRows))
  require(parameterRows > 0 && isPowerOfTwo(parameterRows))
  require(kernelSize > 0 && kernelSize % 2 == 1)
  require(maxOutputBlocks > 0)
  require(maxInputChannels > 0)
  require(maxPostProcessLeftShift > 0)
  require(maxShuffleScale >= 2 && isPowerOfTwo(maxShuffleScale))
  require(nContexts > 0 && nContexts <= (1 << contextIdBits))
  require(contextIdBits > 0)
  require(rsEntries > 0)
  require(loadQueueEntries > 0)
  require(execQueueEntries > 0)
  require(storeQueueEntries > 0)
  require(dmaBeatBytes > 0 && isPowerOfTwo(dmaBeatBytes))
  require(dmaBeatBytes <= 64)
  require(pageBytes > 0 && isPowerOfTwo(pageBytes))
  require(pageBytes % dmaBeatBytes == 0)

  val kernelElements: Int = kernelSize * kernelSize
  val maxPixels: Int = maxImageHeight * maxImageWidth
  val maxWeightOperands: Int = weightBits / 2
  val fullRows: Int = fullBanks * bankRows
  val binaryRows: Int = binaryBanks * bankRows
  val accumulatorRows: Int = accumulatorBanks * bankRows
  // One physical column produces at most one signed W16 value. Lower
  // precisions reduce several narrower values and binary mode counts 16 bits,
  // so 16 signed bits cover every normal-column result exactly.
  val arraySumBits: Int = weightBits
  // Column-reduce mode sums dim/2 independent signed W16 columns.
  val columnReduceBits: Int =
    arraySumBits + log2Ceil((dim / 2) max 1)

  val bankRowBits: Int = log2Ceil(bankRows)
  val fullAddressBits: Int = log2Ceil(fullRows)
  val binaryAddressBits: Int = log2Ceil(binaryRows)
  val accumulatorAddressBits: Int = log2Ceil(accumulatorRows)
  val parameterAddressBits: Int = log2Ceil(parameterRows)
  val parameterRowsPerBlock: Int = (dim + 1) / 2
  val parameterLanePairBits: Int =
    log2Ceil(parameterRowsPerBlock max 2)
  val correctionEntriesPerRow: Int = 512 / accumulatorBits
  val correctionEntryIndexBits: Int =
    log2Ceil(correctionEntriesPerRow max 2)
  val imageDimensionBits: Int = log2Ceil((maxImageHeight max maxImageWidth) + 1)
  val pixelIndexBits: Int = log2Ceil(maxPixels max 2)
  val blockIndexBits: Int = log2Ceil(maxOutputBlocks + 1)
  val channelCountBits: Int = log2Ceil(maxInputChannels + 1)
  val kernelIndexBits: Int = log2Ceil(kernelElements max 2)
  val kernelDimensionBits: Int = log2Ceil(kernelSize + 1)
  val paddingBits: Int = log2Ceil(kernelSize + 1)
  val laneIndexBits: Int = log2Ceil(dim max 2)
  val activationBitIndexBits: Int = log2Ceil(activationBits max 2)
  val activationPrecisionBits: Int = log2Ceil(activationBits + 1)
  val weightPrecisionBits: Int = log2Ceil(weightBits + 1)
  val weightOperandIndexBits: Int = log2Ceil(maxWeightOperands max 2)
  val weightOperandCountBits: Int = log2Ceil(maxWeightOperands + 1)
  val weightFetchCountBits: Int =
    log2Ceil(maxOutputBlocks * kernelElements * maxInputChannels + 1)
  val postShiftAmountBits: Int = log2Ceil(maxPostProcessLeftShift + 1)
  val shuffleLogBits: Int =
    log2Ceil(log2Ceil(maxShuffleScale) + 1)
  val shuffleScaleBits: Int = log2Ceil(maxShuffleScale + 1)
  val contextIndexBits: Int = log2Ceil(nContexts max 2)
  val reservationIndexBits: Int =
    log2Ceil(rsEntries max 2)
  val dmaBeatOffsetBits: Int = log2Ceil(dmaBeatBytes)
  val pageOffsetBits: Int = log2Ceil(pageBytes)

  private def isPowerOfTwo(value: Int): Boolean =
    value > 0 && (value & (value - 1)) == 0
}
