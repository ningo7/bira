package bira

// Software-visible ISA definitions shared by the control and integration layers.

import chisel3._
import chisel3.util._

/** Frozen BIRA ISA RoCC funct7 values.
  *
  * The standalone frontend uses these values without depending on Rocket Chip.
  * A later LazyRoCC adapter only needs to translate RoCCCommand into
  * [[BiRaRawCommand]].
  */
object BiRaFunct {
  val cfgShape = 0x40
  val cfgAddr = 0x41
  val cfgMode = 0x42
  val cfgCommit = 0x43
  val load2d = 0x44
  val execConv = 0x45
  val store2d = 0x46
  val fence = 0x47
  val status = 0x48
  val tlbFlush = 0x49
}

object BiRaAddrRole {
  val input = 0
  val weightLow = 1
  val weightHigh = 2
  val parameter = 3
  val residual = 4
  val correction = 5
  val accumulator = 6
  val outputFull = 7
  val outputBinary = 8
  val count = 9
}

object BiRaArrayMode {
  val dense = 0
  val depthwise = 1
  val binary = 2
  val columnReduce = 3
}

object BiRaWeightPrecision {
  val w2 = 0
  val w4 = 1
  val w8 = 2
  val w16 = 3
}

object BiRaPostMode {
  val none = 0
  val intPrelu = 1
  val intRelu = 2
  val intSignedSign = 3
  val binaryFused = 4
  val finalBilinearResidual = 5
}

object BiRaError {
  val none = 0x00
  val invalidContext = 0x01
  val contextBusy = 0x02
  val contextState = 0x03
  val contextNotReady = 0x04
  val contextFailed = 0x05
  val badEnum = 0x06
  val badRole = 0x07
  val badShape = 0x08
  val missingAddress = 0x09
  val illegalCombination = 0x0a
  val zeroSize = 0x0b
  val rowTooWide = 0x0c
  val badStride = 0x0d
  val localOutOfBounds = 0x0e
  val tlb = 0x0f
  val access = 0x10
  val dependency = 0x11
  val accumulatorOverflow = 0x12
  val internalProtocol = 0x13
}

/** Rocket-independent view of one decoded custom instruction. */
class BiRaRawCommand extends Bundle {
  val funct = UInt(7.W)
  val rs1 = UInt(64.W)
  val rs2 = UInt(64.W)
  val rd = UInt(5.W)
  val xd = Bool()
  val xs1 = Bool()
  val xs2 = Bool()
  /** Captured privilege/translation state for a later RoCC adapter. */
  val translationStatus = UInt(64.W)
}

class BiRaRawResponse extends Bundle {
  val rd = UInt(5.W)
  val data = UInt(64.W)
}

/** Compact LOAD_2D/STORE_2D descriptor after decoding rs2. */
class BiRaDmaTask(p: BiRaParams) extends Bundle {
  val contextId = UInt(p.contextIdBits.W)
  val role = UInt(4.W)
  val dramVirtualAddress = UInt(64.W)
  val localRowOffset = UInt(14.W)
  val rows = UInt(14.W)
  val bytesPerRow = UInt(7.W)
  val dramStrideBytes = UInt(16.W)
  val localStrideRows = UInt(6.W)
  val commandSequence = UInt(16.W)
  val translationStatus = UInt(64.W)
}

class BiRaExecTask(p: BiRaParams) extends Bundle {
  val contextId = UInt(p.contextIdBits.W)
  val commandSequence = UInt(16.W)
}

class BiRaTaskCompletion(p: BiRaParams) extends Bundle {
  val contextId = UInt(p.contextIdBits.W)
  val commandSequence = UInt(16.W)
  val errorCode = UInt(8.W)
}

class BiRaSchedulerStatus extends Bundle {
  val loadQueueCount = UInt(8.W)
  val execQueueCount = UInt(8.W)
  val storeQueueCount = UInt(8.W)
  val loadBusy = Bool()
  val execBusy = Bool()
  val storeBusy = Bool()
}

/** Context format shared by the frontend and the future scheduler. */
class BiRaContext(p: BiRaParams) extends Bundle {
  val building = Bool()
  val ready = Bool()
  val committed = Bool()
  val shapeValid = Bool()
  val modeValid = Bool()

  val inputHeight = UInt(12.W)
  val inputWidth = UInt(12.W)
  val inputChannels = UInt(12.W)
  val outputHeight = UInt(12.W)
  val outputWidth = UInt(12.W)
  val outputChannels = UInt(12.W)
  val kernelHeight = UInt(4.W)
  val kernelWidth = UInt(4.W)
  val paddingHeight = UInt(4.W)
  val paddingWidth = UInt(4.W)

  val addressValid = UInt(BiRaAddrRole.count.W)
  val baseRows = Vec(BiRaAddrRole.count, UInt(16.W))

  val arrayMode = UInt(2.W)
  val weightPrecision = UInt(2.W)
  val inputSigned = Bool()
  val postMode = UInt(3.W)
  val shufflePack2 = Bool()
  val writeFull = Bool()
  val writeBinary = Bool()

  val inflightCount = UInt(8.W)
  val errorCode = UInt(8.W)
  val errorCommandSequence = UInt(16.W)
}

class BiRaTlbFlushRequest extends Bundle {
  /** ISA only defines a complete private-TLB invalidation. */
  val all = Bool()
}
