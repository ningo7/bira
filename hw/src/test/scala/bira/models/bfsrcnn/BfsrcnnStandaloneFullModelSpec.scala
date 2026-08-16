package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.collection.mutable

/** Full-network-topology regression through the standalone ISA and DMA shell.
  *
  * The spatial input is intentionally 1x1 so all fourteen BFSRCNN layers can
  * run in a normal Verilator regression. Weights and post-process parameters
  * are deterministic synthetic values; the layer shapes, precision modes,
  * SPAD mappings, binary/full ping-pong, PixelShuffle, bilinear residual, and
  * final STORE follow docs/models/bfsrcnn.md.
  */
class BfsrcnnStandaloneFullModelSpec
    extends AnyFreeSpec
    with Matchers {

  private case class Blob(address: BigInt, rows: Int, bytesPerRow: Int)

  private case class LocalAddresses(
    input: Int,
    weightLow: Int,
    weightHigh: Option[Int],
    parameter: Int,
    residual: Option[Int],
    correction: Option[Int],
    accumulator: Int,
    outputFull: Option[Int],
    outputBinary: Option[Int]
  )

  private final class ByteDram {
    private val bytes = mutable.Map.empty[BigInt, Int]
    private var nextAddress = BigInt("10000", 16)

    private def align64(value: BigInt): BigInt =
      (value + 63) & ~BigInt(63)

    def allocate(data: Seq[Int]): BigInt = {
      val base = align64(nextAddress)
      data.zipWithIndex.foreach { case (value, index) =>
        bytes(base + index) = value & 0xff
      }
      nextAddress = base + data.length + 64
      base
    }

    def allocateRows(
      rows: Int,
      bytesPerRow: Int
    )(row: Int => Seq[Int]): Blob = {
      require(rows > 0)
      require(bytesPerRow > 0 && bytesPerRow <= 64)
      val data = (0 until rows).flatMap { index =>
        val value = row(index)
        require(value.length == bytesPerRow)
        value
      }
      Blob(allocate(data), rows, bytesPerRow)
    }

    def read(address: BigInt, count: Int): BigInt = {
      require(count > 0 && count <= 64)
      (0 until count).foldLeft(BigInt(0)) { case (result, byte) =>
        val location = address + byte
        require(
          bytes.contains(location),
          f"standalone DRAM read from uninitialized 0x$location%x"
        )
        result | (BigInt(bytes(location)) << (8 * byte))
      }
    }

    def write(address: BigInt, count: Int, data: BigInt): Unit = {
      require(count > 0 && count <= 64)
      for (byte <- 0 until count) {
        bytes(address + byte) =
          ((data >> (8 * byte)) & 0xff).toInt
      }
    }

    def readBytes(address: BigInt, count: Int): Seq[Int] =
      (0 until count).map(index => bytes(address + index))
  }

  private def field(value: BigInt, width: Int): BigInt =
    value & ((BigInt(1) << width) - 1)

  private def littleEndian(value: BigInt, bytes: Int): Seq[Int] =
    (0 until bytes).map(index =>
      ((value >> (8 * index)) & 0xff).toInt
    )

  private def multiRecord(
    bias: Int,
    qMin: Int,
    qMax: Int,
    binaryThreshold: Int = 0
  ): BigInt =
    field(bias, 32) |
      (field(0, 8) << 32) |
      (field(1, 2) << 40) |
      (field(0, 2) << 42) |
      (field(0, 5) << 44) |
      (field(0, 5) << 49) |
      (field(0, 8) << 54) |
      (field(qMin, 32) << 62) |
      (field(qMax, 32) << 94) |
      (field(binaryThreshold, 32) << 126)

  private def binaryRecord(
    threshold: Int,
    qMin: Int,
    qMax: Int,
    outputSignThreshold: Int
  ): BigInt =
    field(threshold, 32) |
      (field(0, 2) << 32) |
      (field(0, 5) << 34) |
      (field(0, 5) << 39) |
      (field(0, 8) << 44) |
      (field(0, 32) << 52) |
      (field(0, 2) << 84) |
      (field(0, 2) << 86) |
      (field(0, 5) << 88) |
      (field(0, 5) << 93) |
      (field(0, 8) << 98) |
      (field(0, 32) << 106) |
      (field(qMin, 32) << 138) |
      (field(qMax, 32) << 170) |
      (field(outputSignThreshold, 32) << 202)

  private def parameterRows(
    channelCount: Int,
    laneRecord: Int => BigInt,
    dim: Int
  ): Seq[Seq[Int]] = {
    require(dim == 16, "ISA full-model regression uses the native 16 lanes")
    val blocks = (channelCount + dim - 1) / dim
    (0 until blocks).flatMap { block =>
      (0 until dim / 2).map { pair =>
        val evenChannel = block * dim + pair * 2
        val oddChannel = evenChannel + 1
        val low =
          if (evenChannel < channelCount) laneRecord(evenChannel)
          else BigInt(0)
        val high =
          if (oddChannel < channelCount) laneRecord(oddChannel)
          else BigInt(0)
        littleEndian(low | (high << 256), 64)
      }
    }
  }

  "standalone ISA and DRAM shell must execute the complete BFSRCNN topology" in {
    val p = BiRaParams(
      dim = 16,
      fullBanks = 10,
      binaryBanks = 4,
      accumulatorBanks = 2,
      bankRows = 512,
      parameterRows = 512,
      maxImageHeight = 4,
      maxImageWidth = 4,
      maxInputChannels = 64,
      maxOutputBlocks = 8
    )

    val dram = new ByteDram
    val zeroFullRow = Seq.fill(16)(0)
    val zeroParameterRow = Seq.fill(64)(0)

    def zeroFullRows(rows: Int): Blob =
      dram.allocateRows(rows, 16)(_ => zeroFullRow)

    def parameterBlob(
      channels: Int,
      record: Int => BigInt
    ): Blob = {
      val rows = parameterRows(channels, record, p.dim)
      dram.allocateRows(rows.length, 64)(rows)
    }

    val input = dram.allocateRows(1, 1)(_ => Seq(7))
    val headWeightLow = zeroFullRows(27)
    val headWeightHigh = zeroFullRows(27)
    val headParameters =
      parameterBlob(48, _ => multiRecord(1, -128, 127))

    val shrink1Weight = zeroFullRows(96)
    val shrink1Parameters =
      parameterBlob(32, _ => multiRecord(2, 0, 255))

    val shrink2WeightLow = zeroFullRows(18)
    val shrink2WeightHigh = zeroFullRows(18)
    val shrink2Parameters =
      parameterBlob(32, _ => multiRecord(3, 0, 255))

    val shrink3WeightLow = zeroFullRows(32)
    val shrink3Parameters =
      parameterBlob(
        16,
        _ => multiRecord(0, -128, 127, binaryThreshold = 0)
      )

    val mappingWeights = Seq.fill(8)(
      dram.allocateRows(144, 2)(_ => Seq(0xff, 0xff))
    )
    val mappingParameters = Seq.fill(8)(
      parameterBlob(
        16,
        _ => binaryRecord(
          threshold = 17,
          qMin = -128,
          qMax = 127,
          outputSignThreshold = 0
        )
      )
    )
    val correction = dram.allocateRows(1, 64) { _ =>
      (0 until p.dim).flatMap(_ =>
        littleEndian(field(-16, 32), 4)
      )
    }

    val expandWeightLow = zeroFullRows(128)
    val expandParameters =
      parameterBlob(128, _ => multiRecord(4, -128, 127))

    val finalWeightLow = zeroFullRows(9)
    val finalWeightHigh = zeroFullRows(9)
    val finalParameters = parameterBlob(
      1,
      channel =>
        if (channel == 0) multiRecord(0, 0, 255)
        else BigInt(0)
    )
    val outputAddress = dram.allocate(Seq.fill(16)(0))

    val f = Array.tabulate(10)(_ * p.bankRows)
    val b = Array.tabulate(4)(_ * p.bankRows)
    val a = Array.tabulate(2)(_ * p.bankRows)
    val originalInputBase = f(9) + 256
    val parameterBase = 0
    val correctionBase = 256

    simulate(new BiRaStandaloneTop(p)) { dut =>
      var cycleCount = 0L
      val maxCycles = 1000000L
      var readResponse: Option[BigInt] = None
      var writeResponsePending = false
      val responses = mutable.Queue.empty[(BigInt, BigInt)]

      dut.io.command.valid.poke(false.B)
      dut.io.response.ready.poke(true.B)
      dut.io.dramReadResponse.valid.poke(false.B)
      dut.io.dramWriteResponse.valid.poke(false.B)
      dut.io.dramReadRequest.ready.poke(false.B)
      dut.io.dramWriteRequest.ready.poke(false.B)

      def tick(): Boolean = {
        require(
          cycleCount < maxCycles,
          s"standalone full-model simulation exceeded $maxCycles cycles"
        )

        dut.io.dramReadRequest.ready.poke(readResponse.isEmpty.B)
        dut.io.dramWriteRequest.ready.poke((!writeResponsePending).B)

        dut.io.dramReadResponse.valid.poke(readResponse.nonEmpty.B)
        dut.io.dramReadResponse.bits.data.poke(
          readResponse.getOrElse(BigInt(0)).U
        )
        dut.io.dramReadResponse.bits.errorCode.poke(BiRaError.none.U)
        dut.io.dramWriteResponse.valid.poke(writeResponsePending.B)
        dut.io.dramWriteResponse.bits.errorCode.poke(BiRaError.none.U)

        val commandFire =
          dut.io.command.valid.peek().litToBoolean &&
            dut.io.command.ready.peek().litToBoolean
        val readRequestFire =
          dut.io.dramReadRequest.valid.peek().litToBoolean &&
            dut.io.dramReadRequest.ready.peek().litToBoolean
        val readResponseFire =
          dut.io.dramReadResponse.valid.peek().litToBoolean &&
            dut.io.dramReadResponse.ready.peek().litToBoolean
        val writeRequestFire =
          dut.io.dramWriteRequest.valid.peek().litToBoolean &&
            dut.io.dramWriteRequest.ready.peek().litToBoolean
        val writeResponseFire =
          dut.io.dramWriteResponse.valid.peek().litToBoolean &&
            dut.io.dramWriteResponse.ready.peek().litToBoolean
        val responseFire =
          dut.io.response.valid.peek().litToBoolean &&
            dut.io.response.ready.peek().litToBoolean

        val newReadResponse =
          if (readRequestFire) {
            val address =
              dut.io.dramReadRequest.bits.virtualAddress.peek().litValue
            val count =
              dut.io.dramReadRequest.bits.bytes.peek().litValue.toInt
            Some(dram.read(address, count))
          } else None

        val newWriteResponse =
          if (writeRequestFire) {
            val address =
              dut.io.dramWriteRequest.bits.virtualAddress.peek().litValue
            val count =
              dut.io.dramWriteRequest.bits.bytes.peek().litValue.toInt
            val data =
              dut.io.dramWriteRequest.bits.data.peek().litValue
            dram.write(address, count, data)
            true
          } else false

        if (responseFire) {
          responses.enqueue(
            dut.io.response.bits.rd.peek().litValue ->
              dut.io.response.bits.data.peek().litValue
          )
        }

        dut.clock.step()
        cycleCount += 1

        if (readResponseFire) {
          readResponse = None
        }
        newReadResponse.foreach(value => readResponse = Some(value))
        if (writeResponseFire) {
          writeResponsePending = false
        }
        if (newWriteResponse) {
          writeResponsePending = true
        }
        commandFire
      }

      def reset(): Unit = {
        dut.reset.poke(true.B)
        for (_ <- 0 until 3) {
          tick()
        }
        dut.reset.poke(false.B)
        tick()
      }

      def send(
        funct: Int,
        rs1: BigInt,
        rs2: BigInt,
        returnsValue: Boolean = false
      ): Unit = {
        dut.io.command.bits.funct.poke(funct.U)
        dut.io.command.bits.rs1.poke(rs1.U)
        dut.io.command.bits.rs2.poke(rs2.U)
        dut.io.command.bits.rd.poke(1.U)
        dut.io.command.bits.xd.poke(returnsValue.B)
        dut.io.command.bits.xs1.poke(true.B)
        dut.io.command.bits.xs2.poke(true.B)
        dut.io.command.bits.translationStatus.poke(0.U)
        dut.io.command.valid.poke(true.B)
        var accepted = false
        while (!accepted) {
          accepted = tick()
        }
        dut.io.command.valid.poke(false.B)
      }

      def awaitResponse(): BigInt = {
        while (responses.isEmpty) {
          tick()
        }
        val (rd, data) = responses.dequeue()
        rd mustBe 1
        data
      }

      def cfgShape(
        context: Int,
        inputHeight: Int,
        inputWidth: Int,
        inputChannels: Int,
        outputHeight: Int,
        outputWidth: Int,
        outputChannels: Int,
        kernelHeight: Int,
        kernelWidth: Int,
        paddingHeight: Int,
        paddingWidth: Int
      ): Unit = {
        val rs1 =
          BigInt(context) |
            (BigInt(inputHeight) << 3) |
            (BigInt(inputWidth) << 15) |
            (BigInt(inputChannels) << 27) |
            (BigInt(outputHeight) << 39)
        val rs2 =
          BigInt(outputWidth) |
            (BigInt(outputChannels) << 12) |
            (BigInt(kernelHeight) << 24) |
            (BigInt(kernelWidth) << 28) |
            (BigInt(paddingHeight) << 32) |
            (BigInt(paddingWidth) << 36)
        send(BiRaFunct.cfgShape, rs1, rs2)
      }

      def cfgAddress(context: Int, role: Int, baseRow: Int): Unit = {
        val packed =
          BigInt(context) |
            (BigInt(role) << 3) |
            (BigInt(baseRow) << 7)
        send(BiRaFunct.cfgAddr, packed, 0)
      }

      def cfgMode(
        context: Int,
        arrayMode: Int,
        weightPrecision: Int,
        inputSigned: Boolean,
        postMode: Int,
        shufflePack2: Boolean,
        writeFull: Boolean,
        writeBinary: Boolean
      ): Unit = {
        val packed =
          BigInt(context) |
            (BigInt(arrayMode) << 3) |
            (BigInt(weightPrecision) << 5) |
            (if (inputSigned) BigInt(1) << 7 else BigInt(0)) |
            (BigInt(postMode) << 8) |
            (if (shufflePack2) BigInt(1) << 11 else BigInt(0)) |
            (if (writeFull) BigInt(1) << 12 else BigInt(0)) |
            (if (writeBinary) BigInt(1) << 13 else BigInt(0))
        send(BiRaFunct.cfgMode, packed, 0)
      }

      def configure(
        context: Int,
        inputHeight: Int,
        inputWidth: Int,
        inputChannels: Int,
        outputHeight: Int,
        outputWidth: Int,
        outputChannels: Int,
        kernelHeight: Int,
        kernelWidth: Int,
        padding: Int,
        addresses: LocalAddresses,
        arrayMode: Int,
        weightPrecision: Int,
        inputSigned: Boolean,
        postMode: Int,
        shufflePack2: Boolean = false,
        writeFull: Boolean = true,
        writeBinary: Boolean = false
      ): Unit = {
        cfgShape(
          context,
          inputHeight,
          inputWidth,
          inputChannels,
          outputHeight,
          outputWidth,
          outputChannels,
          kernelHeight,
          kernelWidth,
          padding,
          padding
        )
        cfgAddress(context, BiRaAddrRole.input, addresses.input)
        cfgAddress(context, BiRaAddrRole.weightLow, addresses.weightLow)
        addresses.weightHigh.foreach(row =>
          cfgAddress(context, BiRaAddrRole.weightHigh, row)
        )
        cfgAddress(context, BiRaAddrRole.parameter, addresses.parameter)
        addresses.residual.foreach(row =>
          cfgAddress(context, BiRaAddrRole.residual, row)
        )
        addresses.correction.foreach(row =>
          cfgAddress(context, BiRaAddrRole.correction, row)
        )
        cfgAddress(
          context,
          BiRaAddrRole.accumulator,
          addresses.accumulator
        )
        addresses.outputFull.foreach(row =>
          cfgAddress(context, BiRaAddrRole.outputFull, row)
        )
        addresses.outputBinary.foreach(row =>
          cfgAddress(context, BiRaAddrRole.outputBinary, row)
        )
        cfgMode(
          context,
          arrayMode,
          weightPrecision,
          inputSigned,
          postMode,
          shufflePack2,
          writeFull,
          writeBinary
        )
        send(BiRaFunct.cfgCommit, context, 0)
      }

      def dmaDescriptor(context: Int, role: Int, blob: Blob): BigInt =
        BigInt(context) |
          (BigInt(role) << 3) |
          (BigInt(0) << 7) |
          (BigInt(blob.rows) << 21) |
          (BigInt(blob.bytesPerRow) << 35) |
          (BigInt(blob.bytesPerRow) << 42) |
          (BigInt(1) << 58)

      def load(context: Int, role: Int, blob: Blob): Unit =
        send(
          BiRaFunct.load2d,
          blob.address,
          dmaDescriptor(context, role, blob)
        )

      def execute(context: Int): Unit =
        send(BiRaFunct.execConv, context, 0)

      def fence(context: Int): Unit = {
        send(
          BiRaFunct.fence,
          BigInt(1) | (BigInt(context) << 1),
          0,
          returnsValue = true
        )
        val result = awaitResponse()
        (result & 1) mustBe 1
        ((result >> 1) & 0xff) mustBe BiRaError.none
      }

      def runLayer(
        context: Int,
        loads: Seq[(Int, Blob)]
      ): Unit = {
        loads.foreach { case (role, blob) =>
          load(context, role, blob)
        }
        execute(context)
        fence(context)
      }

      reset()

      // The standalone top acknowledges TLB_FLUSH without a Rocket TLB.
      send(BiRaFunct.tlbFlush, 0, 0, returnsValue = true)
      awaitResponse() mustBe BiRaError.none

      configure(
        context = 0,
        inputHeight = 1,
        inputWidth = 1,
        inputChannels = 1,
        outputHeight = 1,
        outputWidth = 1,
        outputChannels = 48,
        kernelHeight = 3,
        kernelWidth = 3,
        padding = 1,
        addresses = LocalAddresses(
          originalInputBase,
          f(8),
          Some(f(8) + 27),
          parameterBase,
          None,
          None,
          a(1),
          Some(f(1)),
          None
        ),
        arrayMode = BiRaArrayMode.dense,
        weightPrecision = BiRaWeightPrecision.w16,
        inputSigned = false,
        postMode = BiRaPostMode.intPrelu
      )
      runLayer(
        0,
        Seq(
          BiRaAddrRole.input -> input,
          BiRaAddrRole.weightLow -> headWeightLow,
          BiRaAddrRole.weightHigh -> headWeightHigh,
          BiRaAddrRole.parameter -> headParameters
        )
      )

      configure(
        context = 1,
        inputHeight = 1,
        inputWidth = 1,
        inputChannels = 48,
        outputHeight = 1,
        outputWidth = 1,
        outputChannels = 32,
        kernelHeight = 1,
        kernelWidth = 1,
        padding = 0,
        addresses = LocalAddresses(
          f(1),
          f(9),
          None,
          parameterBase,
          None,
          None,
          a(1),
          Some(f(4)),
          None
        ),
        arrayMode = BiRaArrayMode.dense,
        weightPrecision = BiRaWeightPrecision.w4,
        inputSigned = true,
        postMode = BiRaPostMode.intRelu
      )
      runLayer(
        1,
        Seq(
          BiRaAddrRole.weightLow -> shrink1Weight,
          BiRaAddrRole.parameter -> shrink1Parameters
        )
      )

      configure(
        context = 2,
        inputHeight = 1,
        inputWidth = 1,
        inputChannels = 32,
        outputHeight = 1,
        outputWidth = 1,
        outputChannels = 32,
        kernelHeight = 3,
        kernelWidth = 3,
        padding = 1,
        addresses = LocalAddresses(
          f(4),
          f(8),
          Some(f(8) + 18),
          parameterBase,
          None,
          None,
          a(1),
          Some(f(6)),
          None
        ),
        arrayMode = BiRaArrayMode.depthwise,
        weightPrecision = BiRaWeightPrecision.w16,
        inputSigned = false,
        postMode = BiRaPostMode.intRelu
      )
      runLayer(
        2,
        Seq(
          BiRaAddrRole.weightLow -> shrink2WeightLow,
          BiRaAddrRole.weightHigh -> shrink2WeightHigh,
          BiRaAddrRole.parameter -> shrink2Parameters
        )
      )

      configure(
        context = 3,
        inputHeight = 1,
        inputWidth = 1,
        inputChannels = 32,
        outputHeight = 1,
        outputWidth = 1,
        outputChannels = 16,
        kernelHeight = 1,
        kernelWidth = 1,
        padding = 0,
        addresses = LocalAddresses(
          f(6),
          f(9),
          None,
          parameterBase,
          None,
          None,
          a(1),
          Some(f(0)),
          Some(b(0))
        ),
        arrayMode = BiRaArrayMode.dense,
        weightPrecision = BiRaWeightPrecision.w8,
        inputSigned = false,
        postMode = BiRaPostMode.intSignedSign,
        writeBinary = true
      )
      runLayer(
        3,
        Seq(
          BiRaAddrRole.weightLow -> shrink3WeightLow,
          BiRaAddrRole.parameter -> shrink3Parameters
        )
      )

      for (mapping <- 0 until 8) {
        val context = mapping % p.nContexts
        val even = mapping % 2 == 0
        configure(
          context = context,
          inputHeight = 1,
          inputWidth = 1,
          inputChannels = 16,
          outputHeight = 1,
          outputWidth = 1,
          outputChannels = 16,
          kernelHeight = 3,
          kernelWidth = 3,
          padding = 1,
          addresses = LocalAddresses(
            if (even) b(0) else b(1),
            if (even) b(2) else b(3),
            None,
            parameterBase,
            Some(if (even) f(0) else f(1)),
            Some(correctionBase),
            a(1),
            Some(if (even) f(1) else f(0)),
            if (mapping != 7) Some(if (even) b(1) else b(0))
            else None
          ),
          arrayMode = BiRaArrayMode.binary,
          weightPrecision = BiRaWeightPrecision.w2,
          inputSigned = false,
          postMode = BiRaPostMode.binaryFused,
          writeBinary = mapping != 7
        )
        val commonLoads =
          Seq(
            BiRaAddrRole.weightLow -> mappingWeights(mapping),
            BiRaAddrRole.parameter -> mappingParameters(mapping)
          )
        val loads =
          if (mapping == 0)
            commonLoads :+ (BiRaAddrRole.correction -> correction)
          else commonLoads
        runLayer(context, loads)
      }

      configure(
        context = 0,
        inputHeight = 1,
        inputWidth = 1,
        inputChannels = 16,
        outputHeight = 1,
        outputWidth = 1,
        outputChannels = 128,
        kernelHeight = 1,
        kernelWidth = 1,
        padding = 0,
        addresses = LocalAddresses(
          f(0),
          f(9),
          None,
          parameterBase,
          None,
          None,
          a(1),
          Some(f(1)),
          None
        ),
        arrayMode = BiRaArrayMode.dense,
        weightPrecision = BiRaWeightPrecision.w8,
        inputSigned = true,
        postMode = BiRaPostMode.intPrelu,
        shufflePack2 = true
      )
      runLayer(
        0,
        Seq(
          BiRaAddrRole.weightLow -> expandWeightLow,
          BiRaAddrRole.parameter -> expandParameters
        )
      )

      configure(
        context = 1,
        inputHeight = 4,
        inputWidth = 4,
        inputChannels = 8,
        outputHeight = 4,
        outputWidth = 4,
        outputChannels = 1,
        kernelHeight = 3,
        kernelWidth = 3,
        padding = 1,
        addresses = LocalAddresses(
          f(1),
          f(9),
          Some(f(9) + 9),
          parameterBase,
          Some(originalInputBase),
          None,
          a(1),
          Some(f(0)),
          None
        ),
        arrayMode = BiRaArrayMode.columnReduce,
        weightPrecision = BiRaWeightPrecision.w16,
        inputSigned = true,
        postMode = BiRaPostMode.finalBilinearResidual
      )
      runLayer(
        1,
        Seq(
          BiRaAddrRole.weightLow -> finalWeightLow,
          BiRaAddrRole.weightHigh -> finalWeightHigh,
          BiRaAddrRole.parameter -> finalParameters
        )
      )

      val outputBlob = Blob(outputAddress, rows = 1, bytesPerRow = 16)
      send(
        BiRaFunct.store2d,
        outputAddress,
        dmaDescriptor(1, BiRaAddrRole.outputFull, outputBlob)
      )
      fence(1)

      dram.readBytes(outputAddress, 16) mustBe Seq.fill(16)(7)

      send(BiRaFunct.status, 0, 0, returnsValue = true)
      val globalStatus = awaitResponse()
      ((globalStatus >> 6) & 1) mustBe 0
      dut.io.busy.expect(false.B)

      info(s"complete standalone model topology finished in $cycleCount cycles")
    }
  }
}
