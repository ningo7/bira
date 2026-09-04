package bira.models.bfsrcnn

import bira._

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Paths
import scala.collection.mutable

/** Replays the exact command and DRAM images emitted by the generated C
  * software stack. No layer sequence is duplicated in Scala.
  */
class TraceReplaySpec extends AnyFreeSpec with Matchers {
  private case class TraceCommand(
    funct: Int,
    returnsValue: Boolean,
    rs1: BigInt,
    rs2: BigInt
  )

  private case class TraceImage(
    source: String,
    memoryBase: BigInt,
    commands: Seq[TraceCommand],
    memory: Array[Byte],
    inputPixels: Int,
    stages: Array[Byte],
    expected: Array[Byte]
  )

  private def unsignedLong(value: Long): BigInt =
    if (value >= 0L) BigInt(value)
    else BigInt(value & Long.MaxValue) + (BigInt(1) << 63)

  private def loadFixture(): TraceImage = {
    val configuredRoot =
      sys.props
        .get("bira.fixture")
        .orElse(sys.env.get("BIRA_FIXTURE"))
      .map(Paths.get(_))
    def readFixture(name: String): Array[Byte] =
      configuredRoot
        .map(root => Files.readAllBytes(root.resolve(name)))
        .getOrElse {
          val stream = getClass.getResourceAsStream(
            s"/bira/bfsrcnn4/$name"
          )
          require(stream != null, s"missing fixture resource $name")
          try stream.readAllBytes()
          finally stream.close()
        }
    val raw = readFixture("commands.bin")
    val buffer =
      ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
    val magic = new Array[Byte](8)
    buffer.get(magic)
    new String(magic, "US-ASCII") mustBe "BIRA_CMD"
    buffer.getInt() mustBe 1
    val commandCount = buffer.getInt()
    val memoryBase = unsignedLong(buffer.getLong())
    val commands = (0 until commandCount).map { _ =>
      val funct = buffer.get() & 0xff
      val returnsValue = (buffer.get() & 1) != 0
      buffer.position(buffer.position() + 6)
      TraceCommand(
        funct,
        returnsValue,
        unsignedLong(buffer.getLong()),
        unsignedLong(buffer.getLong())
      )
    }
    buffer.hasRemaining mustBe false
    TraceImage(
      configuredRoot
        .map(_.toAbsolutePath.toString)
        .getOrElse("classpath:/bira/bfsrcnn4"),
      memoryBase,
      commands,
      readFixture("memory.bin"),
      readFixture("input.bin").length,
      readFixture("stages.bin"),
      readFixture("expected.bin")
    )
  }

  "generated C trace with trained parameters must match the C reference" in {
    val fixture = loadFixture()
    val dram = mutable.Map.empty[BigInt, Int]
    fixture.memory.zipWithIndex.foreach { case (value, index) =>
      dram(fixture.memoryBase + index) = value & 0xff
    }

    def read(address: BigInt, count: Int): BigInt =
      (0 until count).foldLeft(BigInt(0)) { case (result, byte) =>
        val location = address + byte
        require(
          dram.contains(location),
          f"trace DRAM read from uninitialized 0x$location%x"
        )
        result | (BigInt(dram(location)) << (8 * byte))
      }

    def write(address: BigInt, count: Int, data: BigInt): Unit =
      for (byte <- 0 until count) {
        dram(address + byte) =
          ((data >> (8 * byte)) & 0xff).toInt
      }

    val p = AccelParams()
    simulate(new StandaloneTop(p)) { dut =>
      var cycleCount = 0L
      val maxCycles = math.max(
        10000000L,
        fixture.inputPixels.toLong * 100000L
      )
      var readResponse: Option[BigInt] = None
      var writeResponsePending = false
      val responses = mutable.Queue.empty[BigInt]
      val layerNames =
        Seq("head", "shrink1", "shrink2", "shrink3") ++
          (0 until 8).map(index => s"mapping$index") ++
          Seq("expand", "final")
      val layerCycles = mutable.ArrayBuffer.empty[(String, Long)]
      val debugStoreCycles = mutable.ArrayBuffer.empty[(String, Long)]

      dut.io.command.valid.poke(false.B)
      dut.io.response.ready.poke(true.B)
      dut.io.dramReadResponse.valid.poke(false.B)
      dut.io.dramWriteResponse.valid.poke(false.B)
      dut.io.dramReadRequest.ready.poke(false.B)
      dut.io.dramWriteRequest.ready.poke(false.B)

      def tick(): Boolean = {
        require(
          cycleCount < maxCycles,
          s"generated trace exceeded $maxCycles cycles"
        )
        dut.io.dramReadRequest.ready.poke(readResponse.isEmpty.B)
        dut.io.dramWriteRequest.ready.poke((!writeResponsePending).B)
        dut.io.dramReadResponse.valid.poke(readResponse.nonEmpty.B)
        dut.io.dramReadResponse.bits.data.poke(
          readResponse.getOrElse(BigInt(0)).U
        )
        dut.io.dramReadResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.dramWriteResponse.valid.poke(writeResponsePending.B)
        dut.io.dramWriteResponse.bits.errorCode.poke(ErrorCode.none.U)

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
            Some(read(address, count))
          } else None
        val newWriteResponse =
          if (writeRequestFire) {
            val address =
              dut.io.dramWriteRequest.bits.virtualAddress.peek().litValue
            val count =
              dut.io.dramWriteRequest.bits.bytes.peek().litValue.toInt
            val data = dut.io.dramWriteRequest.bits.data.peek().litValue
            write(address, count, data)
            true
          } else false
        if (responseFire) {
          responses.enqueue(dut.io.response.bits.data.peek().litValue)
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

      dut.reset.poke(true.B)
      for (_ <- 0 until 3) {
        tick()
      }
      dut.reset.poke(false.B)
      tick()

      var fenceIndex = 0
      var phaseStartCycle = cycleCount
      fixture.commands.foreach { command =>
        dut.io.command.bits.funct.poke(command.funct.U)
        dut.io.command.bits.rs1.poke(command.rs1.U)
        dut.io.command.bits.rs2.poke(command.rs2.U)
        dut.io.command.bits.rd.poke(1.U)
        dut.io.command.bits.xd.poke(command.returnsValue.B)
        dut.io.command.bits.xs1.poke(true.B)
        dut.io.command.bits.xs2.poke(true.B)
        dut.io.command.bits.translationStatus.poke(0.U)
        dut.io.command.valid.poke(true.B)
        var accepted = false
        while (!accepted) {
          accepted = tick()
        }
        dut.io.command.valid.poke(false.B)
        if (command.returnsValue) {
          while (responses.isEmpty) {
            tick()
          }
          val result = responses.dequeue()
          if (command.funct == Funct.fence) {
            (result & 1) mustBe 1
            ((result >> 1) & 0xff) mustBe ErrorCode.none
            val elapsed = cycleCount - phaseStartCycle
            if (fenceIndex < 2 * (layerNames.length - 1)) {
              val layer = layerNames(fenceIndex / 2)
              if ((fenceIndex & 1) == 0) {
                layerCycles += layer -> elapsed
              } else {
                debugStoreCycles += layer -> elapsed
              }
            } else {
              layerCycles += layerNames.last -> elapsed
            }
            fenceIndex += 1
            phaseStartCycle = cycleCount
          }
        }
      }

      fenceIndex mustBe 2 * layerNames.length - 1
      layerCycles.map(_._1).toSeq mustBe layerNames
      val stores = fixture.commands.filter(_.funct == Funct.store2d)
      val pixels = fixture.inputPixels
      val stageSizes =
        Seq(pixels * 48, pixels * 32, pixels * 32) ++
          Seq.fill(9)(pixels * 16) ++ Seq(pixels * 128)
      stores.length mustBe stageSizes.length + 1
      var expectedOffset = 0
      stageSizes.zipWithIndex.foreach { case (size, stage) =>
        val actualStage = (0 until size).map { index =>
          dram(stores(stage).rs1 + index).toByte
        }.toArray
        val expectedStage =
          fixture.stages.slice(expectedOffset, expectedOffset + size)
        withClue(s"first mismatch is in generated layer $stage: ") {
          actualStage mustBe expectedStage
        }
        expectedOffset += size
      }
      expectedOffset mustBe fixture.stages.length

      val finalStore = stores.last
      val actual = (0 until fixture.expected.length).map { index =>
        dram(finalStore.rs1 + index).toByte
      }.toArray
      actual mustBe fixture.expected
      dut.io.busy.expect(false.B)
      val computeTotal = layerCycles.map(_._2).sum
      val debugStoreTotal = debugStoreCycles.map(_._2).sum
      withClue(
        "weight prefetch and array/accumulator/post pipelines regressed: "
      ) {
        computeTotal must be < 120000L
      }
      info(s"fixture=${fixture.source}")
      layerCycles.foreach { case (layer, cycles) =>
        info(f"layer $layer%-8s $cycles%10d cycles")
      }
      info(s"profiled layer total: $computeTotal cycles")
      info(s"debug intermediate STORE total: $debugStoreTotal cycles")
      info(
        s"trained-parameter generated C trace finished in $cycleCount cycles"
      )
    }
  }
}
