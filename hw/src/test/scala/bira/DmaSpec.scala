package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class DmaSpec extends AnyFreeSpec with Matchers {
  private val p = AccelParams(
    dim = 16,
    fullBanks = 4,
    binaryBanks = 2,
    accumulatorBanks = 2,
    bankRows = 16,
    parameterRows = 32,
    maxImageHeight = 4,
    maxImageWidth = 4,
    maxInputChannels = 16,
    maxOutputBlocks = 2,
    nContexts = 4
  )

  private def initializeContext(
    context: Context,
    inputBase: Int = 16,
    outputFullBase: Int = 32,
    outputBinaryBase: Int = 8
  ): Unit = {
    context.building.poke(false.B)
    context.ready.poke(true.B)
    context.committed.poke(true.B)
    context.shapeValid.poke(true.B)
    context.modeValid.poke(true.B)
    context.inputHeight.poke(2.U)
    context.inputWidth.poke(2.U)
    context.inputChannels.poke(16.U)
    context.outputHeight.poke(2.U)
    context.outputWidth.poke(2.U)
    context.outputChannels.poke(16.U)
    context.kernelHeight.poke(1.U)
    context.kernelWidth.poke(1.U)
    context.paddingHeight.poke(0.U)
    context.paddingWidth.poke(0.U)
    context.addressValid.poke(((1 << AddrRole.count) - 1).U)
    for (role <- 0 until AddrRole.count) {
      context.baseRows(role).poke(0.U)
    }
    context.baseRows(AddrRole.input).poke(inputBase.U)
    context.baseRows(AddrRole.outputFull).poke(outputFullBase.U)
    context.baseRows(AddrRole.outputBinary)
      .poke(outputBinaryBase.U)
    context.arrayMode.poke(ArrayMode.dense.U)
    context.weightPrecision.poke(WgtPrecision.w16.U)
    context.inputSigned.poke(false.B)
    context.postMode.poke(PostMode.intRelu.U)
    context.shufflePack2.poke(false.B)
    context.writeFull.poke(true.B)
    context.writeBinary.poke(false.B)
    context.inflightCount.poke(1.U)
    context.errorCode.poke(ErrorCode.none.U)
    context.errorCommandSequence.poke(0.U)
  }

  private def pokeTask(
    task: DmaTask,
    role: Int,
    virtualAddress: BigInt,
    rows: Int,
    bytes: Int,
    localOffset: Int,
    localStride: Int,
    externalStride: Int,
    sequence: Int
  ): Unit = {
    task.contextId.poke(0.U)
    task.role.poke(role.U)
    task.dramVirtualAddress.poke(virtualAddress.U)
    task.localRowOffset.poke(localOffset.U)
    task.rows.poke(rows.U)
    task.bytesPerRow.poke(bytes.U)
    task.dramStrideBytes.poke(externalStride.U)
    task.localStrideRows.poke(localStride.U)
    task.commandSequence.poke(sequence.U)
    task.translationStatus.poke("h1234".U)
  }

  "LOAD_2D walks both strides, zero-fills a short row, and rejects width errors" in {
    simulate(new LoadCtrl(p)) { dut =>
      dut.reset.poke(true.B)
      for (index <- 0 until p.nContexts) {
        initializeContext(dut.io.contexts(index))
      }
      dut.io.task.valid.poke(false.B)
      dut.io.externalRequest.ready.poke(true.B)
      dut.io.externalResponse.valid.poke(false.B)
      dut.io.localWrite.ready.poke(true.B)
      dut.io.completion.ready.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      pokeTask(
        dut.io.task.bits,
        role = AddrRole.input,
        virtualAddress = 0x1000,
        rows = 2,
        bytes = 4,
        localOffset = 1,
        localStride = 2,
        externalStride = 8,
        sequence = 7
      )
      dut.io.task.valid.poke(true.B)
      dut.io.task.ready.expect(true.B)
      dut.clock.step()
      dut.io.task.valid.poke(false.B)
      dut.clock.step() // pipelined descriptor range validation

      def returnExternalRow(expectedAddress: BigInt, data: BigInt): Unit = {
        dut.io.externalRequest.valid.expect(true.B)
        dut.io.externalRequest.bits.virtualAddress
          .expect(expectedAddress.U)
        dut.io.externalRequest.bits.bytes.expect(4.U)
        dut.io.externalRequest.bits.translationStatus.expect("h1234".U)
        dut.clock.step()

        dut.io.externalResponse.bits.data.poke(data.U)
        dut.io.externalResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.externalResponse.valid.poke(true.B)
        dut.io.externalResponse.ready.expect(true.B)
        dut.clock.step()
        dut.io.externalResponse.valid.poke(false.B)
      }

      returnExternalRow(
        expectedAddress = 0x1000,
        data = (BigInt(1) << 512) - 1
      )
      dut.io.localWrite.valid.expect(true.B)
      dut.io.localWrite.bits.memory.expect(LocalMem.full.U)
      dut.io.localWrite.bits.address.expect(17.U)
      dut.io.localWrite.bits.data.expect(BigInt("ffffffff", 16).U)
      dut.clock.step()

      returnExternalRow(
        expectedAddress = 0x1008,
        data = BigInt("deadbeef", 16)
      )
      dut.io.localWrite.valid.expect(true.B)
      dut.io.localWrite.bits.address.expect(19.U)
      dut.io.localWrite.bits.data.expect(BigInt("deadbeef", 16).U)
      dut.clock.step()

      dut.io.completion.valid.expect(true.B)
      dut.io.completion.bits.contextId.expect(0.U)
      dut.io.completion.bits.commandSequence.expect(7.U)
      dut.io.completion.bits.errorCode.expect(ErrorCode.none.U)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step()
      dut.io.completion.ready.poke(false.B)

      // A Binary SPAD row is only two bytes.
      pokeTask(
        dut.io.task.bits,
        role = AddrRole.outputBinary,
        virtualAddress = 0x2000,
        rows = 1,
        bytes = 3,
        localOffset = 0,
        localStride = 1,
        externalStride = 3,
        sequence = 8
      )
      dut.io.task.valid.poke(true.B)
      dut.io.task.ready.expect(true.B)
      dut.clock.step()
      dut.io.task.valid.poke(false.B)
      dut.clock.step() // pipelined descriptor range validation
      dut.io.externalRequest.valid.expect(false.B)
      dut.io.completion.valid.expect(true.B)
      dut.io.completion.bits.commandSequence.expect(8.U)
      dut.io.completion.bits.errorCode.expect(ErrorCode.rowTooWide.U)
    }
  }

  "STORE_2D waits for local data and every external write response" in {
    simulate(new StoreCtrl(p)) { dut =>
      dut.reset.poke(true.B)
      for (index <- 0 until p.nContexts) {
        initializeContext(dut.io.contexts(index))
      }
      dut.io.task.valid.poke(false.B)
      dut.io.localReadRequest.ready.poke(true.B)
      dut.io.localReadResponse.valid.poke(false.B)
      dut.io.externalRequest.ready.poke(true.B)
      dut.io.externalResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      pokeTask(
        dut.io.task.bits,
        role = AddrRole.outputFull,
        virtualAddress = 0x3000,
        rows = 2,
        bytes = 3,
        localOffset = 1,
        localStride = 2,
        externalStride = 5,
        sequence = 11
      )
      dut.io.task.valid.poke(true.B)
      dut.io.task.ready.expect(true.B)
      dut.clock.step()
      dut.io.task.valid.poke(false.B)
      dut.clock.step() // pipelined descriptor range validation

      def provideLocalRow(
        expectedAddress: Int,
        data: BigInt
      ): Unit = {
        dut.io.localReadRequest.valid.expect(true.B)
        dut.io.localReadRequest.bits.memory
          .expect(LocalMem.full.U)
        dut.io.localReadRequest.bits.address
          .expect(expectedAddress.U)
        dut.clock.step()

        dut.io.localReadResponse.bits.data.poke(data.U)
        dut.io.localReadResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.localReadResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.localReadResponse.valid.poke(false.B)
      }

      def acceptExternalRow(
        expectedAddress: BigInt,
        expectedData: BigInt
      ): Unit = {
        dut.io.externalRequest.valid.expect(true.B)
        dut.io.externalRequest.bits.virtualAddress
          .expect(expectedAddress.U)
        dut.io.externalRequest.bits.bytes.expect(3.U)
        dut.io.externalRequest.bits.data.expect(expectedData.U)
        dut.clock.step()

        // Completion must wait for the actual external write response.
        dut.io.completion.valid.expect(false.B)
        dut.io.externalResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.externalResponse.valid.poke(true.B)
        dut.io.externalResponse.ready.expect(true.B)
        dut.clock.step()
        dut.io.externalResponse.valid.poke(false.B)
      }

      provideLocalRow(33, BigInt("11223344", 16))
      acceptExternalRow(0x3000, BigInt("223344", 16))
      provideLocalRow(35, BigInt("a1b2c3d4", 16))
      acceptExternalRow(0x3005, BigInt("b2c3d4", 16))

      dut.io.completion.valid.expect(true.B)
      dut.io.completion.bits.commandSequence.expect(11.U)
      dut.io.completion.bits.errorCode.expect(ErrorCode.none.U)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step()
      dut.io.busy.expect(false.B)
    }
  }
}
