package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Tests the Rocket-independent ISA decoder and Context lifecycle. */
class CmdFrontendSpec extends AnyFreeSpec with Matchers {
  "frontend must build a Context, track inflight work, fence, status, and flush" in {
    val p = AccelParams(
      dim = 16,
      fullBanks = 4,
      binaryBanks = 4,
      accumulatorBanks = 2,
      bankRows = 64,
      maxImageHeight = 8,
      maxImageWidth = 8,
      maxInputChannels = 16,
      maxOutputBlocks = 2,
      nContexts = 4
    )

    simulate(new CmdFrontend(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.command.valid.poke(false.B)
      dut.io.response.ready.poke(false.B)
      dut.io.loadTask.ready.poke(true.B)
      dut.io.execTask.ready.poke(true.B)
      dut.io.storeTask.ready.poke(true.B)
      dut.io.completion.valid.poke(false.B)
      dut.io.tlbFlush.ready.poke(true.B)
      dut.io.tlbFlushDone.valid.poke(false.B)
      dut.io.tlbFlushDone.bits.poke(ErrorCode.none.U)
      dut.io.schedulerStatus.loadQueueCount.poke(0.U)
      dut.io.schedulerStatus.execQueueCount.poke(0.U)
      dut.io.schedulerStatus.storeQueueCount.poke(0.U)
      dut.io.schedulerStatus.loadBusy.poke(false.B)
      dut.io.schedulerStatus.execBusy.poke(false.B)
      dut.io.schedulerStatus.storeBusy.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      def driveCommand(
        funct: Int,
        rs1: BigInt,
        rs2: BigInt,
        rd: Int = 0,
        xd: Boolean = false
      ): Unit = {
        dut.io.command.bits.funct.poke(funct.U)
        dut.io.command.bits.rs1.poke(rs1.U)
        dut.io.command.bits.rs2.poke(rs2.U)
        dut.io.command.bits.rd.poke(rd.U)
        dut.io.command.bits.xd.poke(xd.B)
        dut.io.command.bits.xs1.poke(true.B)
        dut.io.command.bits.xs2.poke(true.B)
        dut.io.command.bits.translationStatus.poke("h1234".U)
        dut.io.command.valid.poke(true.B)
        while (!dut.io.command.ready.peek().litToBoolean) {
          dut.clock.step()
        }
        dut.clock.step()
        dut.io.command.valid.poke(false.B)
      }

      val contextId = 1
      val shapeRs1 =
        BigInt(contextId) |
          (BigInt(4) << 3) |
          (BigInt(4) << 15) |
          (BigInt(1) << 27) |
          (BigInt(4) << 39)
      val shapeRs2 =
        BigInt(4) |
          (BigInt(16) << 12) |
          (BigInt(3) << 24) |
          (BigInt(3) << 28) |
          (BigInt(1) << 32) |
          (BigInt(1) << 36)
      driveCommand(Funct.cfgShape, shapeRs1, shapeRs2)

      def configureAddress(role: Int, baseRow: Int): Unit = {
        val packed =
          BigInt(contextId) |
            (BigInt(role) << 3) |
            (BigInt(baseRow) << 7)
        driveCommand(Funct.cfgAddr, packed, 0)
      }

      configureAddress(AddrRole.input, 0)
      configureAddress(AddrRole.weightLow, 64)
      configureAddress(AddrRole.weightHigh, 128)
      configureAddress(AddrRole.parameter, 0)
      configureAddress(AddrRole.accumulator, 0)
      configureAddress(AddrRole.outputFull, 192)

      val modeRs1 =
        BigInt(contextId) |
          (BigInt(ArrayMode.dense) << 3) |
          (BigInt(WgtPrecision.w16) << 5) |
          (BigInt(PostMode.intRelu) << 8) |
          (BigInt(1) << 12)
      driveCommand(Funct.cfgMode, modeRs1, 0)
      driveCommand(Funct.cfgCommit, contextId, 0)
      while (!dut.io.command.ready.peek().litToBoolean) {
        dut.clock.step()
      }

      dut.io.contexts(contextId).ready.expect(true.B)
      dut.io.contexts(contextId).committed.expect(true.B)
      dut.io.contexts(contextId).errorCode.expect(ErrorCode.none.U)

      val loadRows = 4
      val loadBytes = 16
      val loadDescriptor =
        BigInt(contextId) |
          (BigInt(AddrRole.input) << 3) |
          (BigInt(0) << 7) |
          (BigInt(loadRows) << 21) |
          (BigInt(loadBytes) << 35) |
          (BigInt(loadBytes) << 42) |
          (BigInt(1) << 58)

      dut.io.command.bits.funct.poke(Funct.load2d.U)
      dut.io.command.bits.rs1.poke("h80001000".U)
      dut.io.command.bits.rs2.poke(loadDescriptor.U)
      dut.io.command.bits.rd.poke(0.U)
      dut.io.command.bits.xd.poke(false.B)
      dut.io.command.bits.xs1.poke(true.B)
      dut.io.command.bits.xs2.poke(true.B)
      dut.io.command.bits.translationStatus.poke("h1234".U)
      dut.io.command.valid.poke(true.B)
      dut.io.command.ready.expect(true.B)
      dut.io.loadTask.valid.expect(true.B)
      dut.io.loadTask.bits.contextId.expect(contextId.U)
      dut.io.loadTask.bits.role.expect(AddrRole.input.U)
      dut.io.loadTask.bits.dramVirtualAddress.expect("h80001000".U)
      dut.io.loadTask.bits.rows.expect(loadRows.U)
      dut.io.loadTask.bits.bytesPerRow.expect(loadBytes.U)
      dut.io.loadTask.bits.dramStrideBytes.expect(loadBytes.U)
      dut.io.loadTask.bits.localStrideRows.expect(1.U)
      val loadSequence =
        dut.io.loadTask.bits.commandSequence.peek().litValue
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      dut.io.contexts(contextId).inflightCount.expect(1.U)

      // FENCE(CONTEXT) must not return before the accepted LOAD completes.
      val fenceRequest = BigInt(1) | (BigInt(contextId) << 1)
      driveCommand(
        Funct.fence,
        fenceRequest,
        0,
        rd = 5,
        xd = true
      )
      dut.io.response.valid.expect(false.B)
      dut.io.command.ready.expect(false.B)

      dut.io.completion.bits.contextId.poke(contextId.U)
      dut.io.completion.bits.commandSequence.poke(loadSequence.U)
      dut.io.completion.bits.errorCode.poke(ErrorCode.none.U)
      dut.io.completion.valid.poke(true.B)
      dut.clock.step()
      dut.io.completion.valid.poke(false.B)
      dut.clock.step()

      dut.io.response.valid.expect(true.B)
      dut.io.response.bits.rd.expect(5.U)
      (dut.io.response.bits.data.peek().litValue & 1) mustBe 1
      dut.io.response.ready.poke(true.B)
      dut.clock.step()
      dut.io.response.ready.poke(false.B)
      dut.io.contexts(contextId).inflightCount.expect(0.U)

      // STATUS(CONTEXT) returns the committed/ready snapshot.
      val statusRequest = BigInt(1) | (BigInt(contextId) << 1)
      driveCommand(
        Funct.status,
        statusRequest,
        0,
        rd = 6,
        xd = true
      )
      dut.io.response.valid.expect(true.B)
      dut.io.response.bits.rd.expect(6.U)
      val status = dut.io.response.bits.data.peek().litValue
      ((status >> 0) & 1) mustBe 1 // implemented
      ((status >> 2) & 1) mustBe 1 // ready
      ((status >> 3) & 1) mustBe 1 // committed
      ((status >> 4) & 0xff) mustBe 0 // inflightCount
      dut.io.response.ready.poke(true.B)
      dut.clock.step()
      dut.io.response.ready.poke(false.B)

      // TLB_FLUSH is a blocking response command and delegates invalidation to
      // the future Rocket/PTW adapter.
      driveCommand(
        Funct.tlbFlush,
        0,
        0,
        rd = 7,
        xd = true
      )
      dut.clock.step()
      dut.io.tlbFlush.valid.expect(false.B)
      // The request was accepted in the preceding cycle.
      dut.io.tlbFlushDone.bits.poke(ErrorCode.none.U)
      dut.io.tlbFlushDone.valid.poke(true.B)
      dut.clock.step()
      dut.io.tlbFlushDone.valid.poke(false.B)
      dut.io.response.valid.expect(true.B)
      dut.io.response.bits.rd.expect(7.U)
      dut.io.response.bits.data.expect(ErrorCode.none.U)
    }
  }

  "FENCE on an unimplemented Context must return an error instead of waiting" in {
    val p = AccelParams(
      dim = 4,
      fullBanks = 3,
      binaryBanks = 1,
      accumulatorBanks = 1,
      bankRows = 16,
      maxImageHeight = 2,
      maxImageWidth = 2,
      maxInputChannels = 4,
      maxOutputBlocks = 1,
      nContexts = 4
    )

    simulate(new CmdFrontend(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.command.valid.poke(false.B)
      dut.io.response.ready.poke(false.B)
      dut.io.loadTask.ready.poke(true.B)
      dut.io.execTask.ready.poke(true.B)
      dut.io.storeTask.ready.poke(true.B)
      dut.io.completion.valid.poke(false.B)
      dut.io.tlbFlush.ready.poke(true.B)
      dut.io.tlbFlushDone.valid.poke(false.B)
      dut.io.tlbFlushDone.bits.poke(0.U)
      dut.io.schedulerStatus.loadQueueCount.poke(0.U)
      dut.io.schedulerStatus.execQueueCount.poke(0.U)
      dut.io.schedulerStatus.storeQueueCount.poke(0.U)
      dut.io.schedulerStatus.loadBusy.poke(false.B)
      dut.io.schedulerStatus.execBusy.poke(false.B)
      dut.io.schedulerStatus.storeBusy.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      val invalidContext = 7
      val request = BigInt(1) | (BigInt(invalidContext) << 1)
      dut.io.command.bits.funct.poke(Funct.fence.U)
      dut.io.command.bits.rs1.poke(request.U)
      dut.io.command.bits.rs2.poke(0.U)
      dut.io.command.bits.rd.poke(3.U)
      dut.io.command.bits.xd.poke(true.B)
      dut.io.command.bits.xs1.poke(true.B)
      dut.io.command.bits.xs2.poke(true.B)
      dut.io.command.bits.translationStatus.poke(0.U)
      dut.io.command.valid.poke(true.B)
      dut.io.command.ready.expect(true.B)
      dut.clock.step()
      dut.io.command.valid.poke(false.B)
      dut.clock.step()

      dut.io.response.valid.expect(true.B)
      dut.io.response.bits.rd.expect(3.U)
      val result = dut.io.response.bits.data.peek().litValue
      (result & 1) mustBe 0
      ((result >> 1) & 0xff) mustBe ErrorCode.invalidContext
      ((result >> 9) & 0x7) mustBe invalidContext
    }
  }
}
