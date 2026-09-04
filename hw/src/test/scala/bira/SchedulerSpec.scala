package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

/** Dependency and out-of-order issue tests for the bank-level scheduler. */
class SchedulerSpec extends AnyFreeSpec with Matchers {
  "independent EXEC may bypass a RAW-blocked older EXEC" in {
    val p = AccelParams(
      dim = 16,
      fullBanks = 8,
      binaryBanks = 4,
      accumulatorBanks = 2,
      bankRows = 16,
      maxImageHeight = 4,
      maxImageWidth = 4,
      maxInputChannels = 16,
      maxOutputBlocks = 1,
      nContexts = 4,
      rsEntries = 8
    )

    simulate(new Scheduler(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.loadEnqueue.valid.poke(false.B)
      dut.io.execEnqueue.valid.poke(false.B)
      dut.io.storeEnqueue.valid.poke(false.B)
      dut.io.loadIssue.ready.poke(true.B)
      dut.io.execIssue.ready.poke(true.B)
      dut.io.storeIssue.ready.poke(true.B)
      dut.io.loadCompletion.valid.poke(false.B)
      dut.io.execCompletion.valid.poke(false.B)
      dut.io.storeCompletion.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)

      def configureContext(
        index: Int,
        inputBank: Int,
        weightLowBank: Int,
        weightHighBank: Int,
        outputBank: Int,
        accumulatorBank: Int
      ): Unit = {
        val ctx = dut.io.contexts(index)
        ctx.building.poke(false.B)
        ctx.ready.poke(true.B)
        ctx.committed.poke(true.B)
        ctx.shapeValid.poke(true.B)
        ctx.modeValid.poke(true.B)
        ctx.inputHeight.poke(2.U)
        ctx.inputWidth.poke(2.U)
        ctx.inputChannels.poke(16.U)
        ctx.outputHeight.poke(2.U)
        ctx.outputWidth.poke(2.U)
        ctx.outputChannels.poke(16.U)
        ctx.kernelHeight.poke(1.U)
        ctx.kernelWidth.poke(1.U)
        ctx.paddingHeight.poke(0.U)
        ctx.paddingWidth.poke(0.U)
        ctx.arrayMode.poke(ArrayMode.dense.U)
        ctx.weightPrecision.poke(WgtPrecision.w16.U)
        ctx.inputSigned.poke(false.B)
        ctx.postMode.poke(PostMode.intRelu.U)
        ctx.shufflePack2.poke(false.B)
        ctx.writeFull.poke(true.B)
        ctx.writeBinary.poke(false.B)
        ctx.inflightCount.poke(0.U)
        ctx.errorCode.poke(ErrorCode.none.U)
        ctx.errorCommandSequence.poke(0.U)
        ctx.addressValid.poke(((1 << AddrRole.count) - 1).U)
        for (role <- 0 until AddrRole.count) {
          ctx.baseRows(role).poke(0.U)
        }
        ctx.baseRows(AddrRole.input)
          .poke((inputBank * p.bankRows).U)
        ctx.baseRows(AddrRole.weightLow)
          .poke((weightLowBank * p.bankRows).U)
        ctx.baseRows(AddrRole.weightHigh)
          .poke((weightHighBank * p.bankRows).U)
        ctx.baseRows(AddrRole.parameter).poke(0.U)
        ctx.baseRows(AddrRole.accumulator)
          .poke((accumulatorBank * p.bankRows).U)
        ctx.baseRows(AddrRole.outputFull)
          .poke((outputBank * p.bankRows).U)
      }

      // ctx0 uses Full banks 1/2/3 -> 4 and Acc bank0.
      configureContext(0, 1, 2, 3, 4, 0)
      // ctx1 uses disjoint Full banks 5/6/7 -> 0 and Acc bank1.
      configureContext(1, 5, 6, 7, 0, 1)
      configureContext(2, 1, 2, 3, 4, 0)
      configureContext(3, 1, 2, 3, 4, 0)

      dut.clock.step()
      dut.reset.poke(false.B)

      def enqueueLoad(
        contextId: Int,
        role: Int,
        sequence: Int
      ): Unit = {
        val task = dut.io.loadEnqueue.bits
        task.contextId.poke(contextId.U)
        task.role.poke(role.U)
        task.dramVirtualAddress.poke(0.U)
        task.localRowOffset.poke(0.U)
        task.rows.poke(1.U)
        task.bytesPerRow.poke(16.U)
        task.dramStrideBytes.poke(16.U)
        task.localStrideRows.poke(1.U)
        task.commandSequence.poke(sequence.U)
        task.translationStatus.poke(0.U)
        dut.io.loadEnqueue.valid.poke(true.B)
        dut.io.loadEnqueue.ready.expect(true.B)
        dut.clock.step()
        dut.io.loadEnqueue.valid.poke(false.B)
        dut.clock.step() // DMA span -> bank masks
        dut.clock.step() // registered resource preparation -> reservation entry
        dut.clock.step() // reservation selection -> registered issue stage
      }

      def enqueueExec(contextId: Int, sequence: Int): Unit = {
        dut.io.execEnqueue.bits.contextId.poke(contextId.U)
        dut.io.execEnqueue.bits.commandSequence.poke(sequence.U)
        dut.io.execEnqueue.valid.poke(true.B)
        var waitCycles = 0
        while (!dut.io.execEnqueue.ready.peek().litToBoolean) {
          dut.clock.step()
          waitCycles += 1
          assert(waitCycles < 64, "EXEC resource compilation timed out")
        }
        dut.clock.step()
        dut.io.execEnqueue.valid.poke(false.B)
        dut.clock.step() // registered resource preparation -> reservation entry
        dut.clock.step() // reservation selection -> registered issue stage
      }

      def completeLoad(contextId: Int, sequence: Int): Unit = {
        dut.io.loadCompletion.bits.contextId.poke(contextId.U)
        dut.io.loadCompletion.bits.commandSequence.poke(sequence.U)
        dut.io.loadCompletion.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.loadCompletion.valid.poke(true.B)
        dut.clock.step()
        dut.io.loadCompletion.valid.poke(false.B)
      }

      def completeExec(contextId: Int, sequence: Int): Unit = {
        dut.io.execCompletion.bits.contextId.poke(contextId.U)
        dut.io.execCompletion.bits.commandSequence.poke(sequence.U)
        dut.io.execCompletion.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.execCompletion.valid.poke(true.B)
        dut.clock.step()
        dut.io.execCompletion.valid.poke(false.B)
      }

      // Older LOAD writes ctx0.INPUT on Full bank1.
      enqueueLoad(0, AddrRole.input, sequence = 0)
      dut.io.loadIssue.valid.expect(true.B)
      dut.io.loadIssue.bits.commandSequence.expect(0.U)
      dut.clock.step() // issue LOAD, but do not complete it

      // ctx0 EXEC reads bank1 and must wait for the older LOAD.
      enqueueExec(0, sequence = 1)
      dut.io.execIssue.valid.expect(false.B)

      // ctx1 uses disjoint banks and may bypass the blocked ctx0 EXEC.
      enqueueExec(1, sequence = 2)
      dut.io.execIssue.valid.expect(true.B)
      dut.io.execIssue.bits.contextId.expect(1.U)
      dut.io.execIssue.bits.commandSequence.expect(2.U)
      dut.clock.step()

      completeExec(1, sequence = 2)
      dut.io.execIssue.valid.expect(false.B)

      // Completing the producer LOAD releases ctx0 EXEC's RAW dependency.
      completeLoad(0, sequence = 0)
      dut.clock.step() // released reservation entry -> registered issue stage
      dut.io.execIssue.valid.expect(true.B)
      dut.io.execIssue.bits.contextId.expect(0.U)
      dut.io.execIssue.bits.commandSequence.expect(1.U)
      dut.clock.step()
      completeExec(0, sequence = 1)

      dut.io.status.loadQueueCount.expect(0.U)
      dut.io.status.execQueueCount.expect(0.U)
      dut.io.status.storeQueueCount.expect(0.U)
    }
  }
}
