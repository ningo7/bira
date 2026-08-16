package bira

// Standalone simulation and RTL-generation top.

import chisel3._
import chisel3.util._
import _root_.circt.stage.ChiselStage

/** Rocket-independent BIRA system top.
  *
  * This module contains the complete command frontend, reservation station,
  * 2-D load/store engines, parameter buffer, scratchpads, accumulator, and
  * compute core. A simulator or another non-Rocket host only has to provide:
  *
  *   - decoded BIRA ISA command fields; and
  *   - a byte-addressed memory model behind the row-level DRAM ports.
  *
  * The external memory protocol intentionally stays above virtual-address
  * translation and bus-beat splitting. Chipyard inserts the TLB/PTW and
  * TileLink bridges below the same row requests; standalone simulation can
  * instead treat the address as a direct byte address.
  */
class BiRaStandaloneTop(p: BiRaParams = BiRaParams()) extends Module {
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new BiRaRawCommand))
    val response = Decoupled(new BiRaRawResponse)

    val dramReadRequest =
      Decoupled(new BiRaExternalReadRequest)
    val dramReadResponse =
      Flipped(Decoupled(new BiRaExternalReadResponse))
    val dramWriteRequest =
      Decoupled(new BiRaExternalWriteRequest)
    val dramWriteResponse =
      Flipped(Decoupled(new BiRaExternalWriteResponse))

    val schedulerStatus = Output(new BiRaSchedulerStatus)
    val busy = Output(Bool())
  })

  private val control = Module(new BiRaControlPlane(p))
  private val load = Module(new BiRaLoadCtrl(p))
  private val store = Module(new BiRaStoreCtrl(p))
  private val dataPlane = Module(new BiRaDataPlane(p))

  control.io.loadIssue <> load.io.task
  control.io.execIssue <> dataPlane.io.execTask
  control.io.storeIssue <> store.io.task
  control.io.loadCompletion <> load.io.completion
  control.io.execCompletion <> dataPlane.io.execCompletion
  control.io.storeCompletion <> store.io.completion

  load.io.contexts := control.io.contexts
  store.io.contexts := control.io.contexts
  dataPlane.io.contexts := control.io.contexts

  load.io.localWrite <> dataPlane.io.localWrite
  store.io.localReadRequest <> dataPlane.io.localReadRequest
  store.io.localReadResponse := dataPlane.io.localReadResponse

  io.dramReadRequest <> load.io.externalRequest
  load.io.externalResponse <> io.dramReadResponse
  io.dramWriteRequest <> store.io.externalRequest
  store.io.externalResponse <> io.dramWriteResponse

  control.io.command <> io.command
  io.response <> control.io.response

  // Standalone simulation has no private TLB. A flush is acknowledged on the
  // following cycle after all older work has drained, which preserves the ISA
  // frontend's blocking response semantics without pretending to translate.
  private val flushAccepted = control.io.tlbFlush.fire
  private val flushDone = RegNext(flushAccepted, false.B)
  control.io.tlbFlush.ready := true.B
  control.io.tlbFlushDone.valid := flushDone
  control.io.tlbFlushDone.bits := BiRaError.none.U

  io.schedulerStatus := control.io.status
  io.busy :=
    control.io.status.loadQueueCount =/= 0.U ||
      control.io.status.execQueueCount =/= 0.U ||
      control.io.status.storeQueueCount =/= 0.U ||
      load.io.busy ||
      store.io.busy ||
      dataPlane.io.busy
}

/** Generate the complete non-Rocket top for a C++ Verilator harness. */
object GenBiRaStandaloneTop extends App {
  private val targetDir =
    args.headOption.getOrElse("build/generated-rtl")
  ChiselStage.emitSystemVerilogFile(
    new BiRaStandaloneTop(),
    args = Array("--target-dir", targetDir),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info"
    )
  )
}
