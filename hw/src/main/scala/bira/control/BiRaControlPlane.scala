package bira

// Control-plane integration boundary.

import chisel3._
import chisel3.util._
import _root_.circt.stage.ChiselStage

/** ISA frontend plus bank-aware Reservation Station.
  *
  * This is the complete Rocket-independent control boundary. Future
  * LoadCtrl/ExecCtrl/StoreCtrl modules attach to the three issue/completion
  * pairs; a LazyRoCC wrapper attaches to command/response and TLB flush.
  */
class BiRaControlPlane(p: BiRaParams = BiRaParams()) extends Module {
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new BiRaRawCommand))
    val response = Decoupled(new BiRaRawResponse)

    val loadIssue = Decoupled(new BiRaDmaTask(p))
    val execIssue = Decoupled(new BiRaExecTask(p))
    val storeIssue = Decoupled(new BiRaDmaTask(p))

    val loadCompletion =
      Flipped(Decoupled(new BiRaTaskCompletion(p)))
    val execCompletion =
      Flipped(Decoupled(new BiRaTaskCompletion(p)))
    val storeCompletion =
      Flipped(Decoupled(new BiRaTaskCompletion(p)))

    val tlbFlush = Decoupled(new BiRaTlbFlushRequest)
    val tlbFlushDone = Flipped(Valid(UInt(8.W)))

    val contexts = Output(Vec(p.nContexts, new BiRaContext(p)))
    val status = Output(new BiRaSchedulerStatus)
  })

  private val frontend = Module(new BiRaCmdFrontend(p))
  private val scheduler = Module(new BiRaScheduler(p))

  frontend.io.command <> io.command
  io.response <> frontend.io.response

  scheduler.io.contexts := frontend.io.contexts
  scheduler.io.loadEnqueue <> frontend.io.loadTask
  scheduler.io.execEnqueue <> frontend.io.execTask
  scheduler.io.storeEnqueue <> frontend.io.storeTask
  frontend.io.completion <> scheduler.io.completion
  frontend.io.schedulerStatus := scheduler.io.status

  io.loadIssue <> scheduler.io.loadIssue
  io.execIssue <> scheduler.io.execIssue
  io.storeIssue <> scheduler.io.storeIssue
  scheduler.io.loadCompletion <> io.loadCompletion
  scheduler.io.execCompletion <> io.execCompletion
  scheduler.io.storeCompletion <> io.storeCompletion

  io.tlbFlush <> frontend.io.tlbFlush
  frontend.io.tlbFlushDone <> io.tlbFlushDone

  io.contexts := frontend.io.contexts
  io.status := scheduler.io.status
}

/** Generate the Rocket-independent command frontend and scheduler top. */
object GenBiRaControlPlane extends App {
  private val targetDir =
    args.headOption.getOrElse("build/generated-rtl")
  ChiselStage.emitSystemVerilogFile(
    new BiRaControlPlane(),
    args = Array("--target-dir", targetDir),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info"
    )
  )
}
