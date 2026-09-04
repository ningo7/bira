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
class ControlPlane(p: AccelParams = AccelParams()) extends Module {
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new RawCmd))
    val response = Decoupled(new RawResp)

    val loadIssue = Decoupled(new DmaTask(p))
    val execIssue = Decoupled(new ExecTask(p))
    val storeIssue = Decoupled(new DmaTask(p))

    val loadCompletion =
      Flipped(Decoupled(new Completion(p)))
    val execCompletion =
      Flipped(Decoupled(new Completion(p)))
    val storeCompletion =
      Flipped(Decoupled(new Completion(p)))

    val tlbFlush = Decoupled(new FlushReq)
    val tlbFlushDone = Flipped(Valid(UInt(8.W)))

    val contexts = Output(Vec(p.nContexts, new Context(p)))
    val status = Output(new SchedStatus)
  })

  private val frontend = Module(new CmdFrontend(p))
  private val scheduler = Module(new Scheduler(p))

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
object GenControlPlane extends App {
  private val targetDir =
    args.headOption.getOrElse("build/generated-rtl")
  ChiselStage.emitSystemVerilogFile(
    new ControlPlane(),
    args = Array("--target-dir", targetDir),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info"
    )
  )
}
