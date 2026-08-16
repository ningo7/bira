package bira

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink.TLEdgeOut

/** Complete BIRA RoCC accelerator shell for Chipyard.
  *
  * The default 8-byte DMA beat matches the standard Rocket/Chipyard system
  * bus. All compute and ISA modules remain in the Rocket-independent hw tree.
  */
class BiRaRoCC(
  val biraParams: BiRaParams = BiRaParams(dmaBeatBytes = 8),
  opcodes: OpcodeSet = OpcodeSet.custom3
)(implicit parameters: Parameters)
    extends LazyRoCC(opcodes = opcodes, nPTWPorts = 2) {
  val dma = LazyModule(new BiRaTileLinkDma(biraParams))
  override val atlNode = dma.node
  override lazy val module = new BiRaRoCCModule(this)
}

class BiRaRoCCModule(
  outer: BiRaRoCC
)(implicit parameters: Parameters)
    extends LazyRoCCModuleImp(outer) {
  private val biraParams = outer.biraParams
  private val control =
    Module(new BiRaControlPlane(biraParams))
  private val load = Module(new BiRaLoadCtrl(biraParams))
  private val store = Module(new BiRaStoreCtrl(biraParams))
  private val readVm =
    Module(new BiRaVirtualReadDma(biraParams))
  private val writeVm =
    Module(new BiRaVirtualWriteDma(biraParams))
  private val dataPlane =
    Module(new BiRaDataPlane(biraParams))

  control.io.loadIssue <> load.io.task
  control.io.storeIssue <> store.io.task
  control.io.execIssue <> dataPlane.io.execTask
  control.io.loadCompletion <> load.io.completion
  control.io.storeCompletion <> store.io.completion
  control.io.execCompletion <> dataPlane.io.execCompletion
  load.io.contexts := control.io.contexts
  store.io.contexts := control.io.contexts
  dataPlane.io.contexts := control.io.contexts

  load.io.externalRequest <> readVm.io.request
  load.io.externalResponse <> readVm.io.response
  store.io.externalRequest <> writeVm.io.request
  store.io.externalResponse <> writeVm.io.response
  load.io.localWrite <> dataPlane.io.localWrite
  store.io.localReadRequest <> dataPlane.io.localReadRequest
  store.io.localReadResponse := dataPlane.io.localReadResponse

  readVm.io.physicalRequest <>
    outer.dma.module.io.readRequest
  readVm.io.physicalResponse <>
    outer.dma.module.io.readResponse
  writeVm.io.physicalRequest <>
    outer.dma.module.io.writeRequest
  writeVm.io.physicalResponse <>
    outer.dma.module.io.writeResponse

  implicit private val tlEdge: TLEdgeOut =
    outer.dma.node.edges.out.head
  private val loadTlb =
    Module(new BiRaRocketTlbAdapter(biraParams))
  private val storeTlb =
    Module(new BiRaRocketTlbAdapter(biraParams))
  readVm.io.translationRequest <> loadTlb.io.request
  readVm.io.translationResponse <> loadTlb.io.response
  writeVm.io.translationRequest <> storeTlb.io.request
  writeVm.io.translationResponse <> storeTlb.io.response
  io.ptw(0) <> loadTlb.io.ptw
  io.ptw(1) <> storeTlb.io.ptw

  private val flushPulse = control.io.tlbFlush.fire
  control.io.tlbFlush.ready := true.B
  loadTlb.io.flush := flushPulse
  storeTlb.io.flush := flushPulse
  control.io.tlbFlushDone.valid :=
    loadTlb.io.flushDone && storeTlb.io.flushDone
  control.io.tlbFlushDone.bits := BiRaError.none.U

  control.io.command.valid := io.cmd.valid
  control.io.command.bits.funct := io.cmd.bits.inst.funct
  control.io.command.bits.rs1 := io.cmd.bits.rs1
  control.io.command.bits.rs2 := io.cmd.bits.rs2
  control.io.command.bits.rd := io.cmd.bits.inst.rd
  control.io.command.bits.xd := io.cmd.bits.inst.xd
  control.io.command.bits.xs1 := io.cmd.bits.inst.xs1
  control.io.command.bits.xs2 := io.cmd.bits.inst.xs2
  control.io.command.bits.translationStatus := Cat(
    0.U(58.W),
    io.cmd.bits.status.mxr,
    io.cmd.bits.status.sum,
    io.cmd.bits.status.debug,
    io.cmd.bits.status.dv,
    io.cmd.bits.status.dprv
  )
  io.cmd.ready := control.io.command.ready

  io.resp.valid := control.io.response.valid
  io.resp.bits.rd := control.io.response.bits.rd
  io.resp.bits.data := control.io.response.bits.data
  control.io.response.ready := io.resp.ready

  io.busy :=
    control.io.status.loadQueueCount =/= 0.U ||
      control.io.status.execQueueCount =/= 0.U ||
      control.io.status.storeQueueCount =/= 0.U ||
      load.io.busy ||
      store.io.busy ||
      dataPlane.io.busy
  io.interrupt := false.B

  // BIRA uses its TileLink DMA node rather than the optional HellaCache port.
  io.mem.req.valid := false.B
  io.mem.s1_kill := false.B
  io.mem.s2_kill := false.B
  io.mem.keep_clock_enabled := true.B
}

/** Add this fragment instead of a Gemmini BuildRoCC fragment. */
class WithBiRaRoCC(
  biraParams: BiRaParams = BiRaParams(dmaBeatBytes = 8)
) extends Config((site, here, up) => {
  case BuildRoCC =>
    up(BuildRoCC) ++ Seq(
      (parameters: Parameters) =>
        LazyModule(
          new BiRaRoCC(
            biraParams = biraParams,
            opcodes = OpcodeSet.custom3
          )(parameters)
        )
    )
})
