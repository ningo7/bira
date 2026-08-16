package bira

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.rocket.constants.MemoryOpConstants
import freechips.rocketchip.tile.CoreModule
import freechips.rocketchip.tilelink._

/** TileLink endpoint below the Rocket-independent virtual DMA bridges.
  *
  * Source 0 is reserved for reads and source 1 for writes. Each bridge has at
  * most one request outstanding, while LOAD and STORE may proceed in parallel.
  */
class BiRaTileLinkDma(
  biraParams: BiRaParams
)(implicit parameters: Parameters)
    extends LazyModule {
  val node = TLClientNode(
    Seq(
      TLMasterPortParameters.v1(
        Seq(
          TLMasterParameters.v1(
            name = "bira-dma",
            sourceId = IdRange(0, 2)
          )
        )
      )
    )
  )

  override lazy val module =
    new BiRaTileLinkDmaModule(this, biraParams)
}

class BiRaTileLinkDmaModule(
  outer: BiRaTileLinkDma,
  biraParams: BiRaParams
)(implicit parameters: Parameters)
    extends LazyModuleImp(outer) {
    val io = IO(new Bundle {
      val readRequest =
        Flipped(Decoupled(new BiRaPhysicalReadRequest))
      val readResponse =
        Decoupled(new BiRaPhysicalReadResponse(biraParams))
      val writeRequest =
        Flipped(
          Decoupled(new BiRaPhysicalWriteRequest(biraParams))
        )
      val writeResponse =
        Decoupled(new BiRaPhysicalWriteResponse)
    })

    private val (tl, edge) = outer.node.out.head
    private val beatBytes = edge.manager.beatBytes
    require(
      beatBytes == biraParams.dmaBeatBytes,
      s"BIRA dmaBeatBytes=${biraParams.dmaBeatBytes} must equal TileLink beatBytes=$beatBytes"
    )
    private val lgBeatBytes = log2Ceil(beatBytes)

    private val readA = edge.Get(
      fromSource = 0.U,
      toAddress = io.readRequest.bits.physicalAddress,
      lgSize = lgBeatBytes.U
    )._2
    private val writeA = edge.Put(
      fromSource = 1.U,
      toAddress = io.writeRequest.bits.physicalAddress,
      lgSize = lgBeatBytes.U,
      data = io.writeRequest.bits.data,
      mask = io.writeRequest.bits.mask
    )._2

    // LOAD has priority only for this A-channel cycle. Since it must wait for
    // its D response before requesting another beat, STORE cannot starve.
    private val chooseRead = io.readRequest.valid
    tl.a.valid := io.readRequest.valid || io.writeRequest.valid
    tl.a.bits := Mux(chooseRead, readA, writeA)
    io.readRequest.ready := tl.a.ready && chooseRead
    io.writeRequest.ready :=
      tl.a.ready && !chooseRead && io.writeRequest.valid

    private val responseIsRead = tl.d.bits.source === 0.U
    private val responseIsWrite = tl.d.bits.source === 1.U
    private val responseFailed =
      tl.d.bits.denied || tl.d.bits.corrupt

    io.readResponse.valid := tl.d.valid && responseIsRead
    io.readResponse.bits.data := tl.d.bits.data
    io.readResponse.bits.errorCode :=
      Mux(responseFailed, BiRaError.access.U, BiRaError.none.U)

    io.writeResponse.valid := tl.d.valid && responseIsWrite
    io.writeResponse.bits.errorCode :=
      Mux(responseFailed, BiRaError.access.U, BiRaError.none.U)

    tl.d.ready := Mux(
      responseIsRead,
      io.readResponse.ready,
      io.writeResponse.ready
    )

    when(tl.d.valid) {
      assert(
        responseIsRead || responseIsWrite,
        "TileLink returned an unallocated BIRA DMA source ID"
      )
    }

    tl.b.ready := true.B
    tl.c.valid := false.B
    tl.c.bits := DontCare
    tl.e.valid := false.B
    tl.e.bits := DontCare
}

/** Adapter from BIRA's stable translation protocol to one Rocket private TLB.
  *
  * Translation status encoding carried by BiRaDmaTask:
  *   [1:0] dprv, [2] dv, [3] debug, [4] sum, [5] mxr.
  */
class BiRaRocketTlbAdapter(
  biraParams: BiRaParams,
  entries: Int = 4
)(implicit edge: TLEdgeOut, parameters: Parameters)
    extends CoreModule
    with MemoryOpConstants {
  require(entries > 0)

  val io = IO(new Bundle {
    val request =
      Flipped(Decoupled(new BiRaTranslationRequest))
    val response = Decoupled(new BiRaTranslationResponse)
    val ptw = new TLBPTWIO
    val flush = Input(Bool())
    val flushDone = Output(Bool())
  })

  private val tlb = Module(
    new TLB(
      instruction = false,
      lgMaxSize = log2Ceil(biraParams.dmaBeatBytes),
      cfg = TLBConfig(nSets = 1, nWays = entries)
    )
  )
  private val idle :: lookup :: respond :: Nil = Enum(3)
  private val state = RegInit(idle)
  private val requestReg = Reg(new BiRaTranslationRequest)
  private val responseReg = Reg(new BiRaTranslationResponse)

  io.request.ready := state === idle && !io.flush
  when(io.request.fire) {
    requestReg := io.request.bits
    state := lookup
  }

  private val alignedVirtualAddress = Cat(
    requestReg.virtualAddress(63, biraParams.dmaBeatOffsetBits),
    0.U(biraParams.dmaBeatOffsetBits.W)
  )
  private val originalBeatOffset =
    requestReg.virtualAddress(
      biraParams.dmaBeatOffsetBits - 1,
      0
    )

  tlb.io.req.valid := state === lookup
  tlb.io.req.bits.vaddr :=
    alignedVirtualAddress(vaddrBitsExtended - 1, 0)
  tlb.io.req.bits.passthrough := false.B
  tlb.io.req.bits.size := log2Ceil(biraParams.dmaBeatBytes).U
  tlb.io.req.bits.cmd :=
    Mux(requestReg.isWrite, M_XWR, M_XRD)
  tlb.io.req.bits.prv := requestReg.translationStatus(1, 0)
  tlb.io.req.bits.v := requestReg.translationStatus(2)
  tlb.io.kill := false.B

  private val capturedStatus =
    WireDefault(0.U.asTypeOf(new MStatus))
  capturedStatus.dprv := requestReg.translationStatus(1, 0)
  capturedStatus.dv := requestReg.translationStatus(2)
  capturedStatus.debug := requestReg.translationStatus(3)
  capturedStatus.sum := requestReg.translationStatus(4)
  capturedStatus.mxr := requestReg.translationStatus(5)

  io.ptw <> tlb.io.ptw
  tlb.io.ptw.status := capturedStatus

  private val loadPageFault =
    tlb.io.resp.pf.ld || tlb.io.resp.gf.ld
  private val storePageFault =
    tlb.io.resp.pf.st || tlb.io.resp.gf.st
  private val loadAccessFault =
    tlb.io.resp.ae.ld || tlb.io.resp.ma.ld
  private val storeAccessFault =
    tlb.io.resp.ae.st || tlb.io.resp.ma.st
  private val pageFault =
    Mux(requestReg.isWrite, storePageFault, loadPageFault)
  private val accessFault =
    Mux(requestReg.isWrite, storeAccessFault, loadAccessFault)

  when(
    state === lookup &&
      tlb.io.req.ready &&
      !tlb.io.resp.miss
  ) {
    responseReg.physicalAddress :=
      tlb.io.resp.paddr.pad(64) + originalBeatOffset
    responseReg.errorCode := Mux(
      pageFault,
      BiRaError.tlb.U,
      Mux(accessFault, BiRaError.access.U, BiRaError.none.U)
    )
    state := respond
  }

  io.response.valid := state === respond
  io.response.bits := responseReg
  when(io.response.fire) {
    state := idle
  }

  tlb.io.sfence.valid := io.flush
  tlb.io.sfence.bits.rs1 := false.B
  tlb.io.sfence.bits.rs2 := false.B
  tlb.io.sfence.bits.addr := 0.U
  tlb.io.sfence.bits.asid := 0.U
  tlb.io.sfence.bits.hv := false.B
  tlb.io.sfence.bits.hg := false.B
  io.flushDone := RegNext(io.flush, false.B)

  when(io.flush) {
    assert(state === idle, "BIRA TLB may only flush after DMA drains")
  }
}
