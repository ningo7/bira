package bira

// Multi-bit activation and weight fetch path.

import chisel3._
import chisel3.util._

/** Converts scratchpad rows into independent activation and weight operands.
  *
  * The two paths are intentionally decoupled. A kernel-tap weight can be loaded
  * once and kept stationary in the controller while activation elements for
  * every output pixel are read through the other path.
  */
class ConvFetch(p: AccelParams) extends Module {
  val io = IO(new Bundle {
    val actReq =
      Flipped(Decoupled(new ActivationFetchRequest(p)))
    val actResp =
      Decoupled(new ActivationFetchResponse(p))

    val wgtReq =
      Flipped(Decoupled(new WeightFetchRequest(p)))
    val wgtResp =
      Decoupled(new WeightFetchResponse(p))

    val actReadReq =
      Decoupled(UInt(p.fullAddressBits.W))
    val actReadResp =
      Flipped(Valid(Vec(p.dim, UInt(p.activationBits.W))))

    val wgtLoRdReq =
      Decoupled(UInt(p.fullAddressBits.W))
    val wgtLoRdResp =
      Flipped(Valid(Vec(p.dim, UInt(p.activationBits.W))))

    val wgtHiRdReq =
      Decoupled(UInt(p.fullAddressBits.W))
    val wgtHiRdResp =
      Flipped(Valid(Vec(p.dim, UInt(p.activationBits.W))))
  })

  // The scratchpad already has an explicit one-cycle synchronous-read
  // latency. Forward activation requests directly instead of surrounding that
  // latency with request and response holding states. ConvCtrl always enters
  // its response-wait state when this request fires, so the non-backpressured
  // SPAD response can be consumed directly one cycle later.
  private val actLane = RegEnable(io.actReq.bits.lane, io.actReq.fire)
  io.actReadReq.valid := io.actReq.valid
  io.actReadReq.bits := io.actReq.bits.address
  io.actReq.ready := io.actReadReq.ready

  io.actResp.valid := io.actReadResp.valid
  io.actResp.bits.activation :=
    io.actReadResp.bits(actLane)
  io.actResp.bits.row := io.actReadResp.bits
  when(io.actReadResp.valid) {
    assert(io.actResp.ready, "activation SPAD response must be consumed")
  }

  private val wgtActive = RegInit(false.B)
  private val wgtReqReg = Reg(new WeightFetchRequest(p))
  private val wgtLoIssued = RegInit(false.B)
  private val wgtHiIssued = RegInit(false.B)
  private val wgtLoReceived = RegInit(false.B)
  private val wgtHiReceived = RegInit(false.B)
  private val wgtLoRow = Reg(Vec(p.dim, UInt(p.activationBits.W)))
  private val wgtHiRow = Reg(Vec(p.dim, UInt(p.activationBits.W)))
  private val wgtRespValid = RegInit(false.B)

  io.wgtReq.ready := !wgtActive && !wgtRespValid

  io.wgtLoRdReq.valid := wgtActive && !wgtLoIssued
  io.wgtLoRdReq.bits := wgtReqReg.lowAddress
  io.wgtHiRdReq.valid :=
    wgtActive && !wgtHiIssued &&
      wgtReqReg.highBytePresent
  io.wgtHiRdReq.bits := wgtReqReg.highAddress

  when(io.wgtReq.fire) {
    wgtReqReg := io.wgtReq.bits
    wgtActive := true.B
    wgtLoIssued := false.B
    wgtHiIssued := !io.wgtReq.bits.highBytePresent
    wgtLoReceived := false.B
    wgtHiReceived := !io.wgtReq.bits.highBytePresent
  }

  when(io.wgtLoRdReq.fire) {
    wgtLoIssued := true.B
  }
  when(io.wgtHiRdReq.fire) {
    wgtHiIssued := true.B
  }

  when(io.wgtLoRdResp.valid) {
    wgtLoRow := io.wgtLoRdResp.bits
    wgtLoReceived := true.B
  }
  when(io.wgtHiRdResp.valid) {
    wgtHiRow := io.wgtHiRdResp.bits
    wgtHiReceived := true.B
  }

  val allWgtRespPresent =
    (wgtLoReceived || io.wgtLoRdResp.valid) &&
      (wgtHiReceived || io.wgtHiRdResp.valid)

  when(wgtActive && allWgtRespPresent) {
    wgtActive := false.B
    wgtRespValid := true.B
  }

  io.wgtResp.valid := wgtRespValid
  for (lane <- 0 until p.dim) {
    io.wgtResp.bits.weights(lane) :=
      Cat(
        Mux(
          wgtReqReg.highBytePresent,
          wgtHiRow(lane),
          0.U
        ),
        wgtLoRow(lane)
      )
  }

  when(io.wgtResp.fire) {
    wgtRespValid := false.B
  }
}
