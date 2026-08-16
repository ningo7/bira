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
class ConvFetch(p: BiRaParams) extends Module {
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

  private val actActive = RegInit(false.B)
  private val actReqReg =
    Reg(new ActivationFetchRequest(p))
  private val actIssued = RegInit(false.B)
  private val actRow = Reg(Vec(p.dim, UInt(p.activationBits.W)))
  private val actRespValid = RegInit(false.B)

  io.actReq.ready :=
    !actActive && !actRespValid
  io.actReadReq.valid := actActive && !actIssued
  io.actReadReq.bits := actReqReg.address

  when(io.actReq.fire) {
    actReqReg := io.actReq.bits
    actActive := true.B
    actIssued := false.B
  }

  when(io.actReadReq.fire) {
    actIssued := true.B
  }

  when(io.actReadResp.valid) {
    actRow := io.actReadResp.bits
    actActive := false.B
    actRespValid := true.B
  }

  io.actResp.valid := actRespValid
  io.actResp.bits.activation :=
    actRow(actReqReg.lane)
  io.actResp.bits.row := actRow

  when(io.actResp.fire) {
    actRespValid := false.B
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
