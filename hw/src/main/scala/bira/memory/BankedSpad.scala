package bira

// Banked activation and weight scratchpad.

import chisel3._
import chisel3.util._

/** Banked, vector-row scratchpad with fixed-priority arbitration per bank.
  *
  * Reads are synchronous. A request is accepted on `req.fire`; exactly one
  * cycle later the corresponding Valid response is asserted. Responses cannot
  * be backpressured, which keeps this internal SRAM interface small and makes
  * its latency explicit.
  *
  * Different read ports may proceed in parallel when they select different
  * banks. Requests to the same bank are serialized by port priority.
  */
class BankedSpad(
  banks: Int,
  rowsPerBank: Int,
  lanes: Int,
  elementBits: Int,
  readPorts: Int
) extends Module {
  require(banks > 0)
  require(rowsPerBank > 0 && isPow2(rowsPerBank))
  require(lanes > 0)
  require(elementBits > 0)
  require(readPorts > 0)

  private val rowBits = log2Ceil(rowsPerBank)
  private val addressBits = log2Ceil(banks * rowsPerBank)
  private val rowType = Vec(lanes, UInt(elementBits.W))

  val io = IO(new Bundle {
    val readRequest = Vec(readPorts, Flipped(Decoupled(UInt(addressBits.W))))
    val readResponse = Vec(readPorts, Valid(rowType))
    val writeRequest = Flipped(
      Decoupled(new VectorWriteRequest(addressBits, lanes, elementBits))
    )
  })

  private val memories = Seq.fill(banks)(SyncReadMem(rowsPerBank, rowType))
  private val bankReadData = Wire(Vec(banks, rowType))
  private val delayedGrant = Wire(Vec(banks, Vec(readPorts, Bool())))

  io.readRequest.foreach(_.ready := false.B)

  for (bank <- 0 until banks) {
    val requests = VecInit(io.readRequest.map { request =>
      request.valid && (request.bits >> rowBits) === bank.U
    })
    val grant = PriorityEncoderOH(requests)
    val readEnable = requests.asUInt.orR
    val readRow = Mux1H(
      grant,
      io.readRequest.map(_.bits(rowBits - 1, 0))
    )

    for (port <- 0 until readPorts) {
      when(grant(port)) {
        io.readRequest(port).ready := true.B
      }
      delayedGrant(bank)(port) := RegNext(grant(port), false.B)
    }

    bankReadData(bank) := memories(bank).read(readRow, readEnable)
  }

  for (port <- 0 until readPorts) {
    val responseSelect = VecInit(
      (0 until banks).map(bank => delayedGrant(bank)(port))
    )
    io.readResponse(port).valid := responseSelect.asUInt.orR
    io.readResponse(port).bits := Mux1H(responseSelect, bankReadData)
  }

  val writeBank = io.writeRequest.bits.address >> rowBits
  val writeRow = io.writeRequest.bits.address(rowBits - 1, 0)
  io.writeRequest.ready := writeBank < banks.U

  when(io.writeRequest.fire) {
    for (bank <- 0 until banks) {
      when(writeBank === bank.U) {
        memories(bank).write(writeRow, io.writeRequest.bits.data)
      }
    }
  }
}
