package bira

// Vector accumulator storage.

import chisel3._
import chisel3.util._

/** Banked vector accumulator implemented as a pipelined 1R1W memory.
  *
  * One request may be accepted every cycle. Read and add requests launch the
  * synchronous read in the acceptance cycle; one cycle later an add writes
  * its result through the independent write port. Writes use the same
  * writeback stage, so every operation remains ordered and produces exactly
  * one response.
  *
  * The two response entries are an interface skid buffer, not an execution
  * queue. They only preserve Decoupled semantics when a consumer briefly
  * removes ready; with a continuously-ready consumer the initiation interval
  * is one cycle.
  */
class VecAccum(p: AccelParams) extends Module {
  private val rowBits = p.bankRowBits
  private val rowType = Vec(p.dim, SInt(p.accumulatorBits.W))

  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new AccumulatorRequest(p)))
    val response = Decoupled(new AccumulatorResponse(p))
  })

  private val memories =
    Seq.fill(p.accumulatorBanks)(SyncReadMem(p.bankRows, rowType))
  private val pendingValid = RegInit(false.B)
  private val pending = Reg(new AccumulatorRequest(p))

  private val responses = Module(
    new Queue(new AccumulatorResponse(p), entries = 2, pipe = true, flow = true)
  )
  io.response <> responses.io.deq

  // Reserve one response slot for the request already in the SRAM pipeline.
  // Account for a same-cycle dequeue so a ready consumer sustains II=1.
  private val occupiedAfterDequeue =
    responses.io.count - responses.io.deq.fire.asUInt
  private val reservedResponses =
    occupiedAfterDequeue + pendingValid.asUInt
  private val requestNeedsRead =
    io.request.bits.operation =/= AccumulatorOperation.write
  private val pendingWrites =
    pending.operation =/= AccumulatorOperation.read
  // SyncReadMem read-during-write data is target-dependent. Stall only a true
  // same-row dependency; independent rows retain II=1.
  private val sameRowHazard =
    pendingValid &&
      pendingWrites &&
      requestNeedsRead &&
      pending.address === io.request.bits.address
  io.request.ready := reservedResponses < 2.U && !sameRowHazard

  private val requestBank = io.request.bits.address >> rowBits
  private val requestRow = io.request.bits.address(rowBits - 1, 0)
  private val startRead = io.request.fire &&
    (io.request.bits.operation === AccumulatorOperation.add ||
      io.request.bits.operation === AccumulatorOperation.read)

  private val readData = Wire(Vec(p.accumulatorBanks, rowType))
  for (bank <- 0 until p.accumulatorBanks) {
    readData(bank) := memories(bank).read(
      requestRow,
      startRead && requestBank === bank.U
    )
  }

  when(io.request.fire) {
    assert(requestBank < p.accumulatorBanks.U, "accumulator address out of range")
    pending := io.request.bits
  }
  pendingValid := io.request.fire

  private val pendingBank = pending.address >> rowBits
  private val pendingRow = pending.address(rowBits - 1, 0)
  private val selectedData = Mux1H(
    (0 until p.accumulatorBanks).map(bank => pendingBank === bank.U),
    readData
  )
  private val result = Wire(rowType)

  for (lane <- 0 until p.dim) {
    val signedInput = Mux(
      pending.negate,
      -pending.data(lane),
      pending.data(lane)
    )
    val shiftedWide = signedInput << pending.shift
    val shifted = shiftedWide(p.accumulatorBits - 1, 0).asSInt
    val added =
      (selectedData(lane) + shifted)(p.accumulatorBits - 1, 0).asSInt
    result(lane) := Mux(
      pending.operation === AccumulatorOperation.write,
      pending.data(lane),
      Mux(
        pending.operation === AccumulatorOperation.add,
        added,
        selectedData(lane)
      )
    )
  }

  responses.io.enq.valid := pendingValid
  responses.io.enq.bits.data := result
  responses.io.enq.bits.completesOutput := pending.completesOutput
  responses.io.enq.bits.block := pending.block
  responses.io.enq.bits.pixel := pending.pixel
  responses.io.enq.bits.outputX := pending.outputX
  responses.io.enq.bits.outputY := pending.outputY
  when(pendingValid) {
    assert(responses.io.enq.ready, "accumulator response skid overflow")
  }

  when(pendingValid && pendingWrites) {
    for (bank <- 0 until p.accumulatorBanks) {
      when(pendingBank === bank.U) {
        memories(bank).write(pendingRow, result)
      }
    }
  }
}
