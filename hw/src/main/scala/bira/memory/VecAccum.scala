package bira

// Vector accumulator storage.

import chisel3._
import chisel3.util._

/** Two-bank vector accumulator with overwrite, shifted-add, and read commands.
  *
  * Requests are processed in order. Every accepted request produces exactly
  * one response, so a controller can use the response as its completion event.
  */
class VecAccum(p: BiRaParams) extends Module {
  private val rowBits = p.bankRowBits
  private val rowType = Vec(p.dim, SInt(p.accumulatorBits.W))

  val io = IO(new Bundle {
    val request = Flipped(Decoupled(new AccumulatorRequest(p)))
    val response = Decoupled(new AccumulatorResponse(p))
  })

  private val memories =
    Seq.fill(p.accumulatorBanks)(SyncReadMem(p.bankRows, rowType))

  private val idle :: waitForRead :: respond :: Nil = Enum(3)
  private val state = RegInit(idle)
  private val requestRegister = Reg(new AccumulatorRequest(p))
  private val responseRegister = Reg(new AccumulatorResponse(p))

  io.request.ready := state === idle
  io.response.valid := state === respond
  io.response.bits := responseRegister

  val requestBank = io.request.bits.address >> rowBits
  val requestRow = io.request.bits.address(rowBits - 1, 0)
  val startRead = io.request.fire &&
    (io.request.bits.operation === AccumulatorOperation.add ||
      io.request.bits.operation === AccumulatorOperation.read)

  val readData = Wire(Vec(p.accumulatorBanks, rowType))
  for (bank <- 0 until p.accumulatorBanks) {
    readData(bank) := memories(bank).read(
      requestRow,
      startRead && requestBank === bank.U
    )
  }

  when(io.request.fire) {
    assert(requestBank < p.accumulatorBanks.U, "accumulator address out of range")
    requestRegister := io.request.bits

    when(io.request.bits.operation === AccumulatorOperation.write) {
      for (bank <- 0 until p.accumulatorBanks) {
        when(requestBank === bank.U) {
          memories(bank).write(requestRow, io.request.bits.data)
        }
      }
      responseRegister.data := io.request.bits.data
      state := respond
    }.otherwise {
      state := waitForRead
    }
  }

  when(state === waitForRead) {
    val selectedData = Mux1H(
      (0 until p.accumulatorBanks).map { bank =>
        (requestRegister.address >> rowBits) === bank.U
      },
      readData
    )

    when(requestRegister.operation === AccumulatorOperation.add) {
      val sum = Wire(rowType)
      for (lane <- 0 until p.dim) {
        val signedInput = Mux(
          requestRegister.negate,
          -requestRegister.data(lane),
          requestRegister.data(lane)
        )
        val shiftedWide = signedInput << requestRegister.shift
        val shifted = shiftedWide(p.accumulatorBits - 1, 0).asSInt
        sum(lane) := (selectedData(lane) + shifted)(p.accumulatorBits - 1, 0).asSInt
      }

      val storedBank = requestRegister.address >> rowBits
      val storedRow = requestRegister.address(rowBits - 1, 0)
      for (bank <- 0 until p.accumulatorBanks) {
        when(storedBank === bank.U) {
          memories(bank).write(storedRow, sum)
        }
      }
      responseRegister.data := sum
    }.otherwise {
      responseRegister.data := selectedData
    }
    state := respond
  }

  when(io.response.fire) {
    state := idle
  }
}
