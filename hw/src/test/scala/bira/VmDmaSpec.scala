package bira

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class VmDmaSpec extends AnyFreeSpec with Matchers {
  private val p = AccelParams(
    dmaBeatBytes = 8,
    pageBytes = 16
  )

  private def packBytes(bytes: Seq[Int]): BigInt =
    bytes.zipWithIndex.foldLeft(BigInt(0)) {
      case (value, (byte, index)) =>
        value | (BigInt(byte & 0xff) << (8 * index))
    }

  "read bridge splits at a page, aligns beats, and reassembles bytes" in {
    simulate(new VmReadDma(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.request.valid.poke(false.B)
      dut.io.response.ready.poke(false.B)
      dut.io.translationRequest.ready.poke(true.B)
      dut.io.translationResponse.valid.poke(false.B)
      dut.io.physicalRequest.ready.poke(true.B)
      dut.io.physicalResponse.valid.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      dut.io.request.bits.virtualAddress.poke("he".U)
      dut.io.request.bits.bytes.poke(6.U)
      dut.io.request.bits.translationStatus.poke("h55".U)
      dut.io.request.valid.poke(true.B)
      dut.io.request.ready.expect(true.B)
      dut.clock.step()
      dut.io.request.valid.poke(false.B)

      def translate(expectedVirtual: BigInt, physical: BigInt): Unit = {
        dut.io.translationRequest.valid.expect(true.B)
        dut.io.translationRequest.bits.virtualAddress
          .expect(expectedVirtual.U)
        dut.io.translationRequest.bits.isWrite.expect(false.B)
        dut.io.translationRequest.bits.translationStatus.expect("h55".U)
        dut.clock.step()

        dut.io.translationResponse.bits.physicalAddress.poke(physical.U)
        dut.io.translationResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.translationResponse.valid.poke(true.B)
        dut.io.translationResponse.ready.expect(true.B)
        dut.clock.step()
        dut.io.translationResponse.valid.poke(false.B)
      }

      def returnBeat(expectedPhysical: BigInt, bytes: Seq[Int]): Unit = {
        dut.io.physicalRequest.valid.expect(true.B)
        dut.io.physicalRequest.bits.physicalAddress
          .expect(expectedPhysical.U)
        dut.clock.step()

        dut.io.physicalResponse.bits.data.poke(packBytes(bytes).U)
        dut.io.physicalResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.physicalResponse.valid.poke(true.B)
        dut.io.physicalResponse.ready.expect(true.B)
        dut.clock.step()
        dut.io.physicalResponse.valid.poke(false.B)
      }

      // VA 0x0e has two bytes left in its page and in the 0x28 beat.
      translate(expectedVirtual = 0x0e, physical = 0x2e)
      returnBeat(
        expectedPhysical = 0x28,
        bytes = Seq(0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7)
      )

      // The next virtual page maps to a non-contiguous physical page.
      translate(expectedVirtual = 0x10, physical = 0x40)
      returnBeat(
        expectedPhysical = 0x40,
        bytes = Seq(0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7)
      )

      dut.io.response.valid.expect(true.B)
      dut.io.response.bits.errorCode.expect(ErrorCode.none.U)
      dut.io.response.bits.data.expect(
        packBytes(Seq(0xa6, 0xa7, 0xb0, 0xb1, 0xb2, 0xb3)).U
      )
      dut.io.response.ready.poke(true.B)
      dut.clock.step()
    }
  }

  "write bridge splits at a page and creates aligned partial-write masks" in {
    simulate(new VmWriteDma(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.request.valid.poke(false.B)
      dut.io.response.ready.poke(false.B)
      dut.io.translationRequest.ready.poke(true.B)
      dut.io.translationResponse.valid.poke(false.B)
      dut.io.physicalRequest.ready.poke(true.B)
      dut.io.physicalResponse.valid.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      dut.io.request.bits.virtualAddress.poke("hf".U)
      dut.io.request.bits.bytes.poke(4.U)
      dut.io.request.bits.data.poke(
        packBytes(Seq(0x11, 0x22, 0x33, 0x44)).U
      )
      dut.io.request.bits.translationStatus.poke("haa".U)
      dut.io.request.valid.poke(true.B)
      dut.io.request.ready.expect(true.B)
      dut.clock.step()
      dut.io.request.valid.poke(false.B)

      def translate(expectedVirtual: BigInt, physical: BigInt): Unit = {
        dut.io.translationRequest.valid.expect(true.B)
        dut.io.translationRequest.bits.virtualAddress
          .expect(expectedVirtual.U)
        dut.io.translationRequest.bits.isWrite.expect(true.B)
        dut.io.translationRequest.bits.translationStatus.expect("haa".U)
        dut.clock.step()

        dut.io.translationResponse.bits.physicalAddress.poke(physical.U)
        dut.io.translationResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.translationResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.translationResponse.valid.poke(false.B)
      }

      def acceptWrite(
        expectedPhysical: BigInt,
        expectedMask: BigInt,
        expectedData: BigInt
      ): Unit = {
        dut.io.physicalRequest.valid.expect(true.B)
        dut.io.physicalRequest.bits.physicalAddress
          .expect(expectedPhysical.U)
        dut.io.physicalRequest.bits.mask.expect(expectedMask.U)
        dut.io.physicalRequest.bits.data.expect(expectedData.U)
        dut.clock.step()

        dut.io.physicalResponse.bits.errorCode.poke(ErrorCode.none.U)
        dut.io.physicalResponse.valid.poke(true.B)
        dut.io.physicalResponse.ready.expect(true.B)
        dut.clock.step()
        dut.io.physicalResponse.valid.poke(false.B)
      }

      // Only byte7 of the first beat is updated.
      translate(expectedVirtual = 0x0f, physical = 0x2f)
      acceptWrite(
        expectedPhysical = 0x28,
        expectedMask = 0x80,
        expectedData = BigInt(0x11) << 56
      )

      // Remaining source bytes 22,33,44 occupy bytes0..2 of the next page.
      translate(expectedVirtual = 0x10, physical = 0x40)
      acceptWrite(
        expectedPhysical = 0x40,
        expectedMask = 0x07,
        expectedData = BigInt("443322", 16)
      )

      dut.io.response.valid.expect(true.B)
      dut.io.response.bits.errorCode.expect(ErrorCode.none.U)
      dut.io.response.ready.poke(true.B)
      dut.clock.step()
    }
  }

  "translation errors terminate the row without issuing a physical request" in {
    simulate(new VmReadDma(p)) { dut =>
      dut.reset.poke(true.B)
      dut.io.request.valid.poke(false.B)
      dut.io.response.ready.poke(false.B)
      dut.io.translationRequest.ready.poke(true.B)
      dut.io.translationResponse.valid.poke(false.B)
      dut.io.physicalRequest.ready.poke(true.B)
      dut.io.physicalResponse.valid.poke(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)

      dut.io.request.bits.virtualAddress.poke("h1000".U)
      dut.io.request.bits.bytes.poke(8.U)
      dut.io.request.bits.translationStatus.poke(0.U)
      dut.io.request.valid.poke(true.B)
      dut.clock.step()
      dut.io.request.valid.poke(false.B)

      dut.io.translationRequest.valid.expect(true.B)
      dut.clock.step()
      dut.io.translationResponse.bits.physicalAddress.poke(0.U)
      dut.io.translationResponse.bits.errorCode.poke(ErrorCode.tlb.U)
      dut.io.translationResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.translationResponse.valid.poke(false.B)

      dut.io.physicalRequest.valid.expect(false.B)
      dut.io.response.valid.expect(true.B)
      dut.io.response.bits.errorCode.expect(ErrorCode.tlb.U)
    }
  }
}
