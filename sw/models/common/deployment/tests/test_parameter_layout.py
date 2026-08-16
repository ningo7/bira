import unittest

from models.common.deployment.parameter_layout import (
    BinaryRecord,
    MultiBitRecord,
    binary_correction,
    pack_parameter_rows,
)


class ParameterLayoutTest(unittest.TestCase):
    def test_records_are_padded_to_complete_native_lane_blocks(self):
        packed = pack_parameter_rows(
            [MultiBitRecord(bias=-3, qmin=-128, qmax=127)]
        )
        self.assertEqual(len(packed), 8 * 64)
        self.assertEqual(
            int.from_bytes(packed[:4], "little", signed=True), -3
        )

    def test_binary_record_and_correction_use_hardware_layout(self):
        packed = pack_parameter_rows(
            [BinaryRecord(threshold=7, output_sign_threshold=-2)]
        )
        correction = binary_correction(4, 4, 3, 3, 1, 1, 16)
        self.assertEqual(len(packed), 8 * 64)
        self.assertEqual(len(correction), 64)


if __name__ == "__main__":
    unittest.main()
