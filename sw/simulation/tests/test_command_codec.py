from pathlib import Path
import tempfile
import unittest

from simulation.command_codec import dump_commands, read_commands


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "build/bfsrcnn/fixture/commands.bin"


class CommandCodecTest(unittest.TestCase):
    def test_generated_trace_can_be_human_decoded(self):
        if not FIXTURE.is_file():
            self.skipTest("run make bfsrcnn-fixture first")
        base, commands = read_commands(FIXTURE)
        self.assertEqual(base, 0x10000)
        self.assertEqual(len(commands), 228)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "commands.txt"
            dump_commands(FIXTURE, output)
            text = output.read_text()
            self.assertIn("CFG_SHAPE", text)
            self.assertIn("EXEC_CONV", text)
            self.assertIn("STORE_2D", text)


if __name__ == "__main__":
    unittest.main()
