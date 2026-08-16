from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


BIRA_ROOT = Path(__file__).resolve().parents[1]


class BiRaRuntimeTest(unittest.TestCase):
    def test_trace_backend_records_portable_command_and_memory_images(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / 'test_bira_runtime'
            commands = directory / 'commands.bin'
            memory = directory / 'memory.bin'
            manifest = directory / 'manifest.txt'

            compile_result = subprocess.run(
                [
                    'gcc',
                    '-std=c99',
                    '-O2',
                    '-Wall',
                    '-Wextra',
                    '-Werror',
                    f'-I{BIRA_ROOT / "include"}',
                    str(BIRA_ROOT / 'src/bira_runtime.c'),
                    str(BIRA_ROOT / 'src/bira_trace.c'),
                    str(BIRA_ROOT / 'tests/c/test_bira_runtime.c'),
                    '-o',
                    str(executable),
                ],
                cwd=BIRA_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                compile_result.stdout + compile_result.stderr,
            )
            run_result = subprocess.run(
                [
                    str(executable),
                    str(commands),
                    str(memory),
                    str(manifest),
                ],
                cwd=BIRA_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                run_result.returncode,
                0,
                run_result.stdout + run_result.stderr,
            )
            self.assertIn('recorded 14 commands', run_result.stdout)

            command_data = commands.read_bytes()
            self.assertEqual(command_data[:8], b'BIRA_CMD')
            version, command_count = struct.unpack_from('<II', command_data, 8)
            memory_base, = struct.unpack_from('<Q', command_data, 16)
            self.assertEqual(version, 1)
            self.assertEqual(command_count, 14)
            self.assertEqual(memory_base, 0x20000)
            self.assertEqual(len(command_data), 24 + 24 * command_count)

            first_load = 8
            offset = 24 + 24 * first_load
            funct, flags = struct.unpack_from('<BB', command_data, offset)
            address, descriptor = struct.unpack_from(
                '<QQ', command_data, offset + 8
            )
            self.assertEqual(funct, 0x44)
            self.assertEqual(flags, 0)
            self.assertGreaterEqual(address, memory_base)
            self.assertEqual((descriptor >> 21) & 0x3FFF, 4)
            self.assertEqual((descriptor >> 35) & 0x7F, 16)

            memory_data = memory.read_bytes()
            self.assertEqual(
                memory_data[address - memory_base:address - memory_base + 4],
                bytes([1, 2, 3, 4]),
            )
            manifest_text = manifest.read_text()
            self.assertIn('format=bira-trace-v1', manifest_text)
            self.assertIn('mapping_count=4', manifest_text)


if __name__ == '__main__':
    unittest.main()
