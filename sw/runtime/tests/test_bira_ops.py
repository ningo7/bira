from pathlib import Path
import subprocess
import tempfile
import unittest


BIRA_ROOT = Path(__file__).resolve().parents[1]


class BiRaOperatorsTest(unittest.TestCase):
    def test_dense_conv_automatically_tiles_and_rotates_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'test_bira_ops'
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
                    str(BIRA_ROOT / 'src/bira_ops.c'),
                    str(BIRA_ROOT / 'tests/c/test_bira_ops.c'),
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
                [str(executable)],
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
            self.assertIn('convolution used', run_result.stdout)
            self.assertIn('EXEC commands', run_result.stdout)


if __name__ == '__main__':
    unittest.main()
