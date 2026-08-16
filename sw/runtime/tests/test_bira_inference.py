from pathlib import Path
import subprocess
import tempfile
import unittest


BIRA_ROOT = Path(__file__).resolve().parents[1]


class BiRaInferenceTest(unittest.TestCase):
    def test_multilayer_residency_and_standalone_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "test_bira_inference"
            compile_result = subprocess.run(
                [
                    "gcc",
                    "-std=c99",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{BIRA_ROOT / 'include'}",
                    str(BIRA_ROOT / "src/bira_runtime.c"),
                    str(BIRA_ROOT / "src/bira_trace.c"),
                    str(BIRA_ROOT / "src/bira_inference.c"),
                    str(BIRA_ROOT / "src/bira_planner.c"),
                    str(
                        BIRA_ROOT
                        / "tests/c/test_bira_inference.c"
                    ),
                    "-o",
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
            self.assertIn("ran two layers together", run_result.stdout)


if __name__ == "__main__":
    unittest.main()
