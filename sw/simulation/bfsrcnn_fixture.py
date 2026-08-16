"""Generate a hardware fixture from programmer-written C inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from models.bfsrcnn.deployment.export import export_bfsrcnn_data
from models.bfsrcnn.deployment.input_data import make_integer_input_patch
from simulation.command_codec import dump_commands


SW_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = SW_ROOT / "runtime"
MODEL_ROOT = SW_ROOT / "models/bfsrcnn"
SIM_ROOT = SW_ROOT / "simulation"
APP_ROOT = SW_ROOT / "applications/bfsrcnn"
DEFAULT_CHECKPOINT = (
    MODEL_ROOT / "tests/fixtures/bfsrcnn_integer_v4.pth"
)
DEFAULT_INPUT_IMAGE = (
    MODEL_ROOT / "datasets/Set14/baboon.png"
)


def _run(command):
    result = subprocess.run(
        [str(item) for item in command],
        cwd=SW_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed: {' '.join(map(str, command))}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


def generate_fixture(
    checkpoint,
    generated_dir,
    output_dir,
    *,
    input_image=DEFAULT_INPUT_IMAGE,
    input_scale=4,
    tile_size=24,
    tile_pad=4,
    tile_row=1,
    tile_column=1,
):
    input_patch, input_metadata = make_integer_input_patch(
        input_image,
        scale=input_scale,
        tile_size=tile_size,
        tile_pad=tile_pad,
        tile_row=tile_row,
        tile_column=tile_column,
    )
    height, width = input_patch.shape
    input_data = input_patch.tobytes()
    generated_dir = Path(generated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment = export_bfsrcnn_data(
        checkpoint,
        generated_dir,
        input_height=height,
        input_width=width,
    )
    input_path = output_dir / "input.bin"
    input_path.write_bytes(input_data)

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        trace_exe = temporary / "bfsrcnn_trace"
        layer_exe = temporary / "bfsrcnn_layer_api_test"
        output_exe = temporary / "bfsrcnn_reference_output"
        stages_exe = temporary / "bfsrcnn_reference_stages"
        flags = [
            "gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror"
        ]
        _run([
            *flags,
            f"-I{RUNTIME_ROOT / 'include'}",
            f"-I{generated_dir}",
            RUNTIME_ROOT / "src/bira_runtime.c",
            RUNTIME_ROOT / "src/bira_trace.c",
            RUNTIME_ROOT / "src/bira_inference.c",
            RUNTIME_ROOT / "src/bira_planner.c",
            generated_dir / "bfsrcnn_bira_data.c",
            APP_ROOT / "bfsrcnn_inference.c",
            f"-I{APP_ROOT}",
            SIM_ROOT / "bira_trace_bfsrcnn.c",
            "-o", trace_exe,
        ])
        _run([
            *flags,
            f"-I{RUNTIME_ROOT / 'include'}",
            f"-I{generated_dir}",
            f"-I{APP_ROOT}",
            RUNTIME_ROOT / "src/bira_runtime.c",
            RUNTIME_ROOT / "src/bira_trace.c",
            RUNTIME_ROOT / "src/bira_inference.c",
            RUNTIME_ROOT / "src/bira_planner.c",
            generated_dir / "bfsrcnn_bira_data.c",
            APP_ROOT / "bfsrcnn_inference.c",
            SIM_ROOT / "bfsrcnn_layer_api_test.c",
            "-o", layer_exe,
        ])
        reference_flags = [
            *flags,
            f"-I{MODEL_ROOT / 'reference'}",
            f"-I{generated_dir}",
            MODEL_ROOT / "reference/bfsrcnn_reference.c",
        ]
        _run([
            *reference_flags,
            SIM_ROOT / "bfsrcnn_reference_output.c",
            "-o", output_exe,
        ])
        _run([
            *reference_flags,
            SIM_ROOT / "bfsrcnn_reference_stages.c",
            "-o", stages_exe,
        ])
        trace = _run([
            trace_exe,
            input_path,
            output_dir / "commands.bin",
            output_dir / "memory.bin",
            output_dir / "memory_map.txt",
        ])
        layer_test = _run([layer_exe])
        _run([
            output_exe, height, width, input_path,
            output_dir / "expected.bin",
        ])
        _run([
            stages_exe, height, width, input_path,
            output_dir / "stages.bin",
        ])

    dump_commands(
        output_dir / "commands.bin", output_dir / "commands.txt"
    )
    metadata = {
        "format": "bira-verilator-fixture-v2",
        "input_height": height,
        "input_width": width,
        "output_height": deployment["data"]["output_shape"][0],
        "output_width": deployment["data"]["output_shape"][1],
        "expected_bytes": (
            deployment["data"]["output_shape"][0]
            * deployment["data"]["output_shape"][1]
        ),
        "input_kind": "real-image-patch",
        "input": input_metadata,
        "source_checkpoint": str(Path(checkpoint)),
        "source_generated_data": str(generated_dir),
        "network_flow": (
            "quantized data export -> programmer-written C inference "
            "-> Runtime planner -> Trace/RoCC"
        ),
        "standalone_layer_test": layer_test.stdout.strip(),
    }
    (output_dir / "fixture.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(trace.stdout.strip())
    print(f"generated inference fixture in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-image", default=DEFAULT_INPUT_IMAGE)
    parser.add_argument("--input-scale", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=24)
    parser.add_argument("--tile-pad", type=int, default=4)
    parser.add_argument("--tile-row", type=int, default=1)
    parser.add_argument("--tile-column", type=int, default=1)
    args = parser.parse_args()
    generate_fixture(
        args.checkpoint,
        args.generated_dir,
        args.output_dir,
        input_image=args.input_image,
        input_scale=args.input_scale,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad,
        tile_row=args.tile_row,
        tile_column=args.tile_column,
    )


if __name__ == "__main__":
    main()
