"""Export BFSRCNN's quantized parameters; inference code is handwritten C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.bfsrcnn.deployment.data import write_bfsrcnn_data
from models.bfsrcnn.deployment.hardware_params import (
    render_hardware_header,
)
from models.bfsrcnn.quantization.checkpoint import (
    load_pixel_integer_checkpoint,
)


def export_bfsrcnn_data(
    checkpoint,
    output_dir,
    *,
    input_height: int = 32,
    input_width: int = 32,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
        "bfsrcnn_bira_network.c",
        "bfsrcnn_bira_network.h",
        "bfsrcnn_bira_manifest.json",
        "bfsrcnn_network.json",
        "bfsrcnn_rocc_inference.c",
    ):
        legacy_path = output_dir / legacy_name
        if legacy_path.is_file():
            legacy_path.unlink()
    hardware_header = output_dir / "bfsrcnn_hardware.h"
    model, metadata = load_pixel_integer_checkpoint(checkpoint, "cpu")
    hardware_text, hardware = render_hardware_header(
        model,
        metadata,
        checkpoint,
        hardware_header,
        symbol_prefix="bfsrcnn_hw",
    )
    hardware_header.write_text(hardware_text, encoding="utf-8")
    data = write_bfsrcnn_data(
        model, output_dir, input_height, input_width
    )
    manifest = {
        "format": "bira-model-data-v2",
        "source_checkpoint": str(Path(checkpoint)),
        "hardware": hardware,
        "data": data,
        "inference_source": (
            "applications/bfsrcnn/bfsrcnn_inference.c"
        ),
    }
    (output_dir / "deployment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "export a quantized BFSRCNN checkpoint as native BIRA C data"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    manifest = export_bfsrcnn_data(
        args.checkpoint,
        args.output_dir,
        input_height=args.height,
        input_width=args.width,
    )
    print(
        f"exported {len(manifest['data']['layers'])} layers of "
        f"BFSRCNN data to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
