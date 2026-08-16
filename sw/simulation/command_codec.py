"""Read and human-decode the stable BIRA Trace command format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


MAGIC = b"BIRA_CMD"
NAMES = {
    0x40: "CFG_SHAPE",
    0x41: "CFG_ADDR",
    0x42: "CFG_MODE",
    0x43: "CFG_COMMIT",
    0x44: "LOAD_2D",
    0x45: "EXEC_CONV",
    0x46: "STORE_2D",
    0x47: "FENCE",
    0x48: "STATUS",
    0x49: "TLB_FLUSH",
}


@dataclass(frozen=True)
class Command:
    funct: int
    returns_value: bool
    rs1: int
    rs2: int


def read_commands(path) -> tuple[int, list[Command]]:
    data = Path(path).read_bytes()
    if len(data) < 24 or data[:8] != MAGIC:
        raise ValueError(f"{path} is not a BIRA command file")
    version, count, memory_base = struct.unpack_from("<IIQ", data, 8)
    if version != 1 or len(data) != 24 + count * 24:
        raise ValueError(f"unsupported or truncated command file {path}")
    commands = []
    offset = 24
    for _ in range(count):
        funct, returns = struct.unpack_from("<BB", data, offset)
        rs1, rs2 = struct.unpack_from("<QQ", data, offset + 8)
        commands.append(Command(funct, bool(returns), rs1, rs2))
        offset += 24
    return memory_base, commands


def dump_commands(input_path, output_path) -> Path:
    memory_base, commands = read_commands(input_path)
    lines = [
        "format=bira-command-dump-v1",
        f"memory_base=0x{memory_base:x}",
        f"command_count={len(commands)}",
        "",
    ]
    for index, command in enumerate(commands):
        name = NAMES.get(command.funct, f"FUNCT_{command.funct}")
        lines.append(
            f"{index:03d} {name:<10} "
            f"xd={int(command.returns_value)} "
            f"rs1=0x{command.rs1:016x} "
            f"rs2=0x{command.rs2:016x}"
        )
    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path
