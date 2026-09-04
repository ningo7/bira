#!/usr/bin/env python3
"""Install or remove the BIRA SBT project in a local Chipyard checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


OPTION_BEGIN = "    // BEGIN BIRA INTEGRATION: optional-module"
OPTION_END = "    // END BIRA INTEGRATION: optional-module"
PROJECT_BEGIN = "// BEGIN BIRA INTEGRATION: project"
PROJECT_END = "// END BIRA INTEGRATION: project"

OPTION_BLOCK = "\n".join(
    (
        OPTION_BEGIN,
        '    "bira" -> bira,',
        OPTION_END,
    )
)

PROJECT_BLOCK = "\n".join(
    (
        PROJECT_BEGIN,
        'lazy val bira = freshProject("bira", file("generators/bira/hw"))',
        "  .dependsOn(rocketchip)",
        "  .settings(libraryDependencies ++= rocketLibDeps.value)",
        "  .settings(commonSettings)",
        "  .settings(scalaTestSettings)",
        "  .settings(",
        '    Test / scalaSource := baseDirectory.value / "test" / "scala",',
        "    Test / resourceDirectory :=",
        '      baseDirectory.value / "test" / "resources"',
        "  )",
        PROJECT_END,
    )
)

SUPPORTED_VERSION_PREFIXES = ("1.13",)


class IntegrationError(RuntimeError):
    pass


def bira_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_chipyard_root() -> Path:
    return bira_root().parent.parent


def chipyard_description(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--always"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def validate_layout(root: Path, force_untested: bool) -> Path:
    build_file = root / "build.sbt"
    expected_bira = root / "generators" / "bira"
    required = (
        bira_root() / "hw" / "src" / "main" / "scala" / "bira",
        bira_root()
        / "chipyard"
        / "src"
        / "main"
        / "scala"
        / "chipyard"
        / "AccelConfigs.scala",
        bira_root() / "sw" / "runtime" / "include" / "bira_rocc.h",
    )
    if not build_file.is_file():
        raise IntegrationError(f"Chipyard build.sbt not found: {build_file}")
    if expected_bira.resolve() != bira_root():
        raise IntegrationError(
            "BIRA must be cloned at <chipyard>/generators/bira; "
            f"found {bira_root()}"
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise IntegrationError(
            "BIRA checkout is incomplete; missing: " + ", ".join(missing)
        )

    description = chipyard_description(root)
    if description is not None and not description.startswith(
        SUPPORTED_VERSION_PREFIXES
    ):
        message = (
            f"untested Chipyard version {description!r}; supported prefixes: "
            + ", ".join(SUPPORTED_VERSION_PREFIXES)
        )
        if not force_untested:
            raise IntegrationError(message + "; pass --force-untested to continue")
        print(f"warning: {message}", file=sys.stderr)
    elif description is None:
        print(
            "warning: cannot determine Chipyard version; validating build.sbt "
            "anchors only",
            file=sys.stderr,
        )
    return build_file


def replace_legacy_discovery(text: str) -> str:
    legacy = (
        '    file(s"generators/$dir/.git").exists ||\n'
        '    (dir == "bira" && '
        'file("generators/bira/hw/src/main/scala").exists)'
    )
    return text.replace(
        legacy,
        '    file(s"generators/$dir/.git").exists',
    )


def insert_optional_block(text: str) -> str:
    if OPTION_BEGIN in text or OPTION_END in text:
        if text.count(OPTION_BEGIN) != 1 or text.count(OPTION_END) != 1:
            raise IntegrationError("malformed BIRA optional-module markers")
        pattern = re.compile(
            rf"(?m)^{re.escape(OPTION_BEGIN)}\n.*?^{re.escape(OPTION_END)}",
            re.DOTALL,
        )
        return pattern.sub(OPTION_BLOCK, text, count=1)

    existing = re.compile(r'(?m)^    "bira" -> bira,\n?')
    if existing.search(text):
        return existing.sub(OPTION_BLOCK + "\n", text, count=1)

    anchor = re.compile(r'(?m)^(    "gemmini" -> gemmini,\n)')
    if not anchor.search(text):
        raise IntegrationError(
            'cannot find optionalModules anchor: "gemmini" -> gemmini'
        )
    return anchor.sub(r"\1" + OPTION_BLOCK + "\n", text, count=1)


def insert_project_block(text: str) -> str:
    if PROJECT_BEGIN in text or PROJECT_END in text:
        if text.count(PROJECT_BEGIN) != 1 or text.count(PROJECT_END) != 1:
            raise IntegrationError("malformed BIRA project markers")
        pattern = re.compile(
            rf"(?m)^{re.escape(PROJECT_BEGIN)}\n.*?^{re.escape(PROJECT_END)}",
            re.DOTALL,
        )
        return pattern.sub(PROJECT_BLOCK, text, count=1)

    legacy = re.compile(
        r"(?ms)^lazy val bira = .*?(?=^lazy val [A-Za-z0-9_]+ =)"
    )
    if legacy.search(text):
        return legacy.sub(PROJECT_BLOCK + "\n\n", text, count=1)

    text = re.sub(
        r"\n{3,}(?=lazy val nvdla =)",
        "\n\n",
        text,
    )
    anchor = re.compile(r"(?m)^lazy val nvdla =")
    if not anchor.search(text):
        raise IntegrationError("cannot find project anchor: lazy val nvdla")
    return anchor.sub(PROJECT_BLOCK + "\n\nlazy val nvdla =", text, count=1)


def remove_marked_block(text: str, begin: str, end: str) -> str:
    if begin not in text and end not in text:
        return text
    if text.count(begin) != 1 or text.count(end) != 1:
        raise IntegrationError(f"malformed integration markers for {begin}")
    pattern = re.compile(
        rf"(?m)^{re.escape(begin)}\n.*?^{re.escape(end)}\n?",
        re.DOTALL,
    )
    updated, count = pattern.subn("", text, count=1)
    if count != 1:
        raise IntegrationError(f"cannot remove integration block {begin}")
    return updated


def atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.bira.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def install(build_file: Path) -> None:
    original = build_file.read_text(encoding="utf-8")
    updated = replace_legacy_discovery(original)
    updated = insert_optional_block(updated)
    updated = insert_project_block(updated)
    if updated == original:
        print(f"BIRA integration already installed: {build_file}")
        return
    atomic_write(build_file, updated)
    print(f"installed BIRA integration: {build_file}")


def uninstall(build_file: Path) -> None:
    original = build_file.read_text(encoding="utf-8")
    updated = remove_marked_block(original, OPTION_BEGIN, OPTION_END)
    updated = remove_marked_block(updated, PROJECT_BEGIN, PROJECT_END)
    updated = re.sub(
        r"\n{3,}(?=lazy val nvdla =)",
        "\n\n",
        updated,
    )
    if updated == original:
        print(f"BIRA integration is not installed by this tool: {build_file}")
        return
    if '"bira" -> bira' in updated or re.search(
        r"(?m)^lazy val bira =", updated
    ):
        raise IntegrationError(
            "unmarked BIRA build definitions remain; refusing partial uninstall"
        )
    atomic_write(build_file, updated)
    print(f"removed BIRA integration: {build_file}")


def check(build_file: Path) -> None:
    text = build_file.read_text(encoding="utf-8")
    required = (OPTION_BEGIN, OPTION_END, PROJECT_BEGIN, PROJECT_END)
    missing = [marker for marker in required if text.count(marker) != 1]
    if missing:
        raise IntegrationError(
            "BIRA integration is absent or malformed: " + ", ".join(missing)
        )
    if OPTION_BLOCK not in text or PROJECT_BLOCK not in text:
        raise IntegrationError(
            "BIRA integration markers exist but the installed definition is "
            "stale; run install-chipyard.sh to update it"
        )
    if not (bira_root() / ".git").exists():
        print(
            "warning: generators/bira is not currently a standalone Git "
            "checkout",
            file=sys.stderr,
        )
    print(f"BIRA integration check passed: {build_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage BIRA integration in a local Chipyard checkout."
    )
    parser.add_argument("action", choices=("install", "uninstall", "check"))
    parser.add_argument(
        "--chipyard-root",
        type=Path,
        default=default_chipyard_root(),
        help="Chipyard checkout root (default: inferred from generators/bira)",
    )
    parser.add_argument(
        "--force-untested",
        action="store_true",
        help="allow a Chipyard version outside the tested compatibility list",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        root = arguments.chipyard_root.expanduser().resolve()
        allow_untested = (
            arguments.force_untested or arguments.action != "install"
        )
        build_file = validate_layout(root, allow_untested)
        if arguments.action == "install":
            install(build_file)
        elif arguments.action == "uninstall":
            uninstall(build_file)
        else:
            check(build_file)
    except IntegrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
