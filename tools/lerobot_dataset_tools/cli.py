from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def ensure_lerobot_import_path() -> None:
    """Allow running tools from a fresh LeRobot clone without installing first."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    src = repo_root / "src"
    for path in (repo_root, src):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def add_bool_arg(parser: argparse.ArgumentParser, *names: str, default: bool = False, help: str = ""):
    parser.add_argument(*names, nargs="?", const=True, default=default, type=parse_bool, help=help)


def resolve_workers(value: str | int | None, total_items: int | None = None) -> int:
    if value is None or str(value).lower() == "auto":
        cpu = os.cpu_count() or 1
        limit = total_items if total_items and total_items > 0 else cpu
        return max(1, min(cpu, limit))
    workers = int(value)
    if workers < 1:
        raise ValueError("worker count must be >= 1")
    return workers


def common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--new-repo-id", "--new_repo_id", required=True, help="Output dataset repo id")
    parser.add_argument("--root", default=None, help="Input dataset root or common source root")
    parser.add_argument("--new-root", "--new_root", required=True, help="Output dataset root")
    add_bool_arg(parser, "--push-to-hub", "--push_to_hub", default=False, help="Push output dataset to Hub")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Analyze/report without writing")
    parser.add_argument("--validate-only", "--validate_only", action="store_true", help="Validate without writing")
    parser.add_argument("--report-path", "--report_path", default=None, help="Optional JSON report path")

