#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from lerobot_dataset_tools.cli import common_output_args, ensure_lerobot_import_path

ensure_lerobot_import_path()

from lerobot_dataset_tools.trim import TrimConfig, trim_stationary_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone logical stationary trim for LeRobot datasets")
    parser.add_argument("--repo-id", "--repo_id", required=True, help="Input dataset repo id")
    common_output_args(parser)
    parser.add_argument("--keep-start-seconds", "--keep_start_seconds", type=float, default=0.0)
    parser.add_argument("--keep-end-seconds", "--keep_end_seconds", type=float, default=0.0)
    parser.add_argument("--state-key", "--state_key", default="observation.state")
    parser.add_argument("--state-epsilon", "--state_epsilon", type=float, default=5e-4)
    parser.add_argument("--workers", default="auto", help="'auto' or positive integer")
    parser.add_argument("--video-copy-mode", "--video_copy_mode", default="copy", choices=["copy"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrimConfig(
        repo_id=args.repo_id,
        new_repo_id=args.new_repo_id,
        root=Path(args.root).expanduser() if args.root else None,
        new_root=Path(args.new_root).expanduser(),
        keep_start_seconds=args.keep_start_seconds,
        keep_end_seconds=args.keep_end_seconds,
        state_key=args.state_key,
        state_epsilon=args.state_epsilon,
        workers=args.workers,
        video_copy_mode=args.video_copy_mode,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        push_to_hub=args.push_to_hub,
        report_path=Path(args.report_path).expanduser() if args.report_path else None,
    )
    trim_stationary_dataset(config)


if __name__ == "__main__":
    main()

