#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from lerobot_dataset_tools.cli import add_bool_arg, ensure_lerobot_import_path

ensure_lerobot_import_path()

from lerobot_dataset_tools.merge import MergeConfig, merge_datasets, parse_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone no-reencode merge for LeRobot datasets")
    parser.add_argument("--source", action="append", required=True, help="Source as REPO_ID or REPO_ID=ROOT")
    parser.add_argument("--repo-id", "--repo_id", dest="new_repo_id", default=None, help="Output dataset repo id")
    parser.add_argument("--new-repo-id", "--new_repo_id", dest="new_repo_id", help="Output dataset repo id")
    parser.add_argument("--root", default=None, help="Optional common root for sources without '=ROOT'")
    parser.add_argument("--new-root", "--new_root", required=True, help="Output dataset root")
    add_bool_arg(parser, "--push-to-hub", "--push_to_hub", default=False, help="Push output dataset to Hub")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Analyze/report without writing")
    parser.add_argument("--validate-only", "--validate_only", action="store_true", help="Validate without writing")
    parser.add_argument("--report-path", "--report_path", default=None, help="Optional JSON report path")
    parser.add_argument("--copy-workers", "--copy_workers", default="auto", help="'auto' or positive integer")
    parser.add_argument("--remux-policy", "--remux_policy", default="never", choices=["never"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.new_repo_id:
        raise SystemExit("--new-repo-id is required")
    common_root = Path(args.root).expanduser() if args.root else None
    sources = []
    for src in args.source:
        ref = parse_source(src)
        if ref.root is None and common_root is not None:
            ref = type(ref)(repo_id=ref.repo_id, root=common_root / ref.repo_id)
        sources.append(ref)
    config = MergeConfig(
        sources=sources,
        new_repo_id=args.new_repo_id,
        new_root=Path(args.new_root).expanduser(),
        copy_workers=args.copy_workers,
        remux_policy=args.remux_policy,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        push_to_hub=args.push_to_hub,
        report_path=Path(args.report_path).expanduser() if args.report_path else None,
    )
    merge_datasets(config)


if __name__ == "__main__":
    main()
