from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cli import resolve_workers
from .compat import DatasetRef, feature_map, probe_dataset, video_keys
from .no_reencode import guard_no_reencode
from .reporting import emit_report


@dataclass(frozen=True)
class MergeConfig:
    sources: list[DatasetRef]
    new_repo_id: str
    new_root: Path
    copy_workers: str | int | None = "auto"
    remux_policy: str = "never"
    dry_run: bool = False
    validate_only: bool = False
    push_to_hub: bool = False
    report_path: Path | None = None


def parse_source(text: str) -> DatasetRef:
    if "=" in text:
        repo_id, root = text.split("=", 1)
        return DatasetRef(repo_id=repo_id, root=Path(root).expanduser())
    return DatasetRef(repo_id=text, root=None)


def _probe_source(ref: DatasetRef):
    meta, compat = probe_dataset(ref)
    if not compat.ok or meta is None:
        raise RuntimeError(f"{ref.repo_id}: {'; '.join(compat.errors)}")
    return meta, compat


def validate_merge_sources(metas: list[Any]) -> list[str]:
    if len(metas) < 2:
        raise ValueError("At least two source datasets are required")
    errors: list[str] = []
    base = metas[0]
    base_features = feature_map(base)
    base_video_keys = video_keys(base)
    for meta in metas[1:]:
        if meta.fps != base.fps:
            errors.append(f"fps mismatch: {meta.repo_id} has {meta.fps}, expected {base.fps}")
        if meta.robot_type != base.robot_type:
            errors.append(
                f"robot_type mismatch: {meta.repo_id} has {meta.robot_type}, expected {base.robot_type}"
            )
        if feature_map(meta) != base_features:
            errors.append(f"feature schema mismatch: {meta.repo_id}")
        if video_keys(meta) != base_video_keys:
            errors.append(f"video key mismatch: {meta.repo_id}")
    if errors:
        raise ValueError("; ".join(errors))
    return []


def validate_merge_output(root: Path, repo_id: str, expected_frames: int, expected_episodes: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    if int(dataset.meta.total_frames) != expected_frames:
        raise ValueError(
            f"Existing output total_frames={dataset.meta.total_frames}, expected {expected_frames}"
        )
    if int(dataset.meta.total_episodes) != expected_episodes:
        raise ValueError(
            f"Existing output total_episodes={dataset.meta.total_episodes}, expected {expected_episodes}"
        )


def merge_datasets(config: MergeConfig) -> dict[str, Any]:
    if config.remux_policy != "never":
        raise ValueError("Only --remux-policy never is supported in the no-reencode first pass")
    workers = resolve_workers(config.copy_workers, len(config.sources))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        probed = list(executor.map(_probe_source, config.sources))
    metas = [item[0] for item in probed]
    compat = [item[1].__dict__ for item in probed]
    validate_merge_sources(metas)

    total_frames = sum(int(meta.total_frames) for meta in metas)
    total_episodes = sum(int(meta.total_episodes) for meta in metas)
    report: dict[str, Any] = {
        "operation": "merge_datasets",
        "sources": [{"repo_id": src.repo_id, "root": src.root} for src in config.sources],
        "new_repo_id": config.new_repo_id,
        "new_root": config.new_root,
        "dry_run": config.dry_run,
        "validate_only": config.validate_only,
        "push_to_hub": config.push_to_hub,
        "copy_workers": workers,
        "remux_policy": config.remux_policy,
        "compatibility": compat,
        "planned_total_frames": total_frames,
        "planned_total_episodes": total_episodes,
        "no_reencode": True,
    }

    if config.dry_run or config.validate_only:
        if config.validate_only and config.new_root.exists() and any(config.new_root.iterdir()):
            validate_merge_output(config.new_root, config.new_repo_id, total_frames, total_episodes)
            report["existing_output_validated"] = True
        else:
            report["existing_output_validated"] = False
        report["status"] = "validated"
        return emit_report(report, config.report_path)

    if config.new_root.exists() and any(config.new_root.iterdir()):
        raise FileExistsError(f"Output root already exists and is not empty: {config.new_root}")

    repo_ids = [src.repo_id for src in config.sources]
    roots = [meta.root for meta in metas]

    with guard_no_reencode():
        from lerobot.datasets.aggregate import aggregate_datasets

        aggregate_datasets(
            repo_ids=repo_ids,
            aggr_repo_id=config.new_repo_id,
            roots=roots,
            aggr_root=config.new_root,
            video_files_size_in_mb=0,
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LeRobotDataset(config.new_repo_id, root=config.new_root)
    report["status"] = "written"

    if config.push_to_hub:
        LeRobotDataset(config.new_repo_id, root=config.new_root).push_to_hub()
        report["pushed_to_hub"] = True
    else:
        report["pushed_to_hub"] = False

    return emit_report(report, config.report_path)
