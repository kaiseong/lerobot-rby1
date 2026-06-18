from __future__ import annotations

import copy
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd

from .cli import resolve_workers
from .compat import (
    DatasetRef,
    data_file_path,
    episodes_to_pandas,
    import_lerobot_symbols,
    load_dataset,
    probe_dataset,
    video_file_path,
    video_keys,
)
from .no_reencode import guard_no_reencode
from .reporting import emit_report


@dataclass(frozen=True)
class TrimConfig:
    repo_id: str
    new_repo_id: str
    root: Path | None
    new_root: Path
    keep_start_seconds: float = 0.0
    keep_end_seconds: float = 0.0
    state_key: str = "observation.state"
    state_epsilon: float = 5e-4
    workers: str | int | None = "auto"
    video_copy_mode: str = "copy"
    dry_run: bool = False
    validate_only: bool = False
    push_to_hub: bool = False
    report_path: Path | None = None


@dataclass(frozen=True)
class EpisodeRange:
    old_episode_index: int
    new_episode_index: int
    start_frame: int
    end_frame: int
    original_length: int
    data_chunk: int
    data_file: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def _load_state_array(ep_df: pd.DataFrame, state_key: str) -> np.ndarray:
    values = ep_df[state_key].to_numpy()
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float32)
    first = values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack(values).astype(np.float64)
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float64)
    return arr.astype(np.float64)


def _stationary_prefix(mask: np.ndarray) -> int:
    for idx, value in enumerate(mask):
        if not bool(value):
            return idx
    return len(mask)


def _compute_one_range(
    meta,
    ep_row: pd.Series,
    new_episode_index: int,
    state_key: str,
    state_epsilon: float,
    keep_start_frames: int,
    keep_end_frames: int,
) -> EpisodeRange:
    old_ep = int(ep_row["episode_index"])
    df = pd.read_parquet(data_file_path(meta, ep_row))
    ep_df = df[df["episode_index"] == old_ep].reset_index(drop=True)
    if state_key not in ep_df.columns:
        raise ValueError(f"State feature '{state_key}' not found in data for episode {old_ep}")

    state_array = _load_state_array(ep_df, state_key)
    if len(state_array) == 0:
        raise ValueError(f"Episode {old_ep} has no frames")

    start_diffs = np.max(np.abs(state_array - state_array[0]), axis=1)
    end_diffs = np.max(np.abs(state_array - state_array[-1]), axis=1)
    start_stationary_len = _stationary_prefix(start_diffs <= state_epsilon)
    end_stationary_len = _stationary_prefix((end_diffs <= state_epsilon)[::-1])

    start_idx = max(0, start_stationary_len - keep_start_frames)
    end_idx = len(ep_df) - max(0, end_stationary_len - keep_end_frames)
    if start_idx >= end_idx:
        raise ValueError(
            f"Stationary trim would remove all frames from episode {old_ep}: "
            f"start={start_idx}, end={end_idx}, length={len(ep_df)}"
        )

    return EpisodeRange(
        old_episode_index=old_ep,
        new_episode_index=new_episode_index,
        start_frame=int(start_idx),
        end_frame=int(end_idx),
        original_length=int(len(ep_df)),
        data_chunk=int(ep_row["data/chunk_index"]),
        data_file=int(ep_row["data/file_index"]),
    )


def compute_trim_ranges(config: TrimConfig, meta, episodes_df: pd.DataFrame) -> list[EpisodeRange]:
    if config.keep_start_seconds < 0 or config.keep_end_seconds < 0:
        raise ValueError("keep_start_seconds and keep_end_seconds must be non-negative")
    if config.state_epsilon <= 0:
        raise ValueError("state_epsilon must be positive")
    if config.state_key not in meta.info["features"]:
        raise ValueError(f"State feature '{config.state_key}' not found in dataset features")
    if meta.info["features"][config.state_key].get("dtype") in {"image", "video"}:
        raise ValueError(f"State feature '{config.state_key}' must be numeric")

    keep_start_frames = round(config.keep_start_seconds * meta.fps)
    keep_end_frames = round(config.keep_end_seconds * meta.fps)
    rows = [row for _, row in episodes_df.sort_values("episode_index").iterrows()]
    workers = resolve_workers(config.workers, len(rows))

    if workers == 1:
        ranges = [
            _compute_one_range(
                meta,
                row,
                idx,
                config.state_key,
                config.state_epsilon,
                keep_start_frames,
                keep_end_frames,
            )
            for idx, row in enumerate(rows)
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _compute_one_range,
                    meta,
                    row,
                    idx,
                    config.state_key,
                    config.state_epsilon,
                    keep_start_frames,
                    keep_end_frames,
                )
                for idx, row in enumerate(rows)
            ]
            ranges = [future.result() for future in futures]

    return sorted(ranges, key=lambda r: r.new_episode_index)


def _copy_video_files(meta, output_root: Path, episodes_df: pd.DataFrame, mode: str) -> list[dict[str, Any]]:
    if mode != "copy":
        raise ValueError("Only --video-copy-mode copy is supported in the no-reencode first pass")
    copied = []
    for key in video_keys(meta):
        pairs = sorted(
            {
                (int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"]))
                for _, row in episodes_df.iterrows()
            }
        )
        for chunk_idx, file_idx in pairs:
            src = video_file_path(meta, key, chunk_idx, file_idx)
            dst = output_root / meta.info["video_path"].format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append({"video_key": key, "src": src, "dst": dst})
    return copied


def _copy_stats_or_recompute_numeric(meta, out_df: pd.DataFrame) -> dict[str, Any] | None:
    stats = copy.deepcopy(getattr(meta, "stats", None))
    if stats is None:
        return None
    for key, ft in meta.info["features"].items():
        if key not in out_df.columns or ft.get("dtype") in {"image", "video"}:
            continue
        values = _load_state_array(out_df, key)
        if values.size == 0:
            continue
        stats[key] = {
            "max": np.max(values, axis=0),
            "mean": np.mean(values, axis=0),
            "min": np.min(values, axis=0),
            "std": np.std(values, axis=0),
            "count": np.array([len(values)], dtype=np.int64),
        }
    return stats


def _write_trimmed_dataset(config: TrimConfig, meta, episodes_df: pd.DataFrame, ranges: list[EpisodeRange]):
    symbols = import_lerobot_symbols()
    output_root = config.new_root
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root already exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    range_by_old = {r.old_episode_index: r for r in ranges}
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    out_frames = []
    out_episodes = []
    global_index = 0

    for _, ep_row in episodes_df.sort_values("episode_index").iterrows():
        old_ep = int(ep_row["episode_index"])
        tr = range_by_old[old_ep]
        cache_key = (tr.data_chunk, tr.data_file)
        if cache_key not in data_cache:
            data_cache[cache_key] = pd.read_parquet(data_file_path(meta, ep_row))
        src_df = data_cache[cache_key]
        ep_df = src_df[src_df["episode_index"] == old_ep].reset_index(drop=True)
        keep_df = ep_df.iloc[tr.start_frame : tr.end_frame].copy().reset_index(drop=True)
        keep_df["index"] = np.arange(global_index, global_index + tr.length)
        keep_df["episode_index"] = tr.new_episode_index
        keep_df["frame_index"] = np.arange(tr.length)
        keep_df["timestamp"] = np.arange(tr.length, dtype=np.float64) / float(meta.fps)
        out_frames.append(keep_df)

        new_ep = ep_row.to_dict()
        new_ep["episode_index"] = tr.new_episode_index
        new_ep["meta/episodes/chunk_index"] = 0
        new_ep["meta/episodes/file_index"] = 0
        new_ep["data/chunk_index"] = 0
        new_ep["data/file_index"] = 0
        new_ep["dataset_from_index"] = global_index
        new_ep["dataset_to_index"] = global_index + tr.length
        new_ep["length"] = tr.length
        for key in video_keys(meta):
            original_from = float(ep_row[f"videos/{key}/from_timestamp"])
            new_ep[f"videos/{key}/from_timestamp"] = original_from + tr.start_frame / float(meta.fps)
            new_ep[f"videos/{key}/to_timestamp"] = original_from + tr.end_frame / float(meta.fps)
        out_episodes.append(new_ep)
        global_index += tr.length

    out_df = pd.concat(out_frames, ignore_index=True)
    data_rel = meta.info["data_path"].format(chunk_index=0, file_index=0)
    data_path = output_root / data_rel
    data_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(data_path)

    copied_videos = _copy_video_files(meta, output_root, episodes_df, config.video_copy_mode)

    out_info = copy.deepcopy(meta.info)
    out_info["total_episodes"] = len(out_episodes)
    out_info["total_frames"] = len(out_df)
    out_info["total_tasks"] = len(meta.tasks)
    out_info["splits"] = {"train": f"0:{len(out_episodes)}"}
    symbols["write_info"](out_info, output_root)
    symbols["write_tasks"](meta.tasks.copy(), output_root)

    write_stats = symbols.get("write_stats")
    if write_stats is not None:
        stats = _copy_stats_or_recompute_numeric(meta, out_df)
        if stats is not None:
            write_stats(stats, output_root)

    episodes_ds = datasets.Dataset.from_pandas(pd.DataFrame(out_episodes), preserve_index=False)
    write_episodes = symbols.get("write_episodes")
    if write_episodes is None:
        raise RuntimeError("LeRobot write_episodes helper is required for this dataset format")
    write_episodes(episodes_ds, output_root)

    return {
        "output_root": output_root,
        "total_frames": len(out_df),
        "total_episodes": len(out_episodes),
        "copied_videos": copied_videos,
    }


def validate_trim_output(root: Path, repo_id: str) -> None:
    symbols = import_lerobot_symbols()
    metadata_cls = symbols["LeRobotDatasetMetadata"]
    metadata_cls(repo_id, root=root)


def trim_stationary_dataset(config: TrimConfig) -> dict[str, Any]:
    ref = DatasetRef(config.repo_id, config.root)
    meta, compat = probe_dataset(ref)
    report: dict[str, Any] = {
        "operation": "trim_stationary_dataset",
        "repo_id": config.repo_id,
        "new_repo_id": config.new_repo_id,
        "root": compat.root,
        "new_root": config.new_root,
        "dry_run": config.dry_run,
        "validate_only": config.validate_only,
        "push_to_hub": config.push_to_hub,
        "compatibility": compat.__dict__,
        "no_reencode": True,
    }
    if not compat.ok or meta is None:
        report["status"] = "unsupported"
        emit_report(report, config.report_path)
        raise RuntimeError("; ".join(compat.errors))

    episodes_df = episodes_to_pandas(meta.episodes)
    ranges = compute_trim_ranges(config, meta, episodes_df)
    report["workers"] = resolve_workers(config.workers, len(ranges))
    report["ranges"] = [r.__dict__ for r in ranges]
    report["planned_total_frames"] = sum(r.length for r in ranges)
    report["planned_total_episodes"] = len(ranges)

    if config.dry_run or config.validate_only:
        if config.validate_only and config.new_root.exists() and any(config.new_root.iterdir()):
            validate_trim_output(config.new_root, config.new_repo_id)
            report["existing_output_validated"] = True
        else:
            report["existing_output_validated"] = False
        report["status"] = "validated"
        return emit_report(report, config.report_path)

    if video_keys(meta):
        meta = load_dataset(DatasetRef(config.repo_id, config.root), download_videos=True).meta
        episodes_df = episodes_to_pandas(meta.episodes)

    with guard_no_reencode():
        result = _write_trimmed_dataset(config, meta, episodes_df, ranges)

    validate_trim_output(config.new_root, config.new_repo_id)
    report.update(result)
    report["status"] = "written"

    if config.push_to_hub:
        symbols = import_lerobot_symbols()
        dataset_cls = symbols["LeRobotDataset"]
        dataset_cls(config.new_repo_id, root=config.new_root).push_to_hub()
        report["pushed_to_hub"] = True
    else:
        report["pushed_to_hub"] = False

    return emit_report(report, config.report_path)
