from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_INFO_FIELDS = {
    "codebase_version",
    "fps",
    "robot_type",
    "features",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "data_path",
    "video_path",
}

REQUIRED_EPISODE_FIELDS = {
    "episode_index",
    "dataset_from_index",
    "dataset_to_index",
    "data/chunk_index",
    "data/file_index",
    "length",
}

REQUIRED_DATA_FIELDS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
}


@dataclass(frozen=True)
class DatasetRef:
    repo_id: str
    root: Path | None = None


@dataclass
class CompatibilityResult:
    status: str
    warnings: list[str]
    errors: list[str]
    package_version: str | None
    codebase_version: str | None
    root: str

    @property
    def ok(self) -> bool:
        return self.status in {"supported", "supported_with_warning"}


def import_lerobot_symbols() -> dict[str, Any]:
    dataset_mod = importlib.import_module("lerobot.datasets.lerobot_dataset")
    utils_mod = importlib.import_module("lerobot.datasets.utils")
    symbols = {
        "LeRobotDataset": getattr(dataset_mod, "LeRobotDataset"),
        "LeRobotDatasetMetadata": getattr(dataset_mod, "LeRobotDatasetMetadata"),
        "CODEBASE_VERSION": getattr(dataset_mod, "CODEBASE_VERSION", None),
        "load_episodes": getattr(utils_mod, "load_episodes"),
        "write_info": getattr(utils_mod, "write_info"),
        "write_tasks": getattr(utils_mod, "write_tasks"),
        "write_stats": getattr(utils_mod, "write_stats", None),
        "write_episodes": getattr(utils_mod, "write_episodes", None),
        "DEFAULT_DATA_PATH": getattr(utils_mod, "DEFAULT_DATA_PATH"),
        "DEFAULT_EPISODES_PATH": getattr(utils_mod, "DEFAULT_EPISODES_PATH"),
        "DEFAULT_VIDEO_PATH": getattr(utils_mod, "DEFAULT_VIDEO_PATH"),
    }
    return symbols


def package_version() -> str | None:
    try:
        import lerobot

        return getattr(lerobot, "__version__", None)
    except Exception:
        return None


def load_metadata(ref: DatasetRef):
    symbols = import_lerobot_symbols()
    metadata_cls = symbols["LeRobotDatasetMetadata"]
    return metadata_cls(ref.repo_id, root=ref.root)


def episodes_to_pandas(episodes: Any) -> pd.DataFrame:
    if hasattr(episodes, "to_pandas"):
        return episodes.to_pandas()
    return pd.DataFrame(list(episodes))


def normalize_root(root: str | Path | None) -> Path | None:
    if root is None or str(root) == "":
        return None
    return Path(root).expanduser()


def feature_map(meta) -> dict[str, Any]:
    return copy.deepcopy(meta.info["features"])


def video_keys(meta) -> list[str]:
    return [key for key, ft in feature_map(meta).items() if ft.get("dtype") == "video"]


def camera_keys(meta) -> list[str]:
    return [key for key, ft in feature_map(meta).items() if ft.get("dtype") in {"image", "video"}]


def data_file_path(meta, ep_row: pd.Series) -> Path:
    rel = meta.info["data_path"].format(
        chunk_index=int(ep_row["data/chunk_index"]),
        file_index=int(ep_row["data/file_index"]),
    )
    return Path(meta.root) / rel


def video_file_path(meta, video_key: str, chunk_index: int, file_index: int) -> Path:
    video_path = meta.info.get("video_path")
    if not video_path:
        raise ValueError("Dataset has video features but meta/info.json has no video_path")
    return Path(meta.root) / video_path.format(
        video_key=video_key,
        chunk_index=int(chunk_index),
        file_index=int(file_index),
    )


def probe_dataset(ref: DatasetRef) -> tuple[Any | None, CompatibilityResult]:
    warnings: list[str] = []
    errors: list[str] = []
    meta = None
    pkg_version = package_version()
    codebase_version = None
    root = str(ref.root) if ref.root is not None else "<default>"

    try:
        import_lerobot_symbols()
    except Exception as exc:
        errors.append(f"required LeRobot imports failed: {exc}")
        return None, CompatibilityResult("unsupported", warnings, errors, pkg_version, codebase_version, root)

    try:
        meta = load_metadata(ref)
        root = str(meta.root)
        codebase_version = meta.info.get("codebase_version")
    except Exception as exc:
        errors.append(f"metadata load failed: {exc}")
        return None, CompatibilityResult("unsupported", warnings, errors, pkg_version, codebase_version, root)

    missing_info = sorted(REQUIRED_INFO_FIELDS - set(meta.info))
    if missing_info:
        errors.append(f"meta/info.json missing required fields: {missing_info}")

    try:
        episodes_df = episodes_to_pandas(meta.episodes)
    except Exception as exc:
        errors.append(f"episodes metadata load failed: {exc}")
        episodes_df = pd.DataFrame()

    missing_episode = sorted(REQUIRED_EPISODE_FIELDS - set(episodes_df.columns))
    if missing_episode:
        errors.append(f"episodes metadata missing required fields: {missing_episode}")

    if getattr(meta, "tasks", None) is None or "task_index" not in meta.tasks.columns:
        errors.append("meta/tasks.parquet missing task_index column")

    if len(episodes_df) > 0:
        try:
            first_data_path = data_file_path(meta, episodes_df.iloc[0])
            first_df = pd.read_parquet(first_data_path)
            missing_data = sorted(REQUIRED_DATA_FIELDS - set(first_df.columns))
            if missing_data:
                errors.append(f"data parquet missing required fields: {missing_data}")
        except Exception as exc:
            errors.append(f"data parquet probe failed: {exc}")

    for key in video_keys(meta):
        required = {
            f"videos/{key}/chunk_index",
            f"videos/{key}/file_index",
            f"videos/{key}/from_timestamp",
            f"videos/{key}/to_timestamp",
        }
        missing_video = sorted(required - set(episodes_df.columns))
        if missing_video:
            errors.append(f"video metadata missing required fields for {key}: {missing_video}")

    if pkg_version is None:
        warnings.append("LeRobot package version string is unavailable; relying on metadata probes")
    elif codebase_version and pkg_version != codebase_version:
        warnings.append(
            f"LeRobot package version ({pkg_version}) differs from dataset codebase_version "
            f"({codebase_version}); relying on metadata probes"
        )
    if codebase_version is None:
        warnings.append("Dataset codebase_version is unavailable")

    if errors:
        status = "unsupported"
    elif warnings:
        status = "supported_with_warning"
    else:
        status = "supported"

    return meta, CompatibilityResult(status, warnings, errors, pkg_version, codebase_version, root)
