#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.mixed_dataset import HeterogeneousLeRobotDataset, build_rby1_mixed_dataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import load_info
from lerobot.rl.crop_dataset_roi import convert_lerobot_dataset_to_cropped_lerobot_dataset
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


@dataclass(frozen=True)
class _ResolvedDatasetLocation:
    repo_id: str
    root: str | None
    revision: str | None


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _serialize_crop_params(dataset_cfg: DatasetConfig) -> dict[str, list[int]]:
    return {key: list(value) for key, value in sorted(dataset_cfg.crop.params.items())}


def _get_crop_cache_key(dataset_cfg: DatasetConfig) -> str:
    payload = {
        "repo_id": dataset_cfg.repo_id,
        "revision": dataset_cfg.revision,
        "source_root": str(Path(dataset_cfg.root).resolve()) if dataset_cfg.root else None,
        "resize_size": list(dataset_cfg.crop.resize_size),
        "params": _serialize_crop_params(dataset_cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _get_processed_dataset_location(dataset_cfg: DatasetConfig) -> _ResolvedDatasetLocation:
    safe_repo_id = dataset_cfg.repo_id.replace("/", "__")
    cache_key = _get_crop_cache_key(dataset_cfg)
    root = HF_LEROBOT_HOME / "_training_preprocessed" / safe_repo_id / cache_key
    repo_id = f"local/{safe_repo_id}__train_crop__{cache_key}"
    return _ResolvedDatasetLocation(repo_id=repo_id, root=str(root), revision=None)


def _is_valid_processed_dataset_root(root: Path) -> bool:
    meta_dir = root / "meta"
    required_paths = [
        meta_dir / "info.json",
        meta_dir / "stats.json",
        meta_dir / "tasks.parquet",
    ]
    if not all(path.exists() for path in required_paths):
        return False
    if not (root / "data").exists():
        return False

    try:
        info = load_info(root)
    except Exception:
        return False

    if info.get("total_episodes", 0) <= 0 or info.get("total_frames", 0) <= 0:
        return False

    has_video_features = any(ft.get("dtype") == "video" for ft in info.get("features", {}).values())
    if has_video_features and not (root / "videos").exists():
        return False

    return True


def _write_crop_metadata(dataset_root: Path, dataset_cfg: DatasetConfig) -> None:
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    crop_metadata = {
        "source_repo_id": dataset_cfg.repo_id,
        "source_revision": dataset_cfg.revision,
        "source_root": dataset_cfg.root,
        "resize_size": list(dataset_cfg.crop.resize_size),
        "params": _serialize_crop_params(dataset_cfg),
    }
    (meta_dir / "crop_params.json").write_text(json.dumps(crop_metadata, indent=2, sort_keys=True))


def _resolve_dataset_location(dataset_cfg: DatasetConfig) -> _ResolvedDatasetLocation:
    if not dataset_cfg.crop.enable:
        return _ResolvedDatasetLocation(
            repo_id=dataset_cfg.repo_id,
            root=dataset_cfg.root,
            revision=dataset_cfg.revision,
        )

    if not dataset_cfg.crop.params:
        raise ValueError("dataset.crop.enable is true but dataset.crop.params is empty")

    source_dataset = LeRobotDataset(
        dataset_cfg.repo_id,
        root=dataset_cfg.root,
        revision=dataset_cfg.revision,
    )
    camera_keys = set(source_dataset.meta.camera_keys)
    crop_keys = set(dataset_cfg.crop.params)
    missing_keys = sorted(camera_keys - crop_keys)
    unknown_keys = sorted(crop_keys - camera_keys)
    if missing_keys:
        raise ValueError(f"Missing crop params for camera keys: {missing_keys}")
    if unknown_keys:
        raise ValueError(f"Unknown crop params for non-camera keys: {unknown_keys}")

    processed_location = _get_processed_dataset_location(dataset_cfg)
    processed_root = Path(processed_location.root)
    if _is_valid_processed_dataset_root(processed_root):
        logging.info("Reusing cropped training dataset from %s", processed_root)
        return processed_location

    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Creating cropped training dataset at %s from source dataset %s",
        processed_root,
        dataset_cfg.repo_id,
    )
    convert_lerobot_dataset_to_cropped_lerobot_dataset(
        original_dataset=source_dataset,
        crop_params_dict=dataset_cfg.crop.params,
        new_repo_id=processed_location.repo_id,
        new_dataset_root=processed_root,
        resize_size=dataset_cfg.crop.resize_size,
        push_to_hub=False,
        task=None,
    )
    _write_crop_metadata(processed_root, dataset_cfg)
    return processed_location


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | HeterogeneousLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset."""
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if cfg.dataset.sources:
        source_datasets = []
        for source_cfg in cfg.dataset.sources:
            source_meta = LeRobotDatasetMetadata(
                source_cfg.repo_id,
                root=source_cfg.root,
                revision=source_cfg.revision,
            )
            delta_timestamps = resolve_delta_timestamps(cfg.policy, source_meta)
            dataset = LeRobotDataset(
                source_cfg.repo_id,
                root=source_cfg.root,
                episodes=source_cfg.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=source_cfg.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
            source_datasets.append((source_cfg, dataset, source_meta))

        dataset = build_rby1_mixed_dataset(
            source_datasets,
            mixing_strategy=cfg.dataset.mixing_strategy,
        )
    else:
        dataset_location = _resolve_dataset_location(cfg.dataset)
        ds_meta = LeRobotDatasetMetadata(
            dataset_location.repo_id,
            root=dataset_location.root,
            revision=dataset_location.revision,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                dataset_location.repo_id,
                root=dataset_location.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_location.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                dataset_location.repo_id,
                root=dataset_location.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_location.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )


    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
