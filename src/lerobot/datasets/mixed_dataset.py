from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from lerobot.configs.default import DatasetSourceConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import ACTION, OBS_STATE

RBY1_RIGHT_ARM_LEGACY_ADAPTER = "rby1_right_arm_legacy"
RBY1_BIMANUAL_TORSO_ADAPTER = "rby1_bimanual_torso"
RBY1_BIMANUAL_NO_TORSO_ADAPTER = "rby1_bimanual_no_torso"
SUPPORTED_RBY1_ADAPTERS = {
    RBY1_RIGHT_ARM_LEGACY_ADAPTER,
    RBY1_BIMANUAL_TORSO_ADAPTER,
    RBY1_BIMANUAL_NO_TORSO_ADAPTER,
}

CANONICAL_CAMERA_KEYS = [
    "observation.images.front",
    "observation.images.right",
    "observation.images.left",
]
CANONICAL_STATE_NAMES = [
    *[f"torso_{i}" for i in range(6)],
    *[f"right_arm_{i}" for i in range(7)],
    *[f"left_arm_{i}" for i in range(7)],
    "right_gripper_0",
    "left_gripper_0",
]
CANONICAL_ACTION_NAMES = list(CANONICAL_STATE_NAMES)
_CANONICAL_DIM = len(CANONICAL_STATE_NAMES)

_DIM_ALIASES = {
    name: [name] for name in CANONICAL_STATE_NAMES
}
_DIM_ALIASES["right_gripper_0"].append("right_gripper")
_DIM_ALIASES["left_gripper_0"].append("left_gripper")

_CAMERA_ALIAS_BASENAMES = {
    "observation.images.front": ["front", "head", "top"],
    "observation.images.right": ["right", "right_wrist", "right_hand"],
    "observation.images.left": ["left", "left_wrist", "left_hand"],
}


@dataclass
class MixedDatasetMetadata:
    repo_id: str
    fps: int
    features: dict[str, dict]
    stats: dict[str, dict[str, Any]]
    root: Path | None = None
    revision: str | None = None

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]


@dataclass
class _SourceRuntime:
    config: DatasetSourceConfig
    dataset: LeRobotDataset
    meta: LeRobotDatasetMetadata
    frame_offset: int
    episode_offset: int
    adapter: "_Rby1SourceAdapter"


class _Rby1SourceAdapter:
    def __init__(self, adapter_name: str, source_meta: LeRobotDatasetMetadata, canonical_meta: MixedDatasetMetadata):
        if adapter_name not in SUPPORTED_RBY1_ADAPTERS:
            raise ValueError(f"Unsupported adapter '{adapter_name}'")

        self.adapter_name = adapter_name
        self.source_meta = source_meta
        self.canonical_meta = canonical_meta
        self.state_index_map = self._build_dim_index_map(source_meta.features[OBS_STATE]["names"], OBS_STATE)
        self.action_index_map = self._build_dim_index_map(source_meta.features[ACTION]["names"], ACTION)
        self.camera_key_map = self._build_camera_key_map(source_meta.camera_keys)
        self.camera_tensor_shapes = {
            key: self._camera_feature_to_tensor_shape(canonical_meta.features[key]) for key in CANONICAL_CAMERA_KEYS
        }

    def _is_full_bimanual_adapter(self) -> bool:
        return self.adapter_name in {RBY1_BIMANUAL_TORSO_ADAPTER, RBY1_BIMANUAL_NO_TORSO_ADAPTER}

    def _build_dim_index_map(self, source_names: list[str] | None, feature_key: str) -> dict[int, int]:
        if not source_names:
            raise ValueError(f"Feature '{feature_key}' must define named dimensions for mixed RBY1 training")

        source_name_to_index = {name: idx for idx, name in enumerate(source_names)}
        index_map: dict[int, int] = {}
        for canonical_index, canonical_name in enumerate(CANONICAL_STATE_NAMES):
            for candidate in _DIM_ALIASES[canonical_name]:
                if candidate in source_name_to_index:
                    index_map[canonical_index] = source_name_to_index[candidate]
                    break

        if self.adapter_name == RBY1_BIMANUAL_TORSO_ADAPTER:
            missing = [name for idx, name in enumerate(CANONICAL_STATE_NAMES) if idx not in index_map]
            if missing:
                raise ValueError(
                    f"Adapter '{self.adapter_name}' requires full 22D '{feature_key}' coverage. Missing: {missing}"
                )
        elif self.adapter_name == RBY1_BIMANUAL_NO_TORSO_ADAPTER:
            required = [*CANONICAL_STATE_NAMES[6:20], "right_gripper_0", "left_gripper_0"]
            missing = [
                name
                for name in required
                if all(candidate not in source_name_to_index for candidate in _DIM_ALIASES[name])
            ]
            if missing:
                raise ValueError(
                    f"Adapter '{self.adapter_name}' requires bilateral arm and gripper dims in '{feature_key}'. Missing: {missing}"
                )
        else:
            required = [*CANONICAL_STATE_NAMES[6:13], "right_gripper_0"]
            missing = [
                name
                for name in required
                if all(candidate not in source_name_to_index for candidate in _DIM_ALIASES[name])
            ]
            if missing:
                raise ValueError(
                    f"Adapter '{self.adapter_name}' requires right-arm legacy dims in '{feature_key}'. Missing: {missing}"
                )

        return index_map

    def _build_camera_key_map(self, source_camera_keys: list[str]) -> dict[str, str | None]:
        mapping: dict[str, str | None] = {}
        for canonical_key, aliases in _CAMERA_ALIAS_BASENAMES.items():
            resolved = None
            for source_key in source_camera_keys:
                basename = source_key.split(".")[-1]
                if source_key == canonical_key or basename in aliases:
                    resolved = source_key
                    break
            mapping[canonical_key] = resolved

        if self._is_full_bimanual_adapter():
            missing = [key for key, value in mapping.items() if value is None]
            if missing:
                raise ValueError(
                    f"Adapter '{self.adapter_name}' requires front/right/left cameras. Missing: {missing}"
                )
        else:
            required = ["observation.images.front", "observation.images.right"]
            missing = [key for key in required if mapping[key] is None]
            if missing:
                raise ValueError(
                    f"Adapter '{self.adapter_name}' requires front/right legacy cameras. Missing: {missing}"
                )

        return mapping

    @staticmethod
    def _camera_feature_to_tensor_shape(feature: dict[str, Any]) -> tuple[int, int, int]:
        height, width, channels = feature["shape"]
        return (channels, height, width)

    @staticmethod
    def _as_float_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(torch.float32)
        return torch.as_tensor(value, dtype=torch.float32)

    @staticmethod
    def _as_image_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    def _adapt_vector(self, value: Any, index_map: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = self._as_float_tensor(value)
        vector_shape = tensor.shape
        if tensor.dim() == 1:
            adapted = torch.zeros((_CANONICAL_DIM,), dtype=tensor.dtype, device=tensor.device)
        elif tensor.dim() == 2:
            adapted = torch.zeros((vector_shape[0], _CANONICAL_DIM), dtype=tensor.dtype, device=tensor.device)
        else:
            raise ValueError(f"Expected 1D or 2D tensor for vector feature, got shape {vector_shape}")

        dim_is_pad = torch.ones((_CANONICAL_DIM,), dtype=torch.bool, device=tensor.device)
        for canonical_index, source_index in index_map.items():
            adapted[..., canonical_index] = tensor[..., source_index]
            dim_is_pad[canonical_index] = False

        return adapted, dim_is_pad.cpu()

    def transform(self, item: dict[str, Any], global_index: int, episode_offset: int) -> dict[str, Any]:
        adapted: dict[str, Any] = {}

        state, state_dim_is_pad = self._adapt_vector(item[OBS_STATE], self.state_index_map)
        action, action_dim_is_pad = self._adapt_vector(item[ACTION], self.action_index_map)
        adapted[OBS_STATE] = state
        adapted[ACTION] = action
        adapted[f"{OBS_STATE}_dim_is_pad"] = state_dim_is_pad
        adapted[f"{ACTION}_dim_is_pad"] = action_dim_is_pad

        first_real_image = None
        for canonical_key in CANONICAL_CAMERA_KEYS:
            source_key = self.camera_key_map[canonical_key]
            if source_key is not None:
                first_real_image = self._as_image_tensor(item[source_key])
                break
        if first_real_image is None:
            raise ValueError("At least one real camera is required for mixed RBY1 training")

        for canonical_key in CANONICAL_CAMERA_KEYS:
            source_key = self.camera_key_map[canonical_key]
            if source_key is None:
                image = torch.zeros(
                    self.camera_tensor_shapes[canonical_key],
                    dtype=first_real_image.dtype,
                )
                is_missing = True
            else:
                image = self._as_image_tensor(item[source_key])
                is_missing = False

            adapted[canonical_key] = image
            adapted[f"{canonical_key}_is_missing"] = torch.tensor(is_missing, dtype=torch.bool)

        passthrough_keys = [
            "timestamp",
            "frame_index",
            "task",
            "task_index",
            "subtask",
            "subtask_index",
            "action_is_pad",
        ]
        for key in passthrough_keys:
            if key in item:
                adapted[key] = item[key]

        adapted["episode_index"] = item["episode_index"] + episode_offset
        adapted["index"] = torch.tensor(global_index, dtype=torch.int64)

        return adapted


def _build_canonical_metadata(full_source_meta: LeRobotDatasetMetadata, source_repo_ids: list[str]) -> MixedDatasetMetadata:
    selected_keys = [
        *[key for key in DEFAULT_FEATURES if key in full_source_meta.features],
        OBS_STATE,
        ACTION,
        *CANONICAL_CAMERA_KEYS,
    ]
    features = {key: deepcopy(full_source_meta.features[key]) for key in selected_keys if key in full_source_meta.features}
    features[OBS_STATE]["shape"] = (_CANONICAL_DIM,)
    features[OBS_STATE]["names"] = list(CANONICAL_STATE_NAMES)
    features[ACTION]["shape"] = (_CANONICAL_DIM,)
    features[ACTION]["names"] = list(CANONICAL_ACTION_NAMES)
    stats = {key: deepcopy(full_source_meta.stats[key]) for key in features if key in full_source_meta.stats}
    repo_id = "+".join(source_repo_ids)
    return MixedDatasetMetadata(repo_id=f"mixed/{repo_id}", fps=full_source_meta.fps, features=features, stats=stats)


class HeterogeneousLeRobotDataset(Dataset):
    def __init__(self, sources: list[_SourceRuntime], meta: MixedDatasetMetadata, mixing_strategy: str):
        if not sources:
            raise ValueError("At least one source dataset is required")
        self.sources = sources
        self.meta = meta
        self.mixing_strategy = mixing_strategy
        self.num_frames = sum(source.dataset.num_frames for source in sources)
        self.num_episodes = sum(source.dataset.num_episodes for source in sources)
        self.episodes = None
        self.features = meta.features
        self.repo_id = meta.repo_id
        self.root = None
        self._source_frame_starts = [source.frame_offset for source in sources]

    def __len__(self) -> int:
        return self.num_frames

    def _resolve_source(self, idx: int) -> tuple[_SourceRuntime, int]:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(idx)
        source_index = bisect_right(self._source_frame_starts, idx) - 1
        source = self.sources[source_index]
        local_idx = idx - source.frame_offset
        return source, local_idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source, local_idx = self._resolve_source(idx)
        item = source.dataset[local_idx]
        return source.adapter.transform(item, global_index=idx, episode_offset=source.episode_offset)

    def make_sampler(self) -> WeightedRandomSampler | None:
        if self.mixing_strategy not in {"uniform", "manual"}:
            return None

        weights = torch.empty(self.num_frames, dtype=torch.double)
        for source in self.sources:
            start = source.frame_offset
            end = start + source.dataset.num_frames
            if self.mixing_strategy == "uniform":
                source_weight = 1.0
            else:
                source_weight = source.config.weight
            weights[start:end] = source_weight / source.dataset.num_frames

        return WeightedRandomSampler(weights, num_samples=self.num_frames, replacement=True)


def build_rby1_mixed_dataset(sources: list[tuple[DatasetSourceConfig, LeRobotDataset, LeRobotDatasetMetadata]], mixing_strategy: str) -> HeterogeneousLeRobotDataset:
    full_source_meta = None
    for source_cfg, _dataset, source_meta in sources:
        if source_cfg.adapter == RBY1_BIMANUAL_TORSO_ADAPTER:
            full_source_meta = source_meta
            break
    if full_source_meta is None:
        raise ValueError(
            f"At least one source with adapter '{RBY1_BIMANUAL_TORSO_ADAPTER}' is required to define the canonical schema"
        )

    canonical_meta = _build_canonical_metadata(full_source_meta, [cfg.repo_id for cfg, _, _ in sources])

    runtimes: list[_SourceRuntime] = []
    frame_offset = 0
    episode_offset = 0
    for source_cfg, dataset, source_meta in sources:
        adapter = _Rby1SourceAdapter(source_cfg.adapter, source_meta=source_meta, canonical_meta=canonical_meta)
        runtimes.append(
            _SourceRuntime(
                config=source_cfg,
                dataset=dataset,
                meta=source_meta,
                frame_offset=frame_offset,
                episode_offset=episode_offset,
                adapter=adapter,
            )
        )
        frame_offset += dataset.num_frames
        episode_offset += dataset.num_episodes

    return HeterogeneousLeRobotDataset(runtimes, meta=canonical_meta, mixing_strategy=mixing_strategy)
