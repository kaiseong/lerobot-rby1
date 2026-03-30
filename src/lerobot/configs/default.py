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

from dataclasses import dataclass, field

from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetCropConfig:
    enable: bool = False
    resize_size: tuple[int, int] = (128, 128)
    params: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.resize_size) != 2:
            raise ValueError(f"resize_size must contain two values, got {self.resize_size}")

        resize_height, resize_width = (int(v) for v in self.resize_size)
        if resize_height <= 0 or resize_width <= 0:
            raise ValueError(f"resize_size must use positive values, got {self.resize_size}")
        self.resize_size = (resize_height, resize_width)

        normalized_params = {}
        for key, value in self.params.items():
            if len(value) != 4:
                raise ValueError(
                    f"dataset.crop.params['{key}'] must have four values (top, left, height, width), got {value}"
                )

            top, left, height, width = (int(v) for v in value)
            if top < 0 or left < 0:
                raise ValueError(
                    f"dataset.crop.params['{key}'] must use non-negative top/left offsets, got {value}"
                )
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"dataset.crop.params['{key}'] must use positive height/width, got {value}"
                )

            normalized_params[key] = (top, left, height, width)

        self.params = normalized_params


@dataclass
class DatasetSourceConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    adapter: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.adapter = self.adapter.strip()
        if not self.repo_id:
            raise ValueError("dataset.sources[*].repo_id must be provided")
        if not self.adapter:
            raise ValueError(f"dataset.sources[{self.repo_id}] must define an adapter")
        self.weight = float(self.weight)
        if self.weight <= 0:
            raise ValueError(f"dataset.sources[{self.repo_id}].weight must be positive, got {self.weight}")


@dataclass
class DatasetConfig:
    repo_id: str = ""
    root: str | None = None
    episodes: list[int] | None = None
    sources: list[DatasetSourceConfig] = field(default_factory=list)
    mixing_strategy: str = "uniform"
    crop: DatasetCropConfig = field(default_factory=DatasetCropConfig)
    val_ratio: float = 0.0
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False

    def __post_init__(self) -> None:
        self.val_ratio = float(self.val_ratio)
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"dataset.val_ratio must be in [0.0, 1.0), got {self.val_ratio}")

        if self.mixing_strategy not in {"uniform", "manual"}:
            raise ValueError(
                f"dataset.mixing_strategy must be one of ['uniform', 'manual'], got {self.mixing_strategy}"
            )

        if self.sources:
            if self.repo_id:
                raise ValueError("Use either dataset.repo_id or dataset.sources, not both")
            if self.root is not None:
                raise ValueError("dataset.root is only supported for single-dataset training")
            if self.episodes is not None:
                raise ValueError("dataset.episodes is only supported for single-dataset training")
            if self.revision is not None:
                raise ValueError("dataset.revision is only supported for single-dataset training")
            if self.crop.enable:
                raise ValueError("dataset.crop is not supported together with dataset.sources")
            if self.streaming:
                raise ValueError("dataset.streaming is not supported together with dataset.sources")
            if self.val_ratio > 0:
                raise ValueError("dataset.val_ratio is not supported together with dataset.sources")
            if self.mixing_strategy == "manual":
                for source in self.sources:
                    if source.weight <= 0:
                        raise ValueError(
                            f"dataset.sources[{source.repo_id}].weight must be positive when using manual mixing"
                        )
        elif self.val_ratio > 0 and self.streaming:
            raise ValueError("dataset.val_ratio is not supported together with dataset.streaming")
        elif not self.repo_id:
            raise ValueError("Either dataset.repo_id or dataset.sources must be provided")


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class PeftConfig:
    # PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    # effective methods so we'll focus on those in this high-level config interface.

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16
