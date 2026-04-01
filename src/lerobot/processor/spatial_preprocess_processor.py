#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as F  # type: ignore  # noqa: N812

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.image_preprocessing import (
    normalize_crop_params_dict,
    normalize_resize_size,
    resize_with_pad_torch,
)

SPATIAL_PREPROCESSOR_REGISTRY_NAME = "observation_image_spatial_preprocess"


@ProcessorStepRegistry.register(SPATIAL_PREPROCESSOR_REGISTRY_NAME)
@dataclass
class ObservationImageSpatialPreprocessStep(ObservationProcessorStep):
    mode: str = "none"
    resize_size: tuple[int, int] | None = None
    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in {"none", "crop", "resize_pad"}:
            raise ValueError(
                f"mode must be one of ['none', 'crop', 'resize_pad'], got {self.mode!r}"
            )

        self.crop_params_dict = normalize_crop_params_dict(self.crop_params_dict)
        if self.resize_size is not None:
            self.resize_size = normalize_resize_size(self.resize_size)

        if self.mode in {"crop", "resize_pad"} and self.resize_size is None:
            raise ValueError(f"resize_size must be provided when mode={self.mode!r}")

    @property
    def is_active(self) -> bool:
        return self.mode != "none"

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.is_active:
            return observation

        new_observation = dict(observation)
        for key, value in observation.items():
            if "image" not in key or not isinstance(value, torch.Tensor):
                continue

            transformed = value
            if self.mode == "crop":
                crop_params = self.crop_params_dict.get(key)
                if crop_params is not None:
                    transformed = F.crop(transformed, *crop_params)
                transformed = F.resize(transformed, list(self.resize_size), antialias=True)
            elif self.mode == "resize_pad":
                transformed = resize_with_pad_torch(transformed, self.resize_size)

            if transformed.is_floating_point():
                transformed = transformed.clamp(0.0, 1.0)
            new_observation[key] = transformed

        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "resize_size": self.resize_size,
            "crop_params_dict": self.crop_params_dict,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if self.resize_size is None or not self.is_active:
            return features

        transformed_features = deepcopy(features)
        for key, feature in transformed_features[PipelineFeatureType.OBSERVATION].items():
            if "image" not in key:
                continue
            nb_channel = feature.shape[0]
            transformed_features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                type=feature.type,
                shape=(nb_channel, *self.resize_size),
            )

        return transformed_features


def preprocessor_has_active_spatial_preprocess(preprocessor) -> bool:
    if preprocessor is None:
        return False

    for step in getattr(preprocessor, "steps", []):
        if isinstance(step, ObservationImageSpatialPreprocessStep):
            return step.is_active
        if getattr(step.__class__, "_registry_name", None) == SPATIAL_PREPROCESSOR_REGISTRY_NAME:
            return getattr(step, "mode", "none") != "none"

    return False

