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

from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F  # type: ignore  # noqa: N812


def normalize_resize_size(resize_size: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(resize_size) != 2:
        raise ValueError(f"resize_size must contain two values, got {resize_size}")

    resize_height, resize_width = (int(v) for v in resize_size)
    if resize_height <= 0 or resize_width <= 0:
        raise ValueError(f"resize_size must use positive values, got {resize_size}")

    return (resize_height, resize_width)


def normalize_crop_params_dict(
    crop_params_dict: dict[str, tuple[int, int, int, int] | list[int]] | None,
) -> dict[str, tuple[int, int, int, int]]:
    if not crop_params_dict:
        return {}

    normalized = {}
    for key, value in crop_params_dict.items():
        if len(value) != 4:
            raise ValueError(f"crop params for '{key}' must have four values, got {value}")

        top, left, height, width = (int(v) for v in value)
        if top < 0 or left < 0:
            raise ValueError(f"crop params for '{key}' must use non-negative top/left offsets, got {value}")
        if height <= 0 or width <= 0:
            raise ValueError(f"crop params for '{key}' must use positive height/width, got {value}")

        normalized[key] = (top, left, height, width)

    return normalized


def crop_raw_observation_image(
    image: np.ndarray | torch.Tensor, crop_params: tuple[int, int, int, int]
) -> np.ndarray | torch.Tensor:
    if getattr(image, "ndim", None) != 3:
        raise ValueError(f"Expected an image with 3 dimensions (H, W, C), got {getattr(image, 'shape', None)}")

    top, left, height, width = crop_params
    image_height, image_width = image.shape[:2]
    bottom = top + height
    right = left + width
    if bottom > image_height or right > image_width:
        raise ValueError(f"Crop {crop_params} exceeds image bounds {(image_height, image_width)}")

    cropped = image[top:bottom, left:right, ...]
    if isinstance(cropped, np.ndarray):
        return np.ascontiguousarray(cropped)
    if isinstance(cropped, torch.Tensor):
        return cropped.contiguous()

    return cropped


def apply_observation_crops(
    raw_observation: dict[str, Any],
    crop_params_dict: dict[str, tuple[int, int, int, int]] | None,
) -> dict[str, Any]:
    if not crop_params_dict:
        return raw_observation

    cropped_observation = dict(raw_observation)
    for key, crop_params in crop_params_dict.items():
        if key not in raw_observation:
            raise KeyError(f"Crop requested for missing observation key '{key}'")
        cropped_observation[key] = crop_raw_observation_image(raw_observation[key], crop_params)

    return cropped_observation


def resize_with_pad_torch(image: torch.Tensor, resize_size: tuple[int, int] | list[int]) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
        raise ValueError(
            f"Expected image tensor with shape [C,H,W] or [B,C,H,W], got {type(image)} / {getattr(image, 'shape', None)}"
        )

    target_height, target_width = normalize_resize_size(resize_size)
    cur_height, cur_width = image.shape[-2:]

    if cur_height == target_height and cur_width == target_width:
        resized = image
    else:
        ratio = max(cur_width / target_width, cur_height / target_height)
        resized_height = max(1, int(cur_height / ratio))
        resized_width = max(1, int(cur_width / ratio))
        resized = F.resize(image, [resized_height, resized_width], antialias=True)

    pad_h0, remainder_h = divmod(target_height - resized.shape[-2], 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(target_width - resized.shape[-1], 2)
    pad_w1 = pad_w0 + remainder_w

    return F.pad(resized, [pad_w0, pad_h0, pad_w1, pad_h1], fill=0)

