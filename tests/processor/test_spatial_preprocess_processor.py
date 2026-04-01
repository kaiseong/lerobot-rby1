#!/usr/bin/env python

from pathlib import Path

import torch

from lerobot.processor import (
    ObservationImageSpatialPreprocessStep,
    PolicyProcessorPipeline,
    preprocessor_has_active_spatial_preprocess,
)


def test_spatial_preprocess_crop_resize_roundtrip(tmp_path: Path):
    image = torch.arange(3 * 8 * 12, dtype=torch.float32).reshape(1, 3, 8, 12) / 255.0
    pipeline = PolicyProcessorPipeline(
        steps=[
            ObservationImageSpatialPreprocessStep(
                mode="crop",
                crop_params_dict={"observation.images.front": (2, 3, 4, 4)},
                resize_size=(6, 6),
            )
        ],
        name="policy_preprocessor",
    )

    output = pipeline(
        {
            "observation.images.front": image.clone(),
            "observation.state": torch.zeros(1, 4),
        }
    )

    assert output["observation.images.front"].shape == (1, 3, 6, 6)

    pipeline.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")

    assert preprocessor_has_active_spatial_preprocess(loaded) is True
    assert loaded.steps[0].get_config()["crop_params_dict"] == {"observation.images.front": (2, 3, 4, 4)}


def test_spatial_preprocess_resize_pad_applies_zero_padding():
    image = torch.ones(1, 3, 4, 8)
    step = ObservationImageSpatialPreprocessStep(mode="resize_pad", resize_size=(8, 8))

    output = step.observation({"observation.images.front": image})
    padded = output["observation.images.front"]

    assert padded.shape == (1, 3, 8, 8)
    assert torch.all(padded[:, :, 0, :] == 0)
    assert torch.all(padded[:, :, -1, :] == 0)
