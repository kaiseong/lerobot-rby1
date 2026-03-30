#!/usr/bin/env python

from unittest.mock import patch

import numpy as np

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy_config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DUMMY_REPO_ID = "dummy/repo"


def _make_training_dataset(tmp_path, empty_lerobot_dataset_factory):
    features = {
        ACTION: {"dtype": "float32", "shape": (2,), "names": None},
        OBS_STATE: {"dtype": "float32", "shape": (4,), "names": None},
        f"{OBS_IMAGES}.front": {
            "dtype": "image",
            "shape": (8, 8, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "_dataset", features=features, fps=10)
    for ep_idx in range(4):
        for frame_idx in range(3):
            dataset.add_frame(
                {
                    ACTION: np.full((2,), frame_idx, dtype=np.float32),
                    OBS_STATE: np.arange(4, dtype=np.float32) + ep_idx,
                    f"{OBS_IMAGES}.front": np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8),
                    "task": f"task_{ep_idx}",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return dataset


def _dummy_update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, **kwargs):
    train_metrics.loss = 0.0
    train_metrics.grad_norm = 0.0
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = 0.0
    return train_metrics, {}


def test_train_uses_save_freq_as_default_val_freq(tmp_path, empty_lerobot_dataset_factory):
    dataset = _make_training_dataset(tmp_path, empty_lerobot_dataset_factory)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=str(dataset.root), val_ratio=0.25),
        policy=make_policy_config(
            "act",
            device="cpu",
            push_to_hub=False,
            pretrained_backbone_weights=None,
        ),
        output_dir=tmp_path / "_output_default",
        steps=3,
        save_freq=2,
        log_freq=0,
        num_workers=0,
        batch_size=2,
        seed=123,
        save_checkpoint=False,
    )

    validation_calls = []

    def _dummy_run_validation(*args, **kwargs):
        validation_calls.append("called")
        return {"val/loss": 0.0}

    with (
        patch("lerobot.scripts.lerobot_train.update_policy", _dummy_update_policy),
        patch("lerobot.scripts.lerobot_train.run_validation", _dummy_run_validation),
    ):
        train(cfg)

    assert cfg.val_freq == 2
    assert len(validation_calls) == 2


def test_train_respects_explicit_val_freq(tmp_path, empty_lerobot_dataset_factory):
    dataset = _make_training_dataset(tmp_path, empty_lerobot_dataset_factory)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=str(dataset.root), val_ratio=0.25),
        policy=make_policy_config(
            "act",
            device="cpu",
            push_to_hub=False,
            pretrained_backbone_weights=None,
        ),
        output_dir=tmp_path / "_output_override",
        steps=4,
        save_freq=10,
        val_freq=2,
        log_freq=0,
        num_workers=0,
        batch_size=2,
        seed=123,
        save_checkpoint=False,
    )

    validation_calls = []

    def _dummy_run_validation(*args, **kwargs):
        validation_calls.append("called")
        return {"val/loss": 0.0}

    with (
        patch("lerobot.scripts.lerobot_train.update_policy", _dummy_update_policy),
        patch("lerobot.scripts.lerobot_train.run_validation", _dummy_run_validation),
    ):
        train(cfg)

    assert cfg.val_freq == 2
    assert len(validation_calls) == 2
