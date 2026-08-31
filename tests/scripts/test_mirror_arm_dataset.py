#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import draccus
import pytest

from lerobot.scripts import mirror_arm_dataset as mirror_script
from lerobot.scripts.mirror_arm_dataset import MirrorArmDatasetConfig, run_mirror_arm_dataset


def test_mirror_arm_dataset_config_uses_rby1_mirror_defaults():
    cfg = MirrorArmDatasetConfig(repo_id="test/input")

    assert cfg.sign_flip_indices == [1, 2, 4, 6]
    assert cfg.mirror_task_sides is True


def test_mirror_arm_dataset_config_parses_list_options():
    cfg = draccus.parse(
        MirrorArmDatasetConfig,
        args=[
            "--repo_id",
            "test/input",
            "--new_repo_id",
            "test/output",
            "--mirror_mode",
            "left_to_right",
            "--sign_flip_indices",
            "[0,3]",
            "--image_writer_threads",
            "8",
        ],
    )

    assert cfg.repo_id == "test/input"
    assert cfg.new_repo_id == "test/output"
    assert cfg.mirror_mode == "left_to_right"
    assert cfg.sign_flip_indices == [0, 3]
    assert cfg.image_writer_threads == 8


def test_run_mirror_arm_dataset_passes_config_and_pushes(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    dataset = SimpleNamespace(root=input_root)
    push_calls = []
    new_dataset = SimpleNamespace(
        root=output_root,
        meta=SimpleNamespace(total_episodes=2, total_frames=20),
        push_to_hub=lambda: push_calls.append(True),
    )
    captured = {}

    monkeypatch.setattr(mirror_script, "LeRobotDataset", lambda repo_id, root=None: dataset)

    def fake_mirror_arm_dataset(dataset_arg, **kwargs):
        assert dataset_arg is dataset
        captured.update(kwargs)
        return new_dataset

    monkeypatch.setattr(mirror_script, "mirror_arm_dataset", fake_mirror_arm_dataset)

    cfg = MirrorArmDatasetConfig(
        repo_id="test/input",
        root=str(input_root),
        new_repo_id="test/output",
        new_root=str(output_root),
        mirror_mode="both",
        include_original=True,
        mirror_task_sides=False,
        sign_flip_indices=[1, 2, 4, 6],
        visual_storage="image",
        image_writer_threads=8,
        image_writer_processes=0,
        push_to_hub=True,
    )

    assert run_mirror_arm_dataset(cfg) is new_dataset
    assert captured["output_dir"] == output_root.resolve()
    assert captured["repo_id"] == "test/output"
    assert captured["mirror_mode"] == "both"
    assert captured["include_original"] is True
    assert captured["mirror_task_sides"] is False
    assert captured["sign_flip_indices"] == [1, 2, 4, 6]
    assert captured["visual_storage"] == "image"
    assert captured["image_writer_threads"] == 8
    assert captured["image_writer_processes"] == 0
    assert push_calls == [True]


def test_run_mirror_arm_dataset_rejects_in_place_output(monkeypatch, tmp_path):
    dataset = SimpleNamespace(root=tmp_path / "dataset")
    monkeypatch.setattr(mirror_script, "LeRobotDataset", lambda repo_id, root=None: dataset)

    cfg = MirrorArmDatasetConfig(
        repo_id="test/input",
        root=str(dataset.root),
        new_root=str(dataset.root),
    )

    with pytest.raises(ValueError, match="never edits in place"):
        run_mirror_arm_dataset(cfg)


def test_run_mirror_arm_dataset_requires_repo_id():
    cfg = MirrorArmDatasetConfig(repo_id="")

    with pytest.raises(ValueError, match="--repo_id is required"):
        run_mirror_arm_dataset(cfg)
