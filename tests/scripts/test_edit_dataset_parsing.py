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

from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_edit_dataset as edit_dataset_module
from lerobot.scripts.lerobot_edit_dataset import (
    ConvertImageToVideoConfig,
    DeleteEpisodesConfig,
    EditDatasetConfig,
    InfoConfig,
    MergeConfig,
    ModifyTasksConfig,
    OperationConfig,
    RemoveFeatureConfig,
    SplitConfig,
    TrimEpisodeEdgesConfig,
    TrimStationaryEpisodeEdgesConfig,
    _validate_config,
)


def parse_cfg(cli_args: list[str]) -> EditDatasetConfig:
    """Helper to parse CLI args into an EditDatasetConfig via draccus."""
    return draccus.parse(EditDatasetConfig, args=cli_args)


class TestOperationTypeParsing:
    """Test that --operation.type correctly selects the right config subclass."""

    @pytest.mark.parametrize(
        "type_name, expected_cls",
        [
            ("delete_episodes", DeleteEpisodesConfig),
            ("split", SplitConfig),
            ("merge", MergeConfig),
            ("remove_feature", RemoveFeatureConfig),
            ("modify_tasks", ModifyTasksConfig),
            ("trim_episode_edges", TrimEpisodeEdgesConfig),
            ("trim_stationary_episode_edges", TrimStationaryEpisodeEdgesConfig),
            ("convert_image_to_video", ConvertImageToVideoConfig),
            ("info", InfoConfig),
        ],
    )
    def test_operation_type_resolves_correct_class(self, type_name, expected_cls):
        cfg = parse_cfg(
            ["--repo_id", "test/repo", "--new_repo_id", "test/merged", "--operation.type", type_name]
        )
        assert isinstance(cfg.operation, expected_cls), (
            f"Expected {expected_cls.__name__}, got {type(cfg.operation).__name__}"
        )

    def test_merge_requires_new_repo_id(self):
        cfg = parse_cfg(["--operation.type", "merge"])
        with pytest.raises(ValueError, match="--new_repo_id is required for merge"):
            _validate_config(cfg)

    def test_non_merge_requires_repo_id(self):
        cfg = parse_cfg(["--operation.type", "delete_episodes"])
        with pytest.raises(ValueError, match="--repo_id is required for delete_episodes"):
            _validate_config(cfg)

    @pytest.mark.parametrize(
        "type_name, expected_cls",
        [
            ("delete_episodes", DeleteEpisodesConfig),
            ("split", SplitConfig),
            ("merge", MergeConfig),
            ("remove_feature", RemoveFeatureConfig),
            ("modify_tasks", ModifyTasksConfig),
            ("trim_episode_edges", TrimEpisodeEdgesConfig),
            ("trim_stationary_episode_edges", TrimStationaryEpisodeEdgesConfig),
            ("convert_image_to_video", ConvertImageToVideoConfig),
            ("info", InfoConfig),
        ],
    )
    def test_get_choice_name_roundtrips(self, type_name, expected_cls):
        cfg = parse_cfg(
            ["--repo_id", "test/repo", "--new_repo_id", "test/merged", "--operation.type", type_name]
        )
        resolved_name = OperationConfig.get_choice_name(type(cfg.operation))
        assert resolved_name == type_name


@pytest.mark.parametrize(
    "operation, handler_name, tool_name",
    [
        (
            TrimEpisodeEdgesConfig(trim_start_seconds=1.0, trim_end_seconds=0.5),
            "handle_trim_episode_edges",
            "trim_episode_edges",
        ),
        (
            TrimStationaryEpisodeEdgesConfig(keep_start_seconds=1.0, keep_end_seconds=0.5),
            "handle_trim_stationary_episode_edges",
            "trim_stationary_episode_edges",
        ),
    ],
)
def test_trim_handlers_pass_new_root_to_get_output_path(
    monkeypatch, tmp_path, operation, handler_name, tool_name
):
    dataset = SimpleNamespace(root=tmp_path / "input")
    new_dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=1, total_frames=3))
    captured_paths = {}
    captured_trim_kwargs = {}

    monkeypatch.setattr(edit_dataset_module, "LeRobotDataset", lambda *args, **kwargs: dataset)

    def fake_get_output_path(repo_id, new_repo_id, root, new_root):
        captured_paths.update(
            {"repo_id": repo_id, "new_repo_id": new_repo_id, "root": root, "new_root": new_root}
        )
        return "test/output", tmp_path / "output"

    def fake_trim(dataset_arg, **kwargs):
        assert dataset_arg is dataset
        captured_trim_kwargs.update(kwargs)
        return new_dataset

    monkeypatch.setattr(edit_dataset_module, "get_output_path", fake_get_output_path)
    monkeypatch.setattr(edit_dataset_module, tool_name, fake_trim)

    cfg = EditDatasetConfig(
        operation=operation,
        repo_id="test/input",
        root=str(tmp_path / "input"),
        new_repo_id="test/output",
        new_root=str(tmp_path / "output"),
        push_to_hub=False,
    )

    getattr(edit_dataset_module, handler_name)(cfg)

    assert captured_paths == {
        "repo_id": "test/input",
        "new_repo_id": "test/output",
        "root": Path(tmp_path / "input"),
        "new_root": Path(tmp_path / "output"),
    }
    assert captured_trim_kwargs["repo_id"] == "test/output"
    assert captured_trim_kwargs["output_dir"] == tmp_path / "output"
