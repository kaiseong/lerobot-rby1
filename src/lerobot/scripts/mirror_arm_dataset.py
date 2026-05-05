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

"""Create a mirrored left/right arm LeRobot dataset for symmetric data augmentation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, mirror_arm_dataset
from lerobot.utils.utils import init_logging


@dataclass
class MirrorArmDatasetConfig:
    # Input dataset identifier.
    repo_id: str
    # Local root for the input dataset. Defaults to the standard LeRobot cache for repo_id.
    root: str | None = None
    # Output dataset identifier. Defaults to "<repo_id>_mirrored".
    new_repo_id: str | None = None
    # Local root for the output dataset. Defaults to the standard LeRobot cache for new_repo_id.
    new_root: str | None = None
    # One of: right_to_left, left_to_right, both.
    mirror_mode: str = "right_to_left"
    # Include the original episode before each mirrored episode. Only valid for mirror_mode="both".
    include_original: bool = False
    # Vector feature keys.
    state_key: str = "observation.state"
    action_key: str = "action"
    # Camera feature keys. Use null/None for front_camera_key to disable front flipping.
    front_camera_key: str | None = "observation.images.front"
    right_camera_key: str | None = "observation.images.right"
    left_camera_key: str | None = "observation.images.left"
    # Per-arm zero-based joint indices whose sign changes under left/right mirroring.
    sign_flip_indices: list[int] = field(default_factory=lambda: [1, 2, 4])
    # Output camera storage. "image" writes PNG frames and avoids extra video compression.
    visual_storage: str = "image"
    # Async image writer settings. None uses a CPU-aware default; 0 disables threads.
    image_writer_threads: int | None = None
    image_writer_processes: int = 0
    # Upload the generated output dataset to the Hugging Face Hub.
    push_to_hub: bool = False


def run_mirror_arm_dataset(cfg: MirrorArmDatasetConfig) -> LeRobotDataset:
    if not cfg.repo_id:
        raise ValueError("--repo_id is required")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    output_repo_id = cfg.new_repo_id or f"{cfg.repo_id}_mirrored"
    input_root = Path(cfg.root).resolve() if cfg.root else dataset.root.resolve()
    output_root = Path(cfg.new_root).resolve() if cfg.new_root else None
    if output_root is not None and output_root == input_root:
        raise ValueError(
            "new_root must point to a different directory; mirror augmentation never edits in place"
        )

    logging.info("Mirroring dataset %s with mode=%s", cfg.repo_id, cfg.mirror_mode)
    new_dataset = mirror_arm_dataset(
        dataset,
        output_dir=output_root,
        repo_id=output_repo_id,
        mirror_mode=cfg.mirror_mode,
        include_original=cfg.include_original,
        state_key=cfg.state_key,
        action_key=cfg.action_key,
        front_camera_key=cfg.front_camera_key,
        right_camera_key=cfg.right_camera_key,
        left_camera_key=cfg.left_camera_key,
        sign_flip_indices=cfg.sign_flip_indices,
        visual_storage=cfg.visual_storage,
        image_writer_processes=cfg.image_writer_processes,
        image_writer_threads=cfg.image_writer_threads,
    )

    logging.info(
        "Mirrored dataset saved to %s (episodes=%s, frames=%s)",
        new_dataset.root,
        new_dataset.meta.total_episodes,
        new_dataset.meta.total_frames,
    )

    if cfg.push_to_hub:
        logging.info("Pushing mirrored dataset to hub as %s", output_repo_id)
        new_dataset.push_to_hub()

    return new_dataset


@parser.wrap()
def mirror_arm_dataset_cli(cfg: MirrorArmDatasetConfig) -> None:
    run_mirror_arm_dataset(cfg)


def main() -> None:
    init_logging()
    mirror_arm_dataset_cli()


if __name__ == "__main__":
    main()
