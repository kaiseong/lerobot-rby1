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

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torchvision.transforms.functional as F  # type: ignore  # noqa: N812
from tqdm import tqdm  # type: ignore

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import resolve_vcodec
from lerobot.utils.constants import DONE, REWARD
from lerobot.utils.image_preprocessing import (
    normalize_crop_params_dict,
    normalize_resize_size,
    resize_with_pad_torch,
)


def select_rect_roi(img):
    import cv2

    """
    Allows the user to draw a rectangular ROI on the image.

    The user must click and drag to draw the rectangle.
    - While dragging, the rectangle is dynamically drawn.
    - On mouse button release, the rectangle is fixed.
    - Press 'c' to confirm the selection.
    - Press 'r' to reset the selection.
    - Press ESC to cancel.

    Returns:
        A tuple (top, left, height, width) representing the rectangular ROI,
        or None if no valid ROI is selected.
    """
    # Create a working copy of the image
    clone = img.copy()
    working_img = clone.copy()

    roi = None  # Will store the final ROI as (top, left, height, width)
    drawing = False
    index_x, index_y = -1, -1  # Initial click coordinates

    def mouse_callback(event, x, y, flags, param):
        nonlocal index_x, index_y, drawing, roi, working_img

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing: record starting coordinates
            drawing = True
            index_x, index_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Compute the top-left and bottom-right corners regardless of drag direction
                top = min(index_y, y)
                left = min(index_x, x)
                bottom = max(index_y, y)
                right = max(index_x, x)
                # Show a temporary image with the current rectangle drawn
                temp = working_img.copy()
                cv2.rectangle(temp, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.imshow("Select ROI", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            drawing = False
            top = min(index_y, y)
            left = min(index_x, x)
            bottom = max(index_y, y)
            right = max(index_x, x)
            height = bottom - top
            width = right - left
            roi = (top, left, height, width)  # (top, left, height, width)
            # Draw the final rectangle on the working image and display it
            working_img = clone.copy()
            cv2.rectangle(working_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Select ROI", working_img)

    # Create the window and set the callback
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)
    cv2.imshow("Select ROI", working_img)

    print("Instructions for ROI selection:")
    print("  - Click and drag to draw a rectangular ROI.")
    print("  - Press 'c' to confirm the selection.")
    print("  - Press 'r' to reset and draw again.")
    print("  - Press ESC to cancel the selection.")

    # Wait until the user confirms with 'c', resets with 'r', or cancels with ESC
    while True:
        key = cv2.waitKey(1) & 0xFF
        # Confirm ROI if one has been drawn
        if key == ord("c") and roi is not None:
            break
        # Reset: clear the ROI and restore the original image
        elif key == ord("r"):
            working_img = clone.copy()
            roi = None
            cv2.imshow("Select ROI", working_img)
        # Cancel selection for this image
        elif key == 27:  # ESC key
            roi = None
            break

    cv2.destroyWindow("Select ROI")
    return roi


def select_square_roi_for_images(images: dict) -> dict:
    """
    For each image in the provided dictionary, open a window to allow the user
    to select a rectangular ROI. Returns a dictionary mapping each key to a tuple
    (top, left, height, width) representing the ROI.

    Parameters:
        images (dict): Dictionary where keys are identifiers and values are OpenCV images.

    Returns:
        dict: Mapping of image keys to the selected rectangular ROI.
    """
    selected_rois = {}

    for key, img in images.items():
        if img is None:
            print(f"Image for key '{key}' is None, skipping.")
            continue

        print(f"\nSelect rectangular ROI for image with key: '{key}'")
        roi = select_rect_roi(img)

        if roi is None:
            print(f"No valid ROI selected for '{key}'.")
        else:
            selected_rois[key] = roi
            print(f"ROI for '{key}': {roi}")

    return selected_rois


def get_image_from_lerobot_dataset(dataset: LeRobotDataset):
    """
    Find the first row in the dataset and extract the image in order to be used for the crop.
    """
    row = dataset[0]
    image_dict = {}
    for k in row:
        if "image" in k:
            image_dict[k] = deepcopy(row[k])
    return image_dict


def _resolve_crop_vcodec(dataset: LeRobotDataset) -> str:
    if not dataset.meta.video_keys:
        return "libsvtav1"

    first_video_key = dataset.meta.video_keys[0]
    source_vcodec = dataset.meta.features[first_video_key].get("info", {}).get("video.codec", "libsvtav1")
    codec_aliases = {"av1": "libsvtav1", "h265": "hevc", "x265": "hevc", "x264": "h264"}
    requested_vcodec = codec_aliases.get(str(source_vcodec).lower(), str(source_vcodec).lower())
    try:
        return resolve_vcodec(requested_vcodec)
    except ValueError:
        logging.warning(
            "Unsupported source video codec '%s' in dataset metadata; falling back to 'libsvtav1'",
            source_vcodec,
        )
        return "libsvtav1"


def convert_lerobot_dataset_to_cropped_lerobot_dataset(
    original_dataset: LeRobotDataset,
    crop_params_dict: dict[str, tuple[int, int, int, int] | list[int]],
    new_repo_id: str,
    new_dataset_root: str | Path,
    resize_size: tuple[int, int] = (128, 128),
    push_to_hub: bool = False,
    task: str | None = None,
) -> LeRobotDataset:
    """
    Converts an existing LeRobotDataset by iterating over its episodes and frames,
    applying cropping and resizing to image observations, and saving a new dataset
    with the transformed data.

    Args:
        original_dataset (LeRobotDataset): The source dataset.
        crop_params_dict: A dictionary mapping observation keys to crop parameters
            (top, left, height, width).
        new_repo_id (str): Repository id for the new dataset.
        new_dataset_root (str | Path): Root directory where the new dataset will be written.
        resize_size (Tuple[int, int], optional): Target size (height, width) after cropping.
        push_to_hub (bool, optional): Whether to push the generated dataset to the Hugging Face Hub.
        task (str | None, optional): Optional task override applied to every frame.
            When omitted or empty, source task labels are preserved.

    Returns:
        LeRobotDataset: A new LeRobotDataset where the specified image observations have been cropped
        and resized.
    """
    crop_params_dict = normalize_crop_params_dict(crop_params_dict)
    camera_keys = set(original_dataset.meta.camera_keys)
    crop_keys = set(crop_params_dict)
    missing_keys = sorted(camera_keys - crop_keys)
    unknown_keys = sorted(crop_keys - camera_keys)
    if missing_keys:
        raise ValueError(f"Missing crop params for camera keys: {missing_keys}")
    if unknown_keys:
        raise ValueError(f"Unknown crop params for non-camera keys: {unknown_keys}")

    resize_size = normalize_resize_size(resize_size)

    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=deepcopy(original_dataset.meta.info["features"]),
        use_videos=len(original_dataset.meta.video_keys) > 0,
        vcodec=_resolve_crop_vcodec(original_dataset),
    )
    new_dataset.meta.update_chunk_settings(
        chunks_size=original_dataset.meta.chunks_size,
        data_files_size_in_mb=original_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=original_dataset.meta.video_files_size_in_mb,
    )

    for key in crop_params_dict:
        if key in new_dataset.meta.info["features"]:
            new_dataset.meta.info["features"][key]["shape"] = [*resize_size, 3]

    previous_episode_index = None
    for frame_idx in tqdm(range(len(original_dataset))):
        frame = original_dataset[frame_idx]
        frame_episode_index = int(frame["episode_index"].item())
        if previous_episode_index is None:
            previous_episode_index = frame_episode_index
        elif frame_episode_index != previous_episode_index:
            new_dataset.save_episode()
            previous_episode_index = frame_episode_index

        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            if key in (DONE, REWARD) and isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)

            if key in crop_params_dict:
                top, left, height, width = crop_params_dict[key]
                cropped = F.crop(value, top, left, height, width)
                value = F.resize(cropped, resize_size)
                if isinstance(value, torch.Tensor) and value.dim() == 3:
                    value = value.permute(1, 2, 0)
                value = value.clamp(0, 1)
            if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)
            new_frame[key] = value

        new_frame["task"] = task if task else frame["task"]
        new_dataset.add_frame(new_frame)

    if previous_episode_index is not None:
        new_dataset.save_episode()

    new_dataset.finalize()

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


def convert_lerobot_dataset_to_resized_padded_lerobot_dataset(
    original_dataset: LeRobotDataset,
    new_repo_id: str,
    new_dataset_root: str | Path,
    resize_size: tuple[int, int] = (224, 224),
    push_to_hub: bool = False,
    task: str | None = None,
) -> LeRobotDataset:
    """Convert a LeRobot dataset by applying openpi-style resize-with-pad to every camera image."""
    resize_size = normalize_resize_size(resize_size)

    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=deepcopy(original_dataset.meta.info["features"]),
        use_videos=len(original_dataset.meta.video_keys) > 0,
        vcodec=_resolve_crop_vcodec(original_dataset),
    )
    new_dataset.meta.update_chunk_settings(
        chunks_size=original_dataset.meta.chunks_size,
        data_files_size_in_mb=original_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=original_dataset.meta.video_files_size_in_mb,
    )

    for key in original_dataset.meta.camera_keys:
        if key in new_dataset.meta.info["features"]:
            new_dataset.meta.info["features"][key]["shape"] = [*resize_size, 3]

    previous_episode_index = None
    for frame_idx in tqdm(range(len(original_dataset))):
        frame = original_dataset[frame_idx]
        frame_episode_index = int(frame["episode_index"].item())
        if previous_episode_index is None:
            previous_episode_index = frame_episode_index
        elif frame_episode_index != previous_episode_index:
            new_dataset.save_episode()
            previous_episode_index = frame_episode_index

        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            if key in (DONE, REWARD) and isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)

            if key in original_dataset.meta.camera_keys:
                value = resize_with_pad_torch(value, resize_size)
                value = value.permute(1, 2, 0).clamp(0, 1)

            if key.startswith("complementary_info") and isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)
            new_frame[key] = value

        new_frame["task"] = task if task else frame["task"]
        new_dataset.add_frame(new_frame)

    if previous_episode_index is not None:
        new_dataset.save_episode()

    new_dataset.finalize()

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop rectangular ROIs from a LeRobot dataset.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot",
        help="The repository id of the LeRobot dataset to process.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="The root directory of the LeRobot dataset.",
    )
    parser.add_argument(
        "--crop-params-path",
        type=str,
        default=None,
        help="The path to the JSON file containing the ROIs.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether to push the new dataset to the hub.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="The natural language task to describe the dataset.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="The repository id for the new cropped and resized dataset. If not provided, it defaults to `repo_id` + '_cropped_resized'.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    images = get_image_from_lerobot_dataset(dataset)
    images = {k: v.cpu().permute(1, 2, 0).numpy() for k, v in images.items()}
    images = {k: (v * 255).astype("uint8") for k, v in images.items()}

    if args.crop_params_path is None:
        rois = select_square_roi_for_images(images)
    else:
        with open(args.crop_params_path) as f:
            rois = json.load(f)

    # Print the selected rectangular ROIs
    print("\nSelected Rectangular Regions of Interest (top, left, height, width):")
    for key, roi in rois.items():
        print(f"{key}: {roi}")

    new_repo_id = args.new_repo_id if args.new_repo_id else args.repo_id + "_cropped_resized"

    if args.new_repo_id:
        new_dataset_name = args.new_repo_id.split("/")[-1]
        # Parent 1: HF user, Parent 2: HF LeRobot Home
        new_dataset_root = dataset.root.parent.parent / new_dataset_name
    else:
        new_dataset_root = Path(str(dataset.root) + "_cropped_resized")

    cropped_resized_dataset = convert_lerobot_dataset_to_cropped_lerobot_dataset(
        original_dataset=dataset,
        crop_params_dict=rois,
        new_repo_id=new_repo_id,
        new_dataset_root=new_dataset_root,
        resize_size=(128, 128),
        push_to_hub=args.push_to_hub,
        task=args.task,
    )

    meta_dir = new_dataset_root / "meta"
    meta_dir.mkdir(exist_ok=True)

    with open(meta_dir / "crop_params.json", "w") as f:
        json.dump(rois, f, indent=4)
