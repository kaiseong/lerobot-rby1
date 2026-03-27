import numpy as np
import pytest
import torch

from lerobot.configs.default import DatasetConfig, DatasetSourceConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.mixed_dataset import HeterogeneousLeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy_config
from lerobot.utils.constants import ACTION, OBS_STATE


CANONICAL_NAMES = [
    *[f"torso_{i}" for i in range(6)],
    *[f"right_arm_{i}" for i in range(7)],
    *[f"left_arm_{i}" for i in range(7)],
    "right_gripper_0",
    "left_gripper_0",
]


def _make_features(state_names, action_names, camera_specs):
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (len(state_names),), "names": list(state_names)},
        ACTION: {"dtype": "float32", "shape": (len(action_names),), "names": list(action_names)},
    }
    for key, shape in camera_specs.items():
        features[key] = {"dtype": "image", "shape": shape, "names": ["height", "width", "channels"]}
    return features


def _make_local_dataset(root, repo_id, state_names, action_names, camera_specs, episode_lengths):
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=15,
        features=_make_features(state_names, action_names, camera_specs),
        root=root,
        use_videos=False,
    )

    for episode_index, episode_length in enumerate(episode_lengths):
        for frame_index in range(episode_length):
            frame = {
                "task": f"task_{episode_index}",
                OBS_STATE: torch.arange(len(state_names), dtype=torch.float32) + frame_index,
                ACTION: torch.arange(len(action_names), dtype=torch.float32) + 0.1 * frame_index,
            }
            for cam_key, shape in camera_specs.items():
                height, width, _channels = shape
                frame[cam_key] = np.full((height, width, 3), fill_value=episode_index + frame_index, dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset


def _make_train_cfg(tmp_path, mixing_strategy="uniform", manual_weights=None):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_state_names = [*CANONICAL_NAMES[6:13], "right_gripper"]
    new_state_names = CANONICAL_NAMES

    _make_local_dataset(
        old_root,
        repo_id="local/old",
        state_names=old_state_names,
        action_names=old_state_names,
        camera_specs={
            "observation.images.head": (8, 8, 3),
            "observation.images.right_wrist": (8, 8, 3),
        },
        episode_lengths=[2],
    )
    _make_local_dataset(
        new_root,
        repo_id="local/new",
        state_names=new_state_names,
        action_names=new_state_names,
        camera_specs={
            "observation.images.front": (8, 8, 3),
            "observation.images.right": (8, 8, 3),
            "observation.images.left": (8, 8, 3),
        },
        episode_lengths=[2, 2],
    )

    weights = manual_weights or (1.0, 1.0)
    dataset_cfg = DatasetConfig(
        sources=[
            DatasetSourceConfig(
                repo_id="local/old",
                root=str(old_root),
                adapter="rby1_right_arm_legacy",
                weight=weights[0],
            ),
            DatasetSourceConfig(
                repo_id="local/new",
                root=str(new_root),
                adapter="rby1_bimanual_torso",
                weight=weights[1],
            ),
        ],
        mixing_strategy=mixing_strategy,
    )
    policy_cfg = make_policy_config(policy_type="pi05", chunk_size=2, n_action_steps=2)
    return TrainPipelineConfig(dataset=dataset_cfg, policy=policy_cfg, output_dir=tmp_path / "out")


def test_make_dataset_with_sources_builds_canonical_mixed_dataset(tmp_path):
    cfg = _make_train_cfg(tmp_path)
    dataset = make_dataset(cfg)

    assert isinstance(dataset, HeterogeneousLeRobotDataset)
    assert dataset.num_frames == 6
    assert dataset.num_episodes == 3

    old_item = dataset[0]
    assert old_item[OBS_STATE].shape == (22,)
    assert old_item[ACTION].shape == (2, 22)
    assert old_item["observation.images.left_is_missing"].item() is True
    assert old_item["observation.images.front_is_missing"].item() is False
    assert old_item["observation.images.right_is_missing"].item() is False
    assert old_item[f"{OBS_STATE}_dim_is_pad"][:6].all()
    assert not old_item[f"{OBS_STATE}_dim_is_pad"][6:13].any()
    assert old_item[f"{OBS_STATE}_dim_is_pad"][13:20].all()
    assert not old_item[f"{OBS_STATE}_dim_is_pad"][20].item()
    assert old_item[f"{OBS_STATE}_dim_is_pad"][21].item()

    new_item = dataset[2]
    assert not new_item["observation.images.left_is_missing"].item()
    assert not new_item[f"{OBS_STATE}_dim_is_pad"].any()
    assert not new_item["action_dim_is_pad"].any()

    sampler = dataset.make_sampler()
    weights = sampler.weights.tolist()
    assert weights[0] == pytest.approx(0.5)
    assert weights[-1] == pytest.approx(0.25)


def test_make_dataset_with_torso_missing_source_preserves_bimanual_inputs(tmp_path):
    torso_missing_root = tmp_path / "no_torso"
    full_root = tmp_path / "full"
    no_torso_names = CANONICAL_NAMES[6:]

    _make_local_dataset(
        torso_missing_root,
        repo_id="local/no_torso",
        state_names=no_torso_names,
        action_names=no_torso_names,
        camera_specs={
            "observation.images.front": (8, 8, 3),
            "observation.images.right": (8, 8, 3),
            "observation.images.left": (8, 8, 3),
        },
        episode_lengths=[2],
    )
    _make_local_dataset(
        full_root,
        repo_id="local/full",
        state_names=CANONICAL_NAMES,
        action_names=CANONICAL_NAMES,
        camera_specs={
            "observation.images.front": (8, 8, 3),
            "observation.images.right": (8, 8, 3),
            "observation.images.left": (8, 8, 3),
        },
        episode_lengths=[2],
    )

    dataset_cfg = DatasetConfig(
        sources=[
            DatasetSourceConfig(
                repo_id="local/no_torso",
                root=str(torso_missing_root),
                adapter="rby1_bimanual_no_torso",
            ),
            DatasetSourceConfig(
                repo_id="local/full",
                root=str(full_root),
                adapter="rby1_bimanual_torso",
            ),
        ],
        mixing_strategy="uniform",
    )
    policy_cfg = make_policy_config(policy_type="pi05", chunk_size=2, n_action_steps=2)
    cfg = TrainPipelineConfig(dataset=dataset_cfg, policy=policy_cfg, output_dir=tmp_path / "out")

    dataset = make_dataset(cfg)
    no_torso_item = dataset[0]

    assert no_torso_item[OBS_STATE].shape == (22,)
    assert no_torso_item[ACTION].shape == (2, 22)
    assert no_torso_item[f"{OBS_STATE}_dim_is_pad"][:6].all()
    assert not no_torso_item[f"{OBS_STATE}_dim_is_pad"][6:].any()
    assert no_torso_item["action_dim_is_pad"][:6].all()
    assert not no_torso_item["action_dim_is_pad"][6:].any()
    assert not no_torso_item["observation.images.front_is_missing"].item()
    assert not no_torso_item["observation.images.right_is_missing"].item()
    assert not no_torso_item["observation.images.left_is_missing"].item()


def test_make_dataset_with_sources_manual_weights_override_uniform(tmp_path):
    cfg = _make_train_cfg(tmp_path, mixing_strategy="manual", manual_weights=(1.0, 3.0))
    dataset = make_dataset(cfg)

    sampler = dataset.make_sampler()
    weights = sampler.weights.tolist()
    assert weights[0] == pytest.approx(0.5)
    assert weights[-1] == pytest.approx(0.75)
