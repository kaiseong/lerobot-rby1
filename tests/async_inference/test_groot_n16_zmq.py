from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.async_inference.groot_n16_zmq import (
    GROOT_N16_ACTION_KEYS,
    build_groot_n16_observation,
    groot_n16_action_dict_to_timed_actions,
    validate_groot_robot_compatibility,
)
from tests.mocks.mock_robot import MockRobotConfig


def test_groot_backend_config_allows_missing_policy_fields():
    from lerobot.async_inference.configs import RobotClientConfig

    cfg = RobotClientConfig(
        robot=MockRobotConfig(),
        actions_per_chunk=16,
        backend="groot_n16_zmq",
        server_address="192.168.0.3:5555",
        groot_image_size=[480, 480],
    )

    assert cfg.policy_type == ""
    assert cfg.pretrained_name_or_path == ""
    assert cfg.groot_image_size == (480, 480)
    assert cfg.to_dict()["backend"] == "groot_n16_zmq"


def test_grpc_backend_config_requires_policy_fields():
    from lerobot.async_inference.configs import RobotClientConfig

    with pytest.raises(ValueError, match="policy_type"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            actions_per_chunk=16,
            backend="grpc",
            server_address="localhost:8080",
        )


def test_build_groot_n16_observation_packs_and_resizes_images():
    front = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    right = np.ones((240, 240, 3), dtype=np.float32)
    raw_observation = {
        **{f"right_arm_{i}": float(i) for i in range(7)},
        "right_gripper_0": 1.0,
        "front": front,
        "right": right,
        "task": "pick",
    }

    packed = build_groot_n16_observation(
        raw_observation,
        front_camera_key="front",
        right_wrist_camera_key="right",
        image_size=(480, 480),
    )

    assert packed["video"]["cam_front_head"].shape == (1, 1, 480, 480, 3)
    assert packed["video"]["cam_front_head"].dtype == np.uint8
    assert packed["video"]["cam_right_wrist"].shape == (1, 1, 480, 480, 3)
    assert packed["state"]["right_arm"].shape == (1, 1, 7)
    assert packed["state"]["right_arm"].dtype == np.float32
    assert packed["state"]["right_gripper"].shape == (1, 1, 1)
    assert packed["language"]["annotation.human.task_description"] == [["pick"]]


def test_groot_action_dict_to_timed_actions_preserves_order_and_timing():
    action_dict = {
        "right_arm": np.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                ]
            ],
            dtype=np.float32,
        ),
        "right_gripper": np.array([[[0.25], [0.75]]], dtype=np.float32),
    }

    timed_actions = groot_n16_action_dict_to_timed_actions(
        action_dict,
        timestamp=100.0,
        timestep=7,
        environment_dt=0.1,
    )

    assert len(timed_actions) == 2
    assert timed_actions[0].get_timestep() == 7
    assert timed_actions[1].get_timestep() == 8
    assert timed_actions[0].get_timestamp() == pytest.approx(100.0)
    assert timed_actions[1].get_timestamp() == pytest.approx(100.1)
    torch.testing.assert_close(
        timed_actions[0].get_action(),
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]),
    )
    torch.testing.assert_close(
        timed_actions[1].get_action(),
        torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.75]),
    )


def test_validate_groot_robot_compatibility_rejects_non_right_arm_layout():
    class FakeRobot:
        action_features = {"left_arm_0": float}
        observation_features = {"left_arm_0": float, "front": (480, 480, 3), "right": (480, 480, 3)}

    with pytest.raises(ValueError, match="right_arm_0"):
        validate_groot_robot_compatibility(
            FakeRobot(),
            front_camera_key="front",
            right_wrist_camera_key="right",
        )


def test_robot_client_groot_backend_enqueues_and_executes_actions(monkeypatch):
    pytest.importorskip("grpc")

    import lerobot.async_inference.robot_client as robot_client_module
    from lerobot.async_inference.configs import RobotClientConfig

    class FakeRobot:
        def __init__(self):
            self.front = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
            self.right = np.full((480, 480, 3), 127, dtype=np.uint8)
            self.action_features = {key: float for key in GROOT_N16_ACTION_KEYS}
            self.observation_features = {
                **self.action_features,
                "front": (480, 640, 3),
                "right": (480, 480, 3),
            }
            self.is_connected = False
            self.sent_actions = []

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def get_observation(self):
            return {
                **{f"right_arm_{i}": float(i) for i in range(7)},
                "right_gripper_0": 1.0,
                "front": self.front.copy(),
                "right": self.right.copy(),
            }

        def send_action(self, action):
            self.sent_actions.append(action)
            return action

    class FakeRemoteClient:
        def __init__(self):
            self.last_observation = None
            self.closed = False

        def ping(self):
            return True

        def get_action(self, observation):
            self.last_observation = observation
            arm = np.stack(
                [np.arange(7, dtype=np.float32) + i for i in range(16)],
                axis=0,
            )[np.newaxis, ...]
            gripper = np.linspace(1.0, 0.0, 16, dtype=np.float32).reshape(1, 16, 1)
            return {"right_arm": arm, "right_gripper": gripper}

        def close(self):
            self.closed = True

    fake_robot = FakeRobot()
    fake_remote = FakeRemoteClient()

    monkeypatch.setattr(robot_client_module, "make_robot_from_config", lambda cfg: fake_robot)
    monkeypatch.setattr(robot_client_module, "GR00TZMQClient", lambda **kwargs: fake_remote)

    cfg = RobotClientConfig(
        robot=MockRobotConfig(),
        actions_per_chunk=16,
        backend="groot_n16_zmq",
        server_address="192.168.0.3:5555",
        image_crop_params={"front": [0, 80, 480, 480]},
    )
    client = robot_client_module.RobotClient(cfg)

    assert client.start() is True

    raw_observation = client.control_loop_groot_inference(task="pick")

    assert raw_observation is not None
    assert raw_observation["front"].shape == (480, 480, 3)
    np.testing.assert_array_equal(raw_observation["front"], fake_robot.front[:, 80:560, :])
    assert fake_remote.last_observation["video"]["cam_front_head"].shape == (1, 1, 480, 480, 3)
    assert fake_remote.last_observation["language"]["annotation.human.task_description"] == [["pick"]]
    assert client.actions_available() is True
    assert client.action_chunk_size == 16

    performed = client.control_loop_action()
    assert performed["right_arm_0"] == pytest.approx(0.0)
    assert performed["right_arm_6"] == pytest.approx(6.0)
    assert performed["right_gripper_0"] == pytest.approx(1.0)

    client.stop()
    assert fake_remote.closed is True
    assert fake_robot.is_connected is False
