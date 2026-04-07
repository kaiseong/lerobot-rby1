#!/usr/bin/env python

import numpy as np

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.leader_arm import Rby1LeaderArm, Rby1LeaderArmConfig


def test_rby1_leader_arm_registers_with_package_import():
    assert "rby1_leader_arm" in TeleoperatorConfig._choice_registry


def test_make_teleoperator_from_config_creates_rby1_leader_arm():
    teleop = make_teleoperator_from_config(Rby1LeaderArmConfig())
    assert isinstance(teleop, Rby1LeaderArm)


def test_get_action_uses_latched_joint_targets():
    cfg = Rby1LeaderArmConfig(use_gripper=False)
    teleop = Rby1LeaderArm(cfg)
    teleop._is_connected = True
    teleop._right_q = np.deg2rad(cfg.right_init_q_deg)
    teleop._left_q = np.deg2rad(cfg.left_init_q_deg)

    action = teleop.get_action()

    expected_right = np.deg2rad(cfg.right_init_q_deg)
    expected_right[6] += np.deg2rad(cfg.right_wrist_offset_deg)
    expected_left = np.deg2rad(cfg.left_init_q_deg)
    expected_left[6] += np.deg2rad(cfg.left_wrist_offset_deg)

    for i in range(7):
        assert action[f"right_arm_{i}"] == expected_right[i]
        assert action[f"left_arm_{i}"] == expected_left[i]
