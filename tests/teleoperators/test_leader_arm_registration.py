#!/usr/bin/env python

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.leader_arm import Rby1LeaderArm, Rby1LeaderArmConfig


def test_rby1_leader_arm_registers_with_package_import():
    assert "rby1_leader_arm" in TeleoperatorConfig._choice_registry


def test_make_teleoperator_from_config_creates_rby1_leader_arm():
    teleop = make_teleoperator_from_config(Rby1LeaderArmConfig())
    assert isinstance(teleop, Rby1LeaderArm)
