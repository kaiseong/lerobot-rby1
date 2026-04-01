from lerobot.robots.rby1.config_rby1 import Rby1Config
from lerobot.robots.rby1.rby1 import Rby1


def test_rby1_can_observe_torso_without_commanding_it():
    cfg = Rby1Config(
        cameras={},
        use_torso=False,
        observe_torso=True,
        use_right_arm=True,
        use_left_arm=True,
        use_gripper=True,
        move_to_ready_on_connect=False,
    )
    robot = Rby1(cfg)

    observation_keys = list(robot.observation_features)
    action_keys = list(robot.action_features)

    assert observation_keys[:6] == [f"torso_{i}" for i in range(6)]
    assert "torso_0" not in action_keys
    assert action_keys == [*[f"right_arm_{i}" for i in range(7)], *[f"left_arm_{i}" for i in range(7)], "right_gripper_0", "left_gripper_0"]
