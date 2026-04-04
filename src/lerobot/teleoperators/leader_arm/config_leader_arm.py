from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rby1_leader_arm")
@dataclass
class Rby1LeaderArmConfig(TeleoperatorConfig):
    """Configuration for the RB-Y1 leader arm (physical master arm) teleoperator.

    The leader arm is a 14-DOF device (7 per arm) with per-hand trigger
    and button.  Joint positions are read from the leader arm's Dynamixel
    servos and returned directly as the teleoperator action — following
    LeRobot convention where the teleoperator does NOT send commands to
    the robot itself.
    """

    # ── Leader arm device ─────────────────────────────────────────────
    # Path to the leader arm URDF model (required for gravity compensation).
    # If None, auto-detected relative to the rby1_sdk installation.
    leader_arm_model_path: str | None = None

    # Leader arm control callback frequency in Hz.
    control_frequency: float = 100.0

    # ── Joint group selection ─────────────────────────────────────────
    # Must match the corresponding flags in Rby1Config so that
    # action_features are consistent between teleoperator and robot.
    use_torso: bool = False       # Leader arm has no torso DOFs
    use_right_arm: bool = True
    use_left_arm: bool = True

    # Enable gripper control via trigger buttons on the leader arm.
    use_gripper: bool = True

    # ── Initialisation trajectory ─────────────────────────────────────
    # Duration (seconds) for the smooth cosine interpolation from the
    # current leader arm pose to the init pose on connect().
    init_duration: float = 5.0

    # Init (ready) pose for each side in degrees.
    # Matches the robot ready pose defined in rby1_vr.py and rby1.py so that
    # the leader arm and robot start in the same configuration.
    right_init_q_deg: List[float] = field(
        default_factory=lambda: [20.0, -55.0, -30.0, -95.0, 55.0, -70.0, -90.0]
        
    )
    left_init_q_deg: List[float] = field(
        default_factory=lambda: [20.0, 55.0, 30.0, -95.0, -55.0, -70.0, 90.0]
    )

    # ── Leader arm joint limits (14 DOF, degrees) ─────────────────────
    # Indices 0-6: right arm, 7-13: left arm.
    min_q_deg: List[float] = field(
        default_factory=lambda: [
            -180, -60, -15, -150, -180,  -90, -155,
            -180,  35, -105, -150, -180,  -90, -155,
        ]
    )
    max_q_deg: List[float] = field(
        default_factory=lambda: [
            180, -35, 105, 1,  180, 110, 155,
            180,  60, 15, 1,  180, 110, 155,
        ]
    )

    # ── Leader arm torque / safety parameters ─────────────────────────
    # Maximum torque applied per joint (Nm).
    torque_limit: List[float] = field(
        default_factory=lambda: [3.5, 3.5, 3.5, 1.5, 0.5, 0.5, 0.5] * 2
    )

    # Viscous damping gain per joint.
    viscous_gain: List[float] = field(
        default_factory=lambda: [0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2
    )

    # Barrier gain at joint limits (Nm/rad).
    joint_limit_barrier: float = 1.5

    # Scaling factor for gravity compensation torque when the arm is in
    # free (current-control) mode (button pressed).
    gravity_comp_scale: float = 0.4

    # ── Wrist offset (leader → robot mapping) ────────────────────────
    # Mechanical zero-offset for the last joint of each arm (degrees).
    # Leader-arm reads 0 → robot receives +offset.
    right_wrist_offset_deg: float = 90.0
    left_wrist_offset_deg: float = -90.0

    # ── Gripper trigger ───────────────────────────────────────────────
    # Raw trigger range is [0, gripper_trigger_max].
    # Normalised to 0.0–1.0 for the action dict.
    gripper_trigger_max: float = 1000.0

    # ── Startup ───────────────────────────────────────────────────────
    # Seconds to wait after starting the control loop for the leader arm
    # to reach the ready pose before returning from connect().
    startup_wait_time: float = 5.0
