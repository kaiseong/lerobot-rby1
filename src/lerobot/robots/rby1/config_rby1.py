from dataclasses import dataclass, field
from typing import List

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.robots.config import RobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


@RobotConfig.register_subclass("rby1")
@dataclass
class Rby1Config(RobotConfig):
    # gRPC address of the robot (host:port)
    address: str = "192.168.30.1:50051"

    # Robot model variant: "a" (24-DOF), "m" (26-DOF mecanum), "ub" (18-DOF, no base)
    model: str = "m"

    # Include joint velocities in observation_features
    use_velocity: bool = False

    # Include joint torques in observation_features
    use_torque: bool = False

    # Enable physical Dynamixel gripper (set False for simulation / gripper-less setups)
    use_gripper: bool = True

    # Map of camera name -> CameraConfig
    # cameras: dict[str, CameraConfig] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {
    # "top":   OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
    "front": RealSenseCameraConfig(serial_number_or_name="335122270761", fps=15, width=640, height=480), #front
    "right": RealSenseCameraConfig(serial_number_or_name="335122272086", fps=15, width=480, height=640, rotation=Cv2Rotation.ROTATE_90), #right
    "left": RealSenseCameraConfig(serial_number_or_name="230422270977", fps=15, width=480, height=640, rotation=Cv2Rotation.ROTATE_90), #left
    # "front": OpenCVCameraConfig(index_or_path=2, fps=15, width=848, height=480),
    # "right": OpenCVCameraConfig(index_or_path=10, fps=15, width=480, height=848, rotation=Cv2Rotation.ROTATE_90),
    })

    # Passive (observe-only) mode: connect to the robot for state reading but
    # do NOT create a command stream and do NOT send any joint commands in
    # send_action().  Use this when a high-priority teleop stream (e.g.
    # rby1_vr at priority=10) is already controlling the robot so that the
    # low-priority stream does not generate unnecessary traffic or log noise.
    passive_mode: bool = False

    # ── Joint group selection ──────────────────────────────────────────
    # Select which joint groups to include in observation / action features.
    # Disabled groups are held at their current position during send_action().
    use_torso: bool = False
    use_right_arm: bool = True
    use_left_arm: bool = False
    observe_torso: bool = False

    # ── Impedance control ──────────────────────────────────────────────
    # When True, send_action uses JointImpedanceControlCommandBuilder
    # instead of JointPositionCommandBuilder, yielding compliant behaviour
    # similar to the teleoperation stream.
    use_impedance: bool = False

    # Joint stiffness (Nm/rad) for the 20 body joints in order:
    #   torso_0…5 (6), right_arm_0…6 (7), left_arm_0…6 (7)
    # Higher values → stiffer, lower → more compliant.
    impedance_stiffness: List[float] = field(
        default_factory=lambda: [400.0] * 6 + [100.0] * 7 + [100.0] * 7
    )

    # Torque limits (Nm) for the 20 body joints (same ordering as above).
    impedance_torque_limit: List[float] = field(
        default_factory=lambda: [500.0] * 6 + [40.0] * 7 + [40.0] * 7
    )

    # Damping ratio applied to all joints [0.0, 1.0].
    # 0.7 gives critical damping; lower values allow faster motion.
    impedance_damping_ratio: float = 0.7

    # Move robot to ready pose automatically when connect() is called.
    # Set False to skip the ready pose motion (e.g. during unit tests or
    # when the robot is already in a safe position).
    move_to_ready_on_connect: bool = True

    # ── Startup ramp ───────────────────────────────────────────────────
    # At the beginning of a teleop session the follower robot needs to
    # smoothly converge to the leader's current pose.  A large
    # `minimum_time` per command slows the robot down, giving it time to
    # catch up gradually rather than snapping to the first target.
    #
    # startup_ramp_duration : seconds over which minimum_time is linearly
    #   interpolated from startup_min_time → normal value (0.25 s for
    #   impedance, 0.1 s for position).  Set to 0.0 to disable the ramp.
    # startup_min_time      : minimum_time (seconds) used at t=0.
    startup_ramp_duration: float = 5.0
    startup_min_time: float = 5.0
