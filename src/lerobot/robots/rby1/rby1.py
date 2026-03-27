"""
lerobot_robot_rby1/rby1.py

LeRobot Robot interface for the Rainbow Robotics RB-Y1.
Implements the lerobot.robots.Robot abstract class, enabling the RB-Y1
to be used with all LeRobot tools (data collection, teleoperation,
imitation learning inference).

Joint layout assumed (Model M, 26-DOF):
  wheel_fr, wheel_fl, wheel_rr, wheel_rl  - mobility (excluded from obs/action)
  torso_0 … torso_5                        - torso   (6 DOF, included)
  right_arm_0 … right_arm_6               - right arm (7 DOF, included)
  left_arm_0  … left_arm_6                - left arm  (7 DOF, included)
  head_0, head_1                           - head      (excluded from obs/action)
  + two Dynamixel gripper motors (right=0, left=1) on /dev/rby1_gripper

Gripper normalization: 0.0 = fully open, 1.0 = fully closed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rby1 import Rby1Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamixel constants
# ---------------------------------------------------------------------------
_GRIPPER_BAUD_RATE = 2_000_000
_GRIPPER_IDS = [0, 1]          # 0 = right hand, 1 = left hand
_GRIPPER_HOMING_TORQUE = 0.5   # Nm - homing torque (same as original gripper.py)
_GRIPPER_HOMING_STEPS  = 30   # loop count per direction (0.1 s × 30 = 3 s, matches original)
_GRIPPER_POSITION_TORQUE = 5   # Nm - max torque in position mode


# ---------------------------------------------------------------------------
# Gripper helper
# ---------------------------------------------------------------------------

class Rby1Gripper:
    """
    Controls two Dynamixel gripper motors on /dev/rby1_gripper.

    Motor ID 0 → right-hand gripper
    Motor ID 1 → left-hand  gripper

    Positions are normalised: 0.0 = fully open, 1.0 = fully closed.
    """

    def __init__(self) -> None:
        self._bus = None
        # Physical limits in radians, determined during homing
        self._open_rad: np.ndarray = np.zeros(2)   # encoder value at open
        self._close_rad: np.ndarray = np.zeros(2)  # encoder value at closed
        self._homed: bool = False

    # ------------------------------------------------------------------ #
    #  Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        try:
            import rby1_sdk as rby
            import rby1_sdk.upc as upc
        except ImportError as e:
            raise ImportError(
                "rby1_sdk is required for Rby1Gripper. "
                "Install it from the RB-Y1 SDK."
            ) from e

        self._bus = rby.DynamixelBus(upc.GripperDeviceName)

        if not self._bus.open_port():
            raise RuntimeError(f"Failed to open gripper port: {upc.GripperDeviceName}")

        # Set latency_timer to 1ms (default 16ms causes ~20Hz bottleneck)
        upc.initialize_device(upc.GripperDeviceName)
        logger.info("Gripper USB latency_timer set to 1ms.")

        if not self._bus.set_baud_rate(_GRIPPER_BAUD_RATE):
            raise RuntimeError("Failed to set gripper baud rate.")
        self._bus.set_torque_constant([1.0, 1.0])

        for dev_id in _GRIPPER_IDS:
            if not self._bus.ping(dev_id):
                raise RuntimeError(f"Gripper motor {dev_id} did not respond to ping.")

        self._home()
        logger.info("Rby1Gripper connected and homed.")

    def disconnect(self) -> None:
        if self._bus is not None:
            try:
                self._bus.group_sync_write_torque_enable(
                    [(dev_id, 0) for dev_id in _GRIPPER_IDS]
                )
            except Exception:
                pass
            self._bus = None
            self._homed = False
        logger.info("Rby1Gripper disconnected.")

    # ------------------------------------------------------------------ #
    #  Homing                                                               #
    # ------------------------------------------------------------------ #

    def _home(self) -> None:
        """
        Homing sequence ported exactly from work/jhkim/gripper.py.

        direction=0: +torque for 3 s  (GRIPPER_HOMING_STEPS × 0.1 s)
        direction=1: -torque for 3 s

        min_q/max_q are tracked over the full run.
        open_rad  = max_q  (larger encoder value = open position)
        close_rad = min_q  (smaller encoder value = closed position)

        NOTE: The original uses `prev_q = q` (no copy), making counter always
        increment every loop — effectively a fixed 3-second timer per direction,
        not true stall detection.  We replicate that behaviour with a simple
        step counter to avoid gripper stalling at the closed limit indefinitely.
        """
        import rby1_sdk as rby

        def _set_mode(mode: int) -> None:
            self._bus.group_sync_write_torque_enable(
                [(dev_id, rby.DynamixelBus.TorqueDisable) for dev_id in _GRIPPER_IDS]
            )
            self._bus.group_sync_write_operating_mode(
                [(dev_id, mode) for dev_id in _GRIPPER_IDS]
            )
            self._bus.group_sync_write_torque_enable(
                [(dev_id, rby.DynamixelBus.TorqueEnable) for dev_id in _GRIPPER_IDS]
            )

        logger.info("Homing grippers (3 s per direction) …")
        _set_mode(rby.DynamixelBus.CurrentControlMode)

        q     = np.zeros(2, dtype=np.float64)
        min_q = np.full(2,  np.inf)
        max_q = np.full(2, -np.inf)

        for direction in range(2):
            torque = _GRIPPER_HOMING_TORQUE if direction == 0 else -_GRIPPER_HOMING_TORQUE
            for _ in range(_GRIPPER_HOMING_STEPS):
                self._bus.group_sync_write_send_torque(
                    [(dev_id, torque) for dev_id in _GRIPPER_IDS]
                )
                rv = self._bus.group_fast_sync_read_encoder(_GRIPPER_IDS)
                if rv is not None:
                    for dev_id, enc in rv:
                        q[dev_id] = enc
                min_q = np.minimum(min_q, q)
                max_q = np.maximum(max_q, q)
                time.sleep(0.1)

        # Stop current, switch to position mode
        self._bus.group_sync_write_send_torque(
            [(dev_id, 0.0) for dev_id in _GRIPPER_IDS]
        )
        _set_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self._bus.group_sync_write_send_torque(
            [(dev_id, _GRIPPER_POSITION_TORQUE) for dev_id in _GRIPPER_IDS]
        )

        # Motors are mirror-mounted, so encoder direction differs per side.
        # Right (ID=0): high encoder = CLOSED, low encoder = OPEN
        # Left  (ID=1): low encoder  = CLOSED, high encoder = OPEN
        #
        # Trigger mapping in rby1_vr.py:
        #   right: normalized[0] = trigger        → n=1 when closed → close_rad must be CLOSED
        #   left:  normalized[1] = 1 - trigger    → n=0 when closed → open_rad  must be CLOSED
        #
        # Therefore:
        #   open_rad  = [min_q[0]=OPEN,   max_q[1]=CLOSED]
        #   close_rad = [max_q[0]=CLOSED, min_q[1]=OPEN  ]
        self._open_rad  = np.array([min_q[0], max_q[1]])
        self._close_rad = np.array([max_q[0], min_q[1]])

        self._homed = True

        # ── Sanity check: verify range is non-trivial ──────────────────
        for i, side in enumerate(("right", "left")):
            span = abs(self._close_rad[i] - self._open_rad[i])
            if span < 0.01:
                logger.warning(
                    f"[Gripper] {side} motor range is very small ({span:.4f} rad). "
                    "Homing may not have reached physical limits."
                )
            else:
                logger.info(
                    f"[Gripper] {side} (ID={i}): "
                    f"open_rad={self._open_rad[i]:.4f}, "
                    f"close_rad={self._close_rad[i]:.4f}, "
                    f"span={span:.4f} rad"
                )

    # ------------------------------------------------------------------ #
    #  I/O                                                                  #
    # ------------------------------------------------------------------ #

    def set_positions(self, normalized: np.ndarray) -> None:
        """
        Send goal positions to both gripper motors.

        Parameters
        ----------
        normalized : np.ndarray, shape (2,)
            [right, left], values in [0.0 = open, 1.0 = closed].
        """
        if not self._homed:
            return
        normalized = np.clip(normalized, 0.0, 1.0)
        target_rad = self._open_rad + normalized * (self._close_rad - self._open_rad)
        logger.debug(
            f"[Gripper] set_positions: normalized={normalized.round(3)}, "
            f"target_rad={target_rad.round(4)}"
        )
        self._bus.group_sync_write_send_position(
            [(dev_id, float(r)) for dev_id, r in zip(_GRIPPER_IDS, target_rad)]
        )

    def get_positions(self) -> np.ndarray:
        """
        Read current gripper positions.

        Returns
        -------
        np.ndarray, shape (2,)
            [right, left], values in [0.0 = open, 1.0 = closed].
        """
        if not self._homed:
            return np.zeros(2)
        result = self._bus.group_fast_sync_read_encoder(_GRIPPER_IDS)
        if result is None:
            return np.zeros(2)
        enc_map = {dev_id: enc for dev_id, enc in result}
        current_rad = np.array([enc_map[dev_id] for dev_id in _GRIPPER_IDS])

        span = self._close_rad - self._open_rad
        # Avoid division by zero if homing failed to find a range
        safe_span = np.where(np.abs(span) > 1e-6, span, 1.0)
        normalized = np.clip(
            (current_rad - self._open_rad) / safe_span, 0.0, 1.0
        )
        return normalized


# ---------------------------------------------------------------------------
# Joint-name helpers
# ---------------------------------------------------------------------------

_TORSO_NAMES = [f"torso_{i}" for i in range(6)]
_RIGHT_ARM_NAMES = [f"right_arm_{i}" for i in range(7)]
_LEFT_ARM_NAMES = [f"left_arm_{i}" for i in range(7)]
_GRIPPER_NAMES = ["right_gripper_0", "left_gripper_0"]

_ALL_JOINT_NAMES: list[str] = _TORSO_NAMES + _RIGHT_ARM_NAMES + _LEFT_ARM_NAMES

# ---------------------------------------------------------------------------
# Ready pose — matches _READY_POSE in rby1_vr.py
# ---------------------------------------------------------------------------
_READY_TORSO = np.deg2rad([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
_READY_RIGHT = np.deg2rad([75.0, -5.0, 0.0, -110.0, 20.0, -45.0, 60.0])
_READY_LEFT  = np.deg2rad([75.0, 5.0, 0.0, -110.0, -20.0, -45.0, -60.0])
_READY_POSE  = np.concatenate([_READY_TORSO, _READY_RIGHT, _READY_LEFT])  # (20,)
_READY_HEAD  = np.deg2rad([0.0, 45.0])  # head_0, head_1


# ---------------------------------------------------------------------------
# Robot class
# ---------------------------------------------------------------------------

class Rby1(Robot):
    """
    LeRobot Robot implementation for the RainbowRobotics RB-Y1 (Model M).

    Observation / action space
    --------------------------
    22 positional keys (radians):
        torso_0 … torso_5
        right_arm_0 … right_arm_6
        left_arm_0 … left_arm_6
    2 gripper keys (normalised 0-1):
        right_gripper_0
        left_gripper_0
    Optional (controlled by config.use_velocity / config.use_torque):
        <joint>.vel  (rad/s),  <joint>.torque  (Nm) for the 20 arm/torso joints
    Camera keys (observation only):
        <cam_name> → (H, W, 3) shape

    Usage
    -----
        cfg = Rby1Config(address="192.168.30.1:50051")
        robot = Rby1(cfg)
        robot.connect()
        obs = robot.get_observation()
        robot.send_action(obs)   # replicate current pose
        robot.disconnect()
    """

    config_class = Rby1Config
    name = "rby1"

    def __init__(self, config: Rby1Config) -> None:
        super().__init__(config)
        self._config = config
        self._robot = None
        self._model = None
        self._stream = None
        self._gripper: Rby1Gripper | None = None
        self._is_connected: bool = False
        self.cameras = make_cameras_from_configs(config.cameras)
        # Startup ramp: timestamp of the first send_action() call; None until
        # the first command is sent so that the ramp begins from that moment.
        self._action_start_time: float | None = None

    # ------------------------------------------------------------------ #
    #  Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    @property
    def is_calibrated(self) -> bool:
        # RB-Y1 uses absolute encoders - no calibration required.
        return True

    # ------------------------------------------------------------------ #
    #  Features                                                             #
    # ------------------------------------------------------------------ #

    @property
    def _motors_ft(self) -> dict[str, type]:
        names: list[str] = []
        if self._config.use_torso:
            names += _TORSO_NAMES
        if self._config.use_right_arm:
            names += _RIGHT_ARM_NAMES
        if self._config.use_left_arm:
            names += _LEFT_ARM_NAMES
        return {name: float for name in names}

    @property
    def _gripper_ft(self) -> dict[str, type]:
        if not self._config.use_gripper:
            return {}
        names: list[str] = []
        if self._config.use_right_arm:
            names.append("right_gripper_0")
        if self._config.use_left_arm:
            names.append("left_gripper_0")
        return {name: float for name in names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                self._config.cameras[cam].height,
                self._config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {**self._motors_ft, **self._gripper_ft}
        if self._config.use_velocity:
            for name in _ALL_JOINT_NAMES:
                features[f"{name}.vel"] = float
        if self._config.use_torque:
            for name in _ALL_JOINT_NAMES:
                features[f"{name}.torque"] = float
        features.update(self._cameras_ft)
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._gripper_ft}

    # ------------------------------------------------------------------ #
    #  Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError(
                "rby1_sdk is required. Install it from the RB-Y1 SDK."
            ) from e

        logger.info(f"Connecting to RB-Y1 at {self._config.address} …")

        # ── 1. Create and connect ──────────────────────────────────────
        self._robot = rby.create_robot(self._config.address, self._config.model)
        if not self._robot.connect():
            raise ConnectionError(
                f"Failed to connect to RB-Y1 at {self._config.address}."
            )

        # ── 2. Power and servo on ──────────────────────────────────────
        if not self._robot.is_power_on(".*"):
            if not self._robot.power_on(".*"):
                raise RuntimeError("Failed to power on RB-Y1 actuators.")

        if not self._robot.is_servo_on(".*"):
            if not self._robot.servo_on(".*"):
                raise RuntimeError("Failed to servo on RB-Y1 motors.")

        # ── 3. Clear faults and enable control manager ─────────────────
        cm_state = self._robot.get_control_manager_state()
        if cm_state.state in (
            rby.ControlManagerState.State.MajorFault,
            rby.ControlManagerState.State.MinorFault,
        ):
            logger.warning("Clearing RB-Y1 control manager fault …")
            self._robot.reset_fault_control_manager()

        self._robot.enable_control_manager()

        # ── 4. Cache model for joint index arrays ──────────────────────
        self._model = self._robot.model()

        # ── 5. Open command stream (priority=1, lower than teleop's 10) ─
        # Skip in passive_mode: a higher-priority teleop stream is expected
        # to control the robot; this instance only reads state.
        if not self._config.passive_mode:
            self._stream = self._robot.create_command_stream(priority=1)
        else:
            logger.info(
                "passive_mode=True — command stream not created; "
                "robot will be used for observation only."
            )

        # ── 6. Power tool flanges for the Dynamixel gripper bus ────────
        if self._config.use_gripper:
            self._robot.set_tool_flange_output_voltage("right", 12)
            self._robot.set_tool_flange_output_voltage("left", 12)
            time.sleep(0.5)  # allow capacitors to stabilise

            # ── 7. Initialise and home grippers ───────────────────────────
            self._gripper = Rby1Gripper()
            self._gripper.connect()
        else:
            logger.info("Gripper disabled (use_gripper=False) — skipping flange power and gripper init.")

        # ── 8. Connect cameras ─────────────────────────────────────────
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        self.configure()
        logger.info(f"{self} connected.")

        # ── 9. Move to ready pose before inference begins ──────────────
        if self._config.move_to_ready_on_connect:
            self.move_to_ready_pose()

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        # Grippers
        if self._gripper is not None:
            self._gripper.disconnect()
            self._gripper = None

        # Cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Robot
        if self._robot is not None:
            self._robot.disable_control_manager()
            # self._robot.servo_off(".*")
            if self._config.use_gripper:
                self._robot.set_tool_flange_output_voltage("right", 0)
                self._robot.set_tool_flange_output_voltage("left", 0)
            self._robot.disconnect()
            self._robot = None

        self._model = None
        self._stream = None
        self._is_connected = False
        self._action_start_time = None
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ #
    #  Calibration / Configuration                                          #
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        # RB-Y1 uses absolute encoders; no software calibration needed.
        pass

    def reset(self) -> None:
        """
        Called by lerobot-record between episodes to return the robot to its
        ready pose, matching the behaviour of unitree_g1.reset().
        """
        self.move_to_ready_pose()

    def move_to_ready_pose(
        self,
        minimum_time: float = 5.0,
        hold_time: float = 6.0,
    ) -> None:
        """
        Move the robot to the ready pose using JointPositionCommandBuilder.

        This mirrors lerobot_teleoperator_rby1's _handle_ready_pose() and is
        called automatically at the end of connect() (when
        config.move_to_ready_on_connect is True) so that the robot is in a
        safe, known configuration before inference begins.

        Parameters
        ----------
        minimum_time : float
            Minimum trajectory duration in seconds (default 5.0).
        hold_time : float
            Control hold time in seconds (default 6.0).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError("rby1_sdk is required.") from e

        logger.info("Moving to ready pose …")
        try:
            ctrl_state = self._robot.get_control_manager_state().control_state
            if ctrl_state != rby.ControlManagerState.ControlState.Idle:
                self._robot.cancel_control()
                time.sleep(1.0)

            ready = self._robot.wait_for_control_ready(1000)
            if not ready:
                logger.error(
                    "wait_for_control_ready timed out — skipping ready pose motion."
                )
                return

            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointPositionCommandBuilder()
                    .set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
                    )
                    .set_position(_READY_POSE)
                    .set_minimum_time(minimum_time)
                )
                .set_head_command(
                    rby.JointPositionCommandBuilder()
                    .set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
                    )
                    .set_position(_READY_HEAD)
                    .set_minimum_time(minimum_time)
                )
            )
            self._robot.send_command(
                rby.RobotCommandBuilder().set_command(cbc)
            ).get()
            logger.info("Ready pose reached.")

            # Re-create command stream so inference can start immediately.
            self._stream = self._robot.create_command_stream(priority=1)
        except Exception as exc:
            logger.error(f"move_to_ready_pose error: {exc}")

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # Apply a low-pass filter to joint position commands to reduce jitter.
        self._robot.set_parameter("joint_position_command.cutoff_frequency", "5")
        logger.info("RB-Y1 configured: cutoff_frequency=5 Hz.")

    # ------------------------------------------------------------------ #
    #  Observation                                                          #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        state = self._robot.get_state()
        model = self._model

        # ── Joint positions ────────────────────────────────────────────
        pos = state.position  # full-DOF array (rad)
        torso_pos = pos[model.torso_idx]            # (6,)
        right_pos = pos[model.right_arm_idx]        # (7,)
        left_pos = pos[model.left_arm_idx]          # (7,)

        obs: dict[str, Any] = {}
        if self._config.use_torso:
            for i, name in enumerate(_TORSO_NAMES):
                obs[name] = float(torso_pos[i])
        if self._config.use_right_arm:
            for i, name in enumerate(_RIGHT_ARM_NAMES):
                obs[name] = float(right_pos[i])
        if self._config.use_left_arm:
            for i, name in enumerate(_LEFT_ARM_NAMES):
                obs[name] = float(left_pos[i])

        # ── Optional: velocities ───────────────────────────────────────
        if self._config.use_velocity:
            vel = state.velocity
            torso_vel = vel[model.torso_idx]
            right_vel = vel[model.right_arm_idx]
            left_vel = vel[model.left_arm_idx]
            if self._config.use_torso:
                for i, name in enumerate(_TORSO_NAMES):
                    obs[f"{name}.vel"] = float(torso_vel[i])
            if self._config.use_right_arm:
                for i, name in enumerate(_RIGHT_ARM_NAMES):
                    obs[f"{name}.vel"] = float(right_vel[i])
            if self._config.use_left_arm:
                for i, name in enumerate(_LEFT_ARM_NAMES):
                    obs[f"{name}.vel"] = float(left_vel[i])

        # ── Optional: torques ──────────────────────────────────────────
        if self._config.use_torque:
            torque = state.torque
            torso_trq = torque[model.torso_idx]
            right_trq = torque[model.right_arm_idx]
            left_trq = torque[model.left_arm_idx]
            if self._config.use_torso:
                for i, name in enumerate(_TORSO_NAMES):
                    obs[f"{name}.torque"] = float(torso_trq[i])
            if self._config.use_right_arm:
                for i, name in enumerate(_RIGHT_ARM_NAMES):
                    obs[f"{name}.torque"] = float(right_trq[i])
            if self._config.use_left_arm:
                for i, name in enumerate(_LEFT_ARM_NAMES):
                    obs[f"{name}.torque"] = float(left_trq[i])

        # ── Grippers ───────────────────────────────────────────────────
        # Dataset convention: 1.0 = open, 0.0 = closed
        # (Rby1Gripper internally uses 0=open, 1=closed; flip here for recording)
        if self._config.use_gripper and self._gripper is not None:
            gripper_pos = self._gripper.get_positions()  # [right, left] normalised (0=open, 1=closed)
            if self._config.use_right_arm:
                obs["right_gripper_0"] = 1.0 - float(gripper_pos[0])
            if self._config.use_left_arm:
                obs["left_gripper_0"] = float(gripper_pos[1])

        # ── Cameras ────────────────────────────────────────────────────
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    # ------------------------------------------------------------------ #
    #  Action                                                               #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # In passive_mode no command stream was created; this robot instance
        # is observe-only (a higher-priority teleop controls the hardware).
        if self._config.passive_mode:
            return action

        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError("rby1_sdk is required.") from e

        # ── Startup ramp: compute dynamic minimum_time ─────────────────
        # On the very first send_action() call, record the timestamp so the
        # ramp timer starts from the moment commands are actually sent
        # (after connect() and ready-pose are fully complete).
        now = time.monotonic()
        if self._action_start_time is None:
            self._action_start_time = now
            logger.info(
                f"Startup ramp started: minimum_time will decrease from "
                f"{self._config.startup_min_time:.2f}s → normal over "
                f"{self._config.startup_ramp_duration:.1f}s."
            )

        _normal_min_time = 0.08 if self._config.use_impedance else 0.1
        if self._config.startup_ramp_duration > 0.0:
            elapsed = now - self._action_start_time
            ramp_progress = min(elapsed / self._config.startup_ramp_duration, 1.0)
            _current_min_time = (
                self._config.startup_min_time
                + ramp_progress * (_normal_min_time - self._config.startup_min_time)
            )
        else:
            ramp_progress = 1.0
            _current_min_time = _normal_min_time

        logger.debug(
            f"[ramp] elapsed={elapsed if self._config.startup_ramp_duration > 0.0 else 0.0:.2f}s  "
            f"progress={ramp_progress:.3f}  min_time={_current_min_time:.3f}s"
        )

        # Stiffness / torque-limit slices (order: [6 torso | 7 right | 7 left])
        stiffness  = self._config.impedance_stiffness
        torque_lim = self._config.impedance_torque_limit

        # ── Build per-limb command builders (mirrors rby1_vr.py) ───────
        def _limb_builder(q, stiff, tlim):
            """Return a limb-level command builder for the given joint array."""
            if self._config.use_impedance:
                return (
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(600.0)
                    )
                    .set_minimum_time(_current_min_time)
                    .set_position(q)
                    .set_stiffness(stiff)
                    .set_torque_limit(tlim)
                )
            else:
                return (
                    rby.JointPositionCommandBuilder()
                    .set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(600.0)
                    )
                    .set_minimum_time(_current_min_time)
                    .set_position(q)
                )

        if self._config.use_impedance:
            logger.debug(
                f"[impedance] stiffness={stiffness}, "
                f"torque_limit={torque_lim}, "
                f"damping={self._config.impedance_damping_ratio}"
            )

        # ── Combine only enabled limbs via BodyComponentBasedCommandBuilder ─
        # Disabled limbs are omitted entirely — the SDK holds them internally.
        ctrl_builder = rby.BodyComponentBasedCommandBuilder()
        if self._config.use_torso:
            torso_q = np.array([action[name] for name in _TORSO_NAMES])
            ctrl_builder.set_torso_command(
                _limb_builder(torso_q, stiffness[:6], torque_lim[:6])
            )
        if self._config.use_right_arm:
            right_q = np.array([action[name] for name in _RIGHT_ARM_NAMES])
            ctrl_builder.set_right_arm_command(
                _limb_builder(right_q, stiffness[6:13], torque_lim[6:13])
            )
        if self._config.use_left_arm:
            left_q = np.array([action[name] for name in _LEFT_ARM_NAMES])
            ctrl_builder.set_left_arm_command(
                _limb_builder(left_q, stiffness[13:20], torque_lim[13:20])
            )

        cmd = (
            rby.RobotCommandBuilder()
            .set_command(
                rby.ComponentBasedCommandBuilder()
                .set_body_command(ctrl_builder)
            )
        )
        self._stream.send_command(cmd)
        # logger.info("cmd stream sent")

        # ── Send gripper commands ───────────────────────────────────────
        # Action dict uses dataset convention (1=open, 0=closed);
        # Rby1Gripper.set_positions() expects hardware convention (0=open, 1=closed).
        if self._config.use_gripper and self._gripper is not None:
            current_gripper = self._gripper.get_positions()
            gripper_target = np.array([
                1.0 - action.get("right_gripper_0", 1.0) if self._config.use_right_arm else current_gripper[0],
                action.get("left_gripper_0", 1.0) if self._config.use_left_arm else current_gripper[1],
            ])
            self._gripper.set_positions(gripper_target)

        return action
