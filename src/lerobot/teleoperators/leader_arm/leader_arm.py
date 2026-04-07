"""
rby1_leader_arm.py – LeRobot Teleoperator for Rainbow Robotics RB-Y1 via
physical leader arm (master arm).

Control flow
============
1. connect() initialises the leader arm hardware (Dynamixel servos via
   rby1_sdk.upc.MasterArm) and starts a 100 Hz background control
   callback.
2. The callback handles:
   - Smooth cosine init trajectory (first ``init_duration`` seconds)
   - Gravity compensation + joint-limit barriers + viscous damping
   - Button-based mode switching (pressed → free, released → hold)
   - Trigger reading for gripper control
3. get_action() reads the latest joint positions and trigger values from
   the background thread and returns them as a flat action dict.

Convention compliance
=====================
Unlike the VR teleoperator (rby1_vr.py) which directly sends Cartesian
impedance commands to the robot, this teleoperator strictly follows the
LeRobot convention: get_action() returns joint positions only, and the
robot's send_action() is responsible for sending commands to the hardware.

Action keys (match lerobot_robot_rby1 Rby1.action_features):
    right_arm_0 … right_arm_6          (rad)
    left_arm_0  … left_arm_6           (rad)
    right_gripper_0                     (0.0=open, 1.0=closed)
    left_gripper_0                      (0.0=open, 1.0=closed)

Button mapping (leader arm hardware buttons)
=============================================
    Button (per hand) → pressed: arm free (gravity comp only)
                        released: hold current position
    Trigger (per hand) → 0-1000 analog → normalised gripper command
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_leader_arm import Rby1LeaderArmConfig

logger = logging.getLogger(__name__)

# Joint names — must match lerobot_robot_rby1
_TORSO_NAMES = [f"torso_{i}" for i in range(6)]
_RIGHT_ARM_NAMES = [f"right_arm_{i}" for i in range(7)]
_LEFT_ARM_NAMES = [f"left_arm_{i}" for i in range(7)]

# Physical joint limits from rby1m URDF (rad).
# Order: [torso_0..5 | right_arm_0..6 | left_arm_0..6]
_TORSO_Q_MIN   = np.array([-0.261799388, -0.523598776, -2.617993878, -0.785398163, -0.523598776, -2.35619449])
_TORSO_Q_MAX   = np.array([ 0.261799388,  1.570796327,  1.570796327,  1.570796327,  0.523598776,  2.35619449])
_RIGHT_ARM_Q_MIN = np.array([-3.141592654, -3.141592654, -3.141592654, -2.617993878, -3.141592654, -1.570796327, -2.705260340])
_RIGHT_ARM_Q_MAX = np.array([ 3.141592654,  0.017453293,  3.141592654,  0.017453293,  3.141592654,  1.919862177,  2.705260340])
_LEFT_ARM_Q_MIN  = np.array([-3.141592654, -0.017453293, -3.141592654, -2.617993878, -3.141592654, -1.570796327, -2.705260340])
_LEFT_ARM_Q_MAX  = np.array([ 3.141592654,  3.141592654,  3.141592654,  0.017453293,  3.141592654,  1.919862177,  2.705260340])


class Rby1LeaderArm(Teleoperator):
    """
    LeRobot Teleoperator that reads joint positions from the RB-Y1 leader
    arm and returns them as actions.  Does NOT communicate with the robot.
    """

    config_class = Rby1LeaderArmConfig
    name = "rby1_leader_arm"

    def __init__(self, config: Rby1LeaderArmConfig) -> None:
        super().__init__(config)
        self._config = config
        self._is_connected = False

        # rby1_sdk objects (set during connect)
        self._leader_arm = None

        # Thread-safe shared state written by the 100 Hz callback,
        # read by get_action().
        self._lock = threading.Lock()
        self._right_q = np.zeros(7)
        self._left_q = np.zeros(7)
        self._right_trigger: float = 0.0
        self._left_trigger: float = 0.0
        self._init_done = False
        self._reset_requested = False

    # ------------------------------------------------------------------ #
    #  Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True  # Leader arm uses absolute encoders.

    @property
    def action_features(self) -> dict[str, type]:
        names: list[str] = []
        if self._config.use_torso:
            names += _TORSO_NAMES
        if self._config.use_right_arm:
            names += _RIGHT_ARM_NAMES
        if self._config.use_left_arm:
            names += _LEFT_ARM_NAMES
        features: dict[str, type] = {n: float for n in names}
        if self._config.use_gripper:
            if self._config.use_right_arm:
                features["right_gripper_0"] = float
            if self._config.use_left_arm:
                features["left_gripper_0"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ------------------------------------------------------------------ #
    #  Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError(
                "rby1_sdk is required for the RBY1 leader arm teleoperator. "
                "Install it from the RB-Y1 SDK."
            ) from e

        cfg = self._config

        # ── 1. Resolve URDF model path ────────────────────────────────
        if cfg.leader_arm_model_path is not None:
            model_path = cfg.leader_arm_model_path
        else:
            import importlib.resources
            # Default: look relative to rby1_sdk package installation
            sdk_path = str(importlib.resources.files("rby1_sdk"))
            model_path = f"{sdk_path}/../models/master_arm/model.urdf"
            # Fallback: common known location on UPC
            import os
            if not os.path.isfile(model_path):
                model_path = "/home/nvidia/rby1-sdk/models/master_arm/model.urdf"
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    "Could not find leader arm URDF model. "
                    "Set leader_arm_model_path in Rby1LeaderArmConfig."
                )

        # ── 2. Initialise leader arm hardware ─────────────────────────
        logger.info("Initialising leader arm device …")
        rby.upc.initialize_device(rby.upc.MasterArmDeviceName)
        self._leader_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
        self._leader_arm.set_model_path(model_path)
        self._leader_arm.set_control_period(1.0 / cfg.control_frequency)

        active_ids = self._leader_arm.initialize(verbose=False)
        if len(active_ids) != rby.upc.MasterArm.DeviceCount:
            raise RuntimeError(
                f"Leader arm device count mismatch: expected "
                f"{rby.upc.MasterArm.DeviceCount}, got {len(active_ids)} "
                f"(active IDs: {active_ids})"
            )

        # ── 3. Prepare config arrays (convert degrees → radians) ──────
        right_init_q = np.deg2rad(cfg.right_init_q_deg)
        left_init_q = np.deg2rad(cfg.left_init_q_deg)
        min_q = np.deg2rad(cfg.min_q_deg)
        max_q = np.deg2rad(cfg.max_q_deg)
        torque_limit = np.array(cfg.torque_limit)
        viscous_gain = np.array(cfg.viscous_gain)
        barrier = cfg.joint_limit_barrier
        gravity_scale = cfg.gravity_comp_scale
        init_duration = cfg.init_duration
        loop_period = 1.0 / cfg.control_frequency

        # ── 4. Shared state for cross-thread communication ────────────
        #   Written by the callback, read by get_action().
        lock = self._lock
        self._right_q = right_init_q.copy()
        self._left_q = left_init_q.copy()
        self._right_trigger = 0.0
        self._left_trigger = 0.0
        self._init_done = False

        # Mutable containers captured by the closure below.
        init_start_time: list[float | None] = [None]
        init_right_q_start: list[np.ndarray | None] = [None]
        init_left_q_start: list[np.ndarray | None] = [None]
        right_q_hold: list[np.ndarray | None] = [None]
        left_q_hold: list[np.ndarray | None] = [None]

        # Reference to self for the closure
        teleop = self

        # ── 5. Define the 100 Hz control callback ─────────────────────
        def _control_callback(state: rby.upc.MasterArm.State):
            ma_input = rby.upc.MasterArm.ControlInput()

            # reset() requests a fresh init trajectory to the configured ready pose.
            with lock:
                reset_requested = teleop._reset_requested
                if reset_requested:
                    teleop._reset_requested = False
            if reset_requested:
                init_start_time[0] = None
                init_right_q_start[0] = None
                init_left_q_start[0] = None
                right_q_hold[0] = None
                left_q_hold[0] = None
                teleop._init_done = False

            # -- A. Init trajectory (cosine interpolation) ---------------
            if init_start_time[0] is None:
                init_start_time[0] = time.time()
                init_right_q_start[0] = state.q_joint[0:7].copy()
                init_left_q_start[0] = state.q_joint[7:14].copy()

            t = time.time() - init_start_time[0]

            if t < init_duration:
                s = 0.5 * (1.0 - np.cos(np.pi * t / init_duration))
                right_q_cmd = init_right_q_start[0] + s * (right_init_q - init_right_q_start[0])
                left_q_cmd = init_left_q_start[0] + s * (left_init_q - init_left_q_start[0])

                ma_input.target_operating_mode.fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                ma_input.target_position[0:7] = right_q_cmd
                ma_input.target_position[7:14] = left_q_cmd
                ma_input.target_torque[:] = torque_limit

                with lock:
                    teleop._right_q = right_q_cmd.copy()
                    teleop._left_q = left_q_cmd.copy()
                    teleop._right_trigger = float(state.button_right.trigger)
                    teleop._left_trigger = float(state.button_left.trigger)
                return ma_input

            # First call after init completes
            if not teleop._init_done:
                # Latch the configured init targets, not the measured master-arm pose.
                # This prevents sag from weak gravity compensation from immediately
                # propagating to the robot after reset completes.
                right_q_hold[0] = right_init_q.copy()
                left_q_hold[0] = left_init_q.copy()
                teleop._init_done = True

            # -- B. Read triggers ----------------------------------------
            with lock:
                teleop._right_trigger = float(state.button_right.trigger)
                teleop._left_trigger = float(state.button_left.trigger)

            # -- C. Gravity comp + barrier + viscous damping torque ------
            torque = (
                state.gravity_term
                + barrier * (
                    np.maximum(min_q - state.q_joint, 0)
                    + np.minimum(max_q - state.q_joint, 0)
                )
                + viscous_gain * state.qvel_joint
            )
            torque = np.clip(torque, -torque_limit, torque_limit)

            # -- D. Right arm: button → free / hold ----------------------
            if state.button_right.button == 1:
                # Free mode: gravity comp only
                ma_input.target_operating_mode[0:7].fill(
                    rby.DynamixelBus.CurrentControlMode
                )
                ma_input.target_torque[0:7] = torque[0:7] * gravity_scale
                right_q_hold[0] = state.q_joint[0:7].copy()
            else:
                # Hold mode: keep last position
                ma_input.target_operating_mode[0:7].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                ma_input.target_torque[0:7] = torque_limit[0:7]
                ma_input.target_position[0:7] = right_q_hold[0]

            # -- E. Left arm: button → free / hold -----------------------
            if state.button_left.button == 1:
                ma_input.target_operating_mode[7:14].fill(
                    rby.DynamixelBus.CurrentControlMode
                )
                ma_input.target_torque[7:14] = torque[7:14] * gravity_scale
                left_q_hold[0] = state.q_joint[7:14].copy()
            else:
                ma_input.target_operating_mode[7:14].fill(
                    rby.DynamixelBus.CurrentBasedPositionControlMode
                )
                ma_input.target_torque[7:14] = torque_limit[7:14]
                ma_input.target_position[7:14] = left_q_hold[0]

            # -- F. Publish to shared state ------------------------------
            with lock:
                teleop._right_q = right_q_hold[0].copy()
                teleop._left_q = left_q_hold[0].copy()

            return ma_input

        # ── 6. Start control loop ─────────────────────────────────────
        self._leader_arm.start_control(_control_callback)
        self._is_connected = True
        logger.info(
            f"{self} control loop started. Waiting {cfg.startup_wait_time:.1f} s "
            "for leader arm to reach ready pose …"
        )
        time.sleep(cfg.startup_wait_time)
        logger.info(f"{self} connected. Leader arm control loop running at {cfg.control_frequency} Hz.")

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        if self._leader_arm is not None:
            try:
                self._leader_arm.stop_control()
            except Exception as exc:
                logger.warning(f"Error stopping leader arm control: {exc}")
            self._leader_arm = None

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ #
    #  Calibration / Configuration                                          #
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        pass  # Absolute encoders — no calibration needed.

    def reset(self) -> None:
        """Re-run init trajectory so each episode starts from the ready pose."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info(f"{self} reset requested: moving leader arm to ready pose.")
        with self._lock:
            self._reset_requested = True
            self._init_done = False

        timeout_s = max(float(self._config.init_duration) + 2.0, 2.0)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._init_done and not self._reset_requested:
                    logger.info(f"{self} reset complete.")
                    return
            time.sleep(0.05)

        logger.warning(
            f"{self} reset timed out after {timeout_s:.1f}s; continuing recording loop."
        )

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  get_action                                                           #
    # ------------------------------------------------------------------ #

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        cfg = self._config

        with self._lock:
            right_q = self._right_q.copy()
            left_q = self._left_q.copy()
            right_trigger = self._right_trigger
            left_trigger = self._left_trigger

        action: dict[str, Any] = {}

        if cfg.use_torso:
            # Leader arm has no torso — return zeros as placeholder.
            for name in _TORSO_NAMES:
                action[name] = 0.0

        if cfg.use_right_arm:
            right_q_out = right_q.copy()
            right_q_out[6] += np.deg2rad(cfg.right_wrist_offset_deg)
            right_q_out = np.clip(right_q_out, _RIGHT_ARM_Q_MIN, _RIGHT_ARM_Q_MAX)
            for i, name in enumerate(_RIGHT_ARM_NAMES):
                action[name] = float(right_q_out[i])

        if cfg.use_left_arm:
            left_q_out = left_q.copy()
            left_q_out[6] += np.deg2rad(cfg.left_wrist_offset_deg)
            left_q_out = np.clip(left_q_out, _LEFT_ARM_Q_MIN, _LEFT_ARM_Q_MAX)
            for i, name in enumerate(_LEFT_ARM_NAMES):
                action[name] = float(left_q_out[i])

        if cfg.use_gripper:
            tmax = cfg.gripper_trigger_max
            if cfg.use_right_arm:
                action["right_gripper_0"] = float(np.clip(right_trigger / tmax, 0.0, 1.0))
            if cfg.use_left_arm:
                action["left_gripper_0"] = float(np.clip(left_trigger / tmax, 0.0, 1.0))

        return action

    # ------------------------------------------------------------------ #
    #  Feedback                                                             #
    # ------------------------------------------------------------------ #

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # No force feedback implemented.
