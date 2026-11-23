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

"""Demo Pi0.5 async robot client with terminal pause/resume controls.

This module intentionally leaves ``robot_client.py`` unchanged. It reuses the
base ``RobotClient`` implementation and isolates RB-Y1/Pi0.5 operator controls
behind a separate entrypoint:

    python -m lerobot.async_inference.robot_client_demo ...
"""

import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus
import numpy as np

from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    Observation,
    RawObservation,
    TimedAction,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)
from .infer_recorder import DiskLowError, InferenceRecorder, RequestRecord, create_recorder_for_client
from .robot_client import RobotClient
from .policy.pi05_zmq import Pi05ZMQClient

SUPPORTED_DEMO_BACKENDS = {"pi05_zmq", "pi05_thor"}
RECONNECT_DELAY_S = 1.0
# 'w' never commands left_arm_6 past this (URDF limit is +-2.705 rad).
LEFT_WRIST_LIMIT_RAD = 2.6


@dataclass
class DemoRobotClientConfig(RobotClientConfig):
    backend: str = "pi05_zmq"
    # Bound individual requests; the receiver retries indefinitely.
    zmq_timeout_ms: int = 90000
    # Left-right mirrored instruction. Keys 'a'/'b' switch between --task and this one at runtime.
    task_mirrored: str = ""
    # Debug recording (see infer_recorder.py) from startup; key 'r' turns it on/off at runtime.
    debug_record: bool = False
    # Each ON period is a new session folder <dir>/<YYYYmmdd_HHMMSS>/.
    debug_record_dir: str = "~/infer_records"
    # JPEG of the camera images that were sent to the server.
    debug_record_images: bool = True
    debug_record_jpeg_quality: int = 90
    # SDK state stream rate; 0 = off.
    debug_record_state_hz: float = 100.0
    # Free text stored in meta.json, e.g. the checkpoint name.
    debug_record_note: str = ""
    # Below this free space stop images; below half of it stop recording.
    debug_record_min_free_gb: float = 2.0
    # 's' opens the grippers to this value (dataset convention: 1.0 = open, 0.0 = closed).
    # JIMTOF_1022 / JIMTOF_1023v2 actions never open past 0.5; 1.0 is the mechanical limit.
    pause_gripper_open: float = 0.5
    # minimum_time (s) of the 's' ready-pose motion; Rby1.move_to_ready_pose() defaults to 5.0.
    ready_pose_min_time: float = 1.5
    # 'w' turns the left wrist (left_arm_6) clockwise by this many degrees while the policy keeps running.
    # Each press adds a turn commanded over wrist_nudge_s. The turn is held until the measured wrist gets there
    # or wrist_nudge_settle_s after the last press; after that the policy owns the new angle. 's' clears it.
    wrist_nudge_deg: float = 30.0
    wrist_nudge_s: float = 0.3
    wrist_nudge_settle_s: float = 1.5
    # Same as --robot.torso_impedance=true: hold the torso with joint impedance control (joint mode).
    torso_impedance: bool = False


def validate_demo_backend(backend: str) -> None:
    if backend not in SUPPORTED_DEMO_BACKENDS:
        raise ValueError(
            "robot_client_demo only supports remote Pi0.5 backends "
            f"{sorted(SUPPORTED_DEMO_BACKENDS)}, got {backend!r}"
        )


class DemoPi05RobotClient(RobotClient):
    """Pi0.5-only client that can pause, ready-pose, and resume safely."""

    prefix = "robot_client_demo"
    logger = get_logger(prefix)
    recorder: InferenceRecorder | None = None

    def __init__(self, config: RobotClientConfig):
        validate_demo_backend(getattr(config, "backend", "grpc"))
        config = replace(config, backend="pi05_zmq")
        if getattr(config, "torso_impedance", False):
            if not hasattr(config.robot, "torso_impedance"):
                raise ValueError("--torso_impedance needs the robot driver with Rby1Config.torso_impedance")
            config = replace(config, robot=replace(config.robot, torso_impedance=True))
        super().__init__(config)
        self._init_demo_pause_state()
        self.task_options = {"a": ("original", config.task)}
        if getattr(config, "task_mirrored", ""):
            self.task_options["b"] = ("mirrored", config.task_mirrored)
        self.active_task_key = "a"

    def _init_demo_pause_state(self) -> None:
        self.pause_event = threading.Event()
        self.policy_control_lock = threading.Lock()
        self.pause_state_lock = threading.Lock()
        self.pause_generation = 0
        self.ready_pose_in_progress = False
        self.resume_requested = False
        self._ready_pose_thread: threading.Thread | None = None

        self._keyboard_stop_event = threading.Event()
        self._keyboard_thread: threading.Thread | None = None
        self._terminal_lock = threading.Lock()
        self._terminal_fd: int | None = None
        self._terminal_attrs: list[Any] | None = None
        self._request_generation: int | None = None
        self._receiver_done = threading.Event()
        self._receiver_done.set()
        # Held while a recorder is opened or closed; stop() sets the flag first, then waits for it.
        self._record_toggle_lock = threading.Lock()
        self._record_stopped = False
        self._record_toggle_thread: threading.Thread | None = None
        # 'w' turns of left_arm_6. A group starts at a press and collects the presses until the measured wrist
        # gets there or it settles: {"ref": measured rad at the first press, "presses": [(time.time(), delta rad)],
        # "done": time.time() it finished or None}. Cleared on pause.
        self._wrist_nudge_lock = threading.Lock()
        self._wrist_groups: list[dict[str, Any]] = []
        self._wrist_last_l6: float | None = None
        self._left_wrist_idx: int | None = None

    def start(self) -> bool:
        # The receiver owns its ZMQ socket, including connection and cleanup.
        self.shutdown_event.clear()
        self._receiver_done.clear()
        self._record_stopped = False
        if getattr(self.config, "debug_record", False):
            with self._record_toggle_lock:
                if self.recorder is None:
                    self._open_debug_record()
        return True

    def _open_debug_record(self) -> None:
        """Start a new recording session. Caller holds _record_toggle_lock."""
        try:
            recorder = create_recorder_for_client(self)
        except DiskLowError as exc:
            self.logger.error("Debug recording not started: %s", exc)
            return
        except Exception as exc:
            self.logger.exception("Debug recording not started: %s", exc)
            return
        self.recorder = recorder
        self.logger.info("Debug recording ON -> %s", recorder.session_dir)

    def _close_debug_record(self) -> None:
        """Detach and close the current recorder. Caller holds _record_toggle_lock."""
        recorder, self.recorder = self.recorder, None
        if recorder is None:
            return
        # Threads that already hold it finish their hook; a closed recorder ignores late calls.
        stopped_early = not recorder.active  # the disk guard stopped it during the session
        recorder.close()
        self.logger.info(
            "Debug recording OFF (saved %s%s)",
            recorder.session_dir, "; it had already stopped on low disk" if stopped_early else "",
        )

    def _toggle_debug_record(self) -> None:
        with self._record_toggle_lock:
            if self._record_stopped or self.shutdown_event.is_set():
                return
            if self.recorder is None:
                self._open_debug_record()
            else:
                self._close_debug_record()

    def request_debug_record_toggle(self) -> None:
        """Key 'r': toggle on a worker so the keyboard ('s' pause) stays responsive."""
        if self._record_toggle_thread is not None and self._record_toggle_thread.is_alive():
            self.logger.info("Debug recording toggle already in progress; ignoring 'r'")
            return
        self._record_toggle_thread = threading.Thread(
            target=self._toggle_debug_record, daemon=True, name="DebugRecordToggle"
        )
        self._record_toggle_thread.start()

    def _record_event(self, name: str, **fields: Any) -> None:
        recorder = self.recorder
        if recorder is not None:
            recorder.event(name, **fields)

    def _request_cancelled(self) -> bool:
        return self.shutdown_event.is_set() or (
            self._request_generation is not None
            and not self._can_accept_remote_response(self._request_generation)
        )

    def _wait_for_server(self) -> bool:
        attempt = 0
        while self.running:
            if self.remote_client.ping(timeout_ms=1000):
                self.logger.info("Connected to Pi05 server at %s", self.server_address)
                self._record_event("server_connected", attempts=attempt + 1)
                return True
            attempt += 1
            if attempt == 1:
                self._record_event("server_wait")
            if attempt == 1 or attempt % 10 == 0:
                self.logger.warning("Waiting for Pi05 server at %s; retrying (attempt %s)", self.server_address, attempt)
            if self.shutdown_event.wait(RECONNECT_DELAY_S):
                break
        return False

    def _discard_failed_request(self, generation: int) -> None:
        with self.policy_control_lock:
            if not self._can_accept_remote_response(generation):
                return
            cleared = self._clear_action_queue()
            self._drain_remote_observation_queue()
            self.must_go.set()
        self._record_event("queue_cleared", reason="request_failed", generation=generation, cleared=cleared)

    def _clear_action_queue(self) -> int:
        """Drop all queued actions; return how many were dropped."""
        with self.action_queue_lock:
            cleared = self.action_queue.qsize()
            self.action_queue = Queue()
        return cleared

    def _drain_remote_observation_queue(self) -> int:
        drained = 0
        while True:
            try:
                self.remote_observation_queue.get_nowait()
                drained += 1
            except Empty:
                return drained

    def _is_control_blocked(self) -> bool:
        with self.pause_state_lock:
            return self.pause_event.is_set() or self.ready_pose_in_progress

    def _current_pause_generation(self) -> int:
        with self.pause_state_lock:
            return self.pause_generation

    def _can_accept_remote_response(self, generation: int) -> bool:
        with self.pause_state_lock:
            return (
                not self.shutdown_event.is_set()
                and generation == self.pause_generation
                and not self.pause_event.is_set()
                and not self.ready_pose_in_progress
            )

    def pause_and_ready(self) -> None:
        """Pause policy control and start one best-effort ready-pose worker."""
        start_worker = False
        with self.policy_control_lock:
            with self.pause_state_lock:
                self.pause_event.set()
                self.pause_generation += 1
                generation = self.pause_generation
                self.resume_requested = False

                if not self.ready_pose_in_progress:
                    self.ready_pose_in_progress = True
                    start_worker = True

            cleared = self._clear_action_queue()
            drained = self._drain_remote_observation_queue()
            self.must_go.clear()
            with self._wrist_nudge_lock:
                self._wrist_groups.clear()

        self.logger.info(
            "Paused policy control; cleared queued actions and drained %s remote observations", drained
        )
        self._record_event(
            "pause", generation=generation, cleared=cleared, drained=drained, ready_pose=start_worker
        )

        if start_worker:
            self._start_ready_pose_worker()
        else:
            self.logger.info("Ready-pose worker already running; pause request refreshed stale-action generation")

    def resume_from_pause(self) -> None:
        """Resume immediately, or defer resume until ready-pose motion returns."""
        with self.pause_state_lock:
            if self.ready_pose_in_progress:
                self.resume_requested = True
                self.logger.info("Resume requested; deferring until ready-pose motion completes")
                self._record_event("resume_deferred")
                return

        self._resume_now()

    def _resume_now(self) -> None:
        with self.policy_control_lock:
            cleared = self._clear_action_queue()
            drained = self._drain_remote_observation_queue()

            with self.pause_state_lock:
                if self.ready_pose_in_progress:
                    self.resume_requested = True
                    self.logger.info("Resume deferred because ready-pose motion restarted")
                    return

                self.pause_generation += 1
                self.resume_requested = False
                self.must_go.set()
                self.pause_event.clear()
                generation = self.pause_generation

        self.logger.info(
            "Resumed policy control with generation %s after draining %s remote observations",
            generation,
            drained,
        )
        self._record_event("resume", generation=generation, cleared=cleared, drained=drained)

    def _start_ready_pose_worker(self) -> None:
        thread = threading.Thread(
            target=self._ready_pose_worker,
            daemon=True,
            name="DemoReadyPoseWorker",
        )
        with self.pause_state_lock:
            self._ready_pose_thread = thread
        thread.start()

    def _ready_pose_worker(self) -> None:
        ready_succeeded = False
        try:
            self._open_grippers_for_pause()
            move_to_ready_pose = getattr(self.robot, "move_to_ready_pose", None)
            if callable(move_to_ready_pose):
                min_time = float(getattr(self.config, "ready_pose_min_time", 5.0))
                self.logger.info("Moving robot to ready pose (minimum_time %.1f s)", min_time)
                move_to_ready_pose(minimum_time=min_time)
                ready_succeeded = True
            else:
                self.logger.warning("Robot does not expose move_to_ready_pose(); staying paused")
        except Exception as exc:
            self.logger.exception("Ready-pose motion failed; staying paused until explicit resume: %s", exc)
        finally:
            self._complete_ready_pose_transition(ready_succeeded)

    def _open_grippers_for_pause(self) -> bool:
        """Best-effort RB-Y1 gripper open command for the pause/ready transition."""
        gripper = getattr(self.robot, "_gripper", None)
        if gripper is None:
            self.logger.info("Robot gripper handle is unavailable; skipping pause gripper-open command")
            return False

        set_positions = getattr(gripper, "set_positions", None)
        if not callable(set_positions):
            self.logger.warning("Robot gripper handle does not expose set_positions(); skipping gripper-open command")
            return False

        # Rby1Gripper.set_positions() takes [right, left] with 0.0 = open, 1.0 = closed (the reverse of the dataset).
        open_value = float(np.clip(getattr(self.config, "pause_gripper_open", 1.0), 0.0, 1.0))
        try:
            set_positions(np.full(2, 1.0 - open_value, dtype=np.float64))
        except Exception as exc:
            self.logger.exception("Failed to open grippers for pause/ready transition: %s", exc)
            return False

        self.logger.info("Opened RB-Y1 grippers to %.2f (1.0 = fully open) for pause/ready transition", open_value)
        return True

    def _complete_ready_pose_transition(self, ready_succeeded: bool = True) -> None:
        with self.pause_state_lock:
            self.ready_pose_in_progress = False
            self._ready_pose_thread = None
            should_resume = self.resume_requested and ready_succeeded and not self.shutdown_event.is_set()
            if not should_resume:
                self.resume_requested = False

        self._record_event("ready_pose_done", succeeded=ready_succeeded)
        if should_resume:
            self._resume_now()
        elif ready_succeeded:
            self.logger.info("Ready-pose motion complete; client remains paused")
        else:
            self.logger.warning("Ready-pose motion did not complete; client remains paused")

    def request_wrist_nudge(self) -> None:
        """'w': turn the left wrist clockwise (left_arm_6 decreases) while the policy keeps running.

        The policy predicts joint deltas from the observed state, so a chunk computed after the wrist has turned
        already contains the turn. Each queued chunk therefore holds ``action - (turn its observation shows)``
        and the control loop adds the commanded turn; a plain offset would be added again with every new chunk.
        The wrist follows about 0.5 s late, so the part an observation shows is measured from its left_arm_6.
        """
        if self._is_control_blocked():
            self.logger.info("'w' ignored while paused")
            return
        step = -np.deg2rad(float(getattr(self.config, "wrist_nudge_deg", 30.0)))
        now = time.time()
        with self._wrist_nudge_lock:
            self._finish_wrist_groups(now)
            group = self._wrist_groups[-1] if self._wrist_groups and self._wrist_groups[-1]["done"] is None else None
            if group is None:
                if self._wrist_last_l6 is None:
                    self.logger.info("'w' ignored: no observation yet")
                    return
                group = {"ref": self._wrist_last_l6, "presses": [], "done": None}
                self._wrist_groups.append(group)
            group["presses"].append((now, step))
            turn = sum(d for _, d in group["presses"])
        self.logger.info("Turning left wrist %.0f deg (this turn %.0f deg)", np.degrees(step), np.degrees(turn))
        self._record_event("wrist_nudge", delta_deg=float(np.degrees(step)), turn_deg=float(np.degrees(turn)))

    def _finish_wrist_groups(self, now: float) -> None:
        """Close the open 'w' turn once the measured wrist is within 3 deg of it or it has settled (lock held)."""
        if not self._wrist_groups or self._wrist_groups[-1]["done"] is not None:
            return
        group = self._wrist_groups[-1]
        turn = sum(d for _, d in group["presses"])
        last_end = group["presses"][-1][0] + float(getattr(self.config, "wrist_nudge_s", 0.3))
        reached = self._wrist_last_l6 is not None and abs(turn - (self._wrist_last_l6 - group["ref"])) <= np.deg2rad(3.0)
        settled = now >= last_end + float(getattr(self.config, "wrist_nudge_settle_s", 1.5))
        if now >= last_end and (reached or settled):
            group["done"] = now

    def _wrist_turn_sent(self, t: float) -> float:
        """Extra left_arm_6 turn (rad) to send at time.time() ``t``: every press, ramped in over wrist_nudge_s."""
        ramp = max(float(getattr(self.config, "wrist_nudge_s", 0.3)), 1e-3)
        with self._wrist_nudge_lock:
            self._finish_wrist_groups(t)
            return sum(
                d * min(max((t - t0) / ramp, 0.0), 1.0) for group in self._wrist_groups for t0, d in group["presses"]
            )

    def _wrist_turn_seen(self, t_obs: float, l6_obs: float | None) -> float:
        """Part of the commanded turn (rad) that an observation taken at ``t_obs`` with left_arm_6 ``l6_obs`` shows."""
        with self._wrist_nudge_lock:
            seen = 0.0
            for group in self._wrist_groups:
                turn = sum(d for _, d in group["presses"])
                if group["done"] is not None and t_obs >= group["done"]:
                    seen += turn  # the policy owns this turn now
                elif t_obs >= group["presses"][0][0] and l6_obs is not None:
                    seen += min(max((l6_obs - group["ref"]) / turn, 0.0), 1.0) * turn
            return seen

    def _shift_left_wrist(self, timed_action: TimedAction, shift: float, clip: bool = False) -> TimedAction:
        """Copy of ``timed_action`` with ``shift`` (rad) added to left_arm_6, optionally kept inside its limits."""
        if self._left_wrist_idx is None:
            self._left_wrist_idx = list(self.robot.action_features).index("left_arm_6")
        action = timed_action.get_action().clone()
        value = float(action[self._left_wrist_idx]) + shift
        if clip:
            value = float(np.clip(value, -LEFT_WRIST_LIMIT_RAD, LEFT_WRIST_LIMIT_RAD))
        action[self._left_wrist_idx] = value
        return replace(timed_action, action=action)

    @property
    def active_task(self) -> str:
        return self.task_options[self.active_task_key][1]

    def log_active_task(self) -> None:
        label, task = self.task_options[self.active_task_key]
        self.logger.info("Instruction [%s: %s] %r", self.active_task_key, label, task)

    def select_task(self, key: str) -> None:
        if key not in self.task_options:
            self.logger.warning("No instruction for '%s'; pass --task_mirrored to enable it", key)
            return
        self.active_task_key = key
        self.log_active_task()
        self._record_event("task_switch", key=key, label=self.task_options[key][0])

    def start_keyboard_listener(self) -> bool:
        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            return True

        if not sys.stdin.isatty():
            self.logger.warning("stdin is not a TTY; demo pause/resume keyboard controls are disabled")
            return False

        self._keyboard_stop_event.clear()
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop,
            daemon=True,
            name="DemoKeyboardListener",
        )
        self._keyboard_thread.start()
        self.logger.info(
            "Demo keyboard controls enabled: 's'=pause/ready/open-grippers, 'f'=resume, "
            "'a'=original instruction, 'b'=mirrored instruction, 'r'=debug recording on/off (now %s), "
            "'w'=turn left wrist clockwise",
            "on" if self.recorder is not None else "off",
        )
        return True

    def _keyboard_loop(self) -> None:
        fd = sys.stdin.fileno()
        try:
            attrs = termios.tcgetattr(fd)
            with self._terminal_lock:
                self._terminal_fd = fd
                self._terminal_attrs = attrs

            tty.setcbreak(fd)

            while self.running and not self._keyboard_stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue

                # os.read, not buffered sys.stdin: keys that arrive together are all handled now.
                data = os.read(fd, 64)
                if not data:
                    break  # stdin closed
                for key in data.decode(errors="ignore").lower():
                    if key == "s":
                        self.pause_and_ready()
                    elif key == "f":
                        self.resume_from_pause()
                    elif key in ("a", "b"):
                        self.select_task(key)
                    elif key == "r":
                        self.request_debug_record_toggle()
                    elif key == "w":
                        self.request_wrist_nudge()

        except Exception as exc:
            self.logger.warning("Demo keyboard listener stopped: %s", exc)
        finally:
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        with self._terminal_lock:
            fd = self._terminal_fd
            attrs = self._terminal_attrs
            self._terminal_fd = None
            self._terminal_attrs = None

        if fd is None or attrs is None:
            return

        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except Exception as exc:
            self.logger.warning("Failed to restore terminal settings: %s", exc)

    # Drift-control note: the following remote-observation, remote-action, and
    # control-loop overrides intentionally mirror selected logic from
    # robot_client.py. Review them against the parent implementation whenever
    # parent remote ZMQ/control-loop behavior changes.
    def _put_latest_remote_observation(
        self,
        generation: int,
        observation: TimedObservation,
        remote_observation: dict[str, Any],
    ) -> TimedObservation | None:
        """Queue the observation; return the not-yet-sent observation it replaced, if any."""
        payload = (generation, observation, remote_observation)
        try:
            self.remote_observation_queue.put_nowait(payload)
            return None
        except Full:
            pass

        replaced = None
        with suppress(Empty):
            replaced = self.remote_observation_queue.get_nowait()[1]
        self.remote_observation_queue.put_nowait(payload)
        return replaced

    def control_loop_remote_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        if not self.uses_remote_zmq_backend:
            raise RuntimeError("control_loop_remote_observation is only valid for remote ZMQ backends")

        if self._is_control_blocked():
            return None

        try:
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task
            if "left_arm_6" in raw_observation:
                self._wrist_last_l6 = float(raw_observation["left_arm_6"])  # reference for the next 'w'

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            generation = self._current_pause_generation()
            remote_observation = self._build_remote_observation(raw_observation)

            if not self._can_accept_remote_response(generation):
                return None

            replaced = self._put_latest_remote_observation(generation, observation, remote_observation)
            recorder = self.recorder
            if recorder is not None:
                recorder.observation(
                    observation, raw_observation, start_time, obs_capture_time,
                    generation, current_queue_size, replaced, task,
                )

            self.logger.debug(
                "QUEUE SIZE: %s (Must go: %s) | Queued %s observation #%s (generation %s)",
                current_queue_size,
                observation.must_go,
                self.remote_backend_name,
                observation.get_timestep(),
                generation,
            )

            if observation.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(
                    "Obs #%s | Avg FPS: %.2f | Target: %.2f",
                    observation.get_timestep(),
                    fps_metrics["avg_fps"],
                    fps_metrics["target_fps"],
                )
                self.logger.debug(
                    "Ts=%.6f | Capturing observation took %.6fs",
                    observation.get_timestamp(),
                    obs_capture_time,
                )

            return raw_observation

        except Exception as exc:
            self.logger.error("Error in %s observation loop: %s", self.remote_backend_name, exc)
            self._record_event("obs_error", error=repr(exc)[:200])
            return None

    def control_loop_action(
        self, verbose: bool = False, recorder: InferenceRecorder | None = None
    ) -> dict[str, Any] | None:
        """Pop and send one action unless policy control is paused.

        ``recorder`` is the control loop's recorder for this iteration, so the
        action and its tick land in the same session.
        """
        with self.policy_control_lock:
            if self._is_control_blocked():
                return None

            get_start = time.perf_counter()
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                try:
                    timed_action = self.action_queue.get_nowait()
                except Empty:
                    return None
            get_end = time.perf_counter() - get_start

            shift = self._wrist_turn_sent(time.time())
            if shift:
                timed_action = self._shift_left_wrist(timed_action, shift, clip=True)
            action = self._action_tensor_to_action_dict(timed_action.get_action())

            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()

            performed_action = self.robot.send_action(action)

        if recorder is not None:
            # action_queue_size[-1] is the queue size before this pop.
            recorder.action_sent(timed_action, self.action_queue_size[-1])

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                "Ts=%s | Action #%s performed | Queue size: %s",
                timed_action.get_timestamp(),
                timed_action.get_timestep(),
                current_queue_size,
            )
            self.logger.debug(
                "Popping action from queue to perform took %.6fs | Queue size: %s",
                get_end,
                current_queue_size,
            )

        return performed_action

    def _receive_and_enqueue_remote_actions(
        self,
        generation: int,
        observation: TimedObservation,
        remote_observation: dict[str, Any],
        verbose: bool = False,
        record: RequestRecord | None = None,
    ) -> bool:
        request_start = time.perf_counter()
        action_dict = self.remote_client.get_action(remote_observation)
        request_time = time.perf_counter() - request_start
        if record is not None:
            record.received(action_dict)

        if not self._can_accept_remote_response(generation):
            self.logger.info("Discarding stale %s action response for generation %s", self.remote_backend_name, generation)
            if record is not None:
                record.discard("stale_generation")
            return False

        timed_actions = self._convert_remote_actions(action_dict, observation)
        # The observation already shows part of the 'w' turn; queue the chunk without that part.
        seen = self._wrist_turn_seen(observation.get_timestamp(), observation.get_observation().get("left_arm_6"))
        queued_actions = [self._shift_left_wrist(a, -seen) for a in timed_actions] if seen else timed_actions

        with self.policy_control_lock:
            if not self._can_accept_remote_response(generation):
                self.logger.info("Discarding %s action response after conversion because state changed", self.remote_backend_name)
                if record is not None:
                    record.discard("state_changed_after_conversion")
                return False

            self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

            if verbose and timed_actions:
                old_size, old_timesteps = self._inspect_action_queue()
                if not old_timesteps:
                    with self.latest_action_lock:
                        old_timesteps = [self.latest_action]
            else:
                old_size, old_timesteps = 0, []

            if record is not None:
                record.before_aggregate(self.action_queue.qsize(), self.latest_action)
            queue_update_start = time.perf_counter()
            self._aggregate_action_queues(queued_actions, self.config.aggregate_fn)
            queue_update_time = time.perf_counter() - queue_update_start
            if record is not None:
                record.accept(timed_actions, self.action_queue.qsize())

            self.must_go.set()

        self.logger.debug(
            "%s action request for obs #%s took %.2fms",
            self.remote_backend_name,
            observation.get_timestep(),
            request_time * 1000,
        )

        if verbose and timed_actions:
            new_size, new_timesteps = self._inspect_action_queue()
            incoming_timesteps = [a.get_timestep() for a in timed_actions]
            self.logger.info(
                "Received %s action chunk for step #%s | Incoming actions: %s:%s | Request time: %.2fms",
                self.remote_backend_name,
                incoming_timesteps[0],
                incoming_timesteps[0],
                incoming_timesteps[-1],
                request_time * 1000,
            )
            self.logger.debug(
                "Queue update complete (%.6fs) | Before: %s items (%s) | After: %s items (%s)",
                queue_update_time,
                old_size,
                old_timesteps[:1] + old_timesteps[-1:] if old_timesteps else [],
                new_size,
                new_timesteps[:1] + new_timesteps[-1:] if new_timesteps else [],
            )

        return True

    def receive_remote_actions(self, verbose: bool = False):
        try:
            self.remote_client = Pi05ZMQClient(
                self.server_address,
                self.config.zmq_timeout_ms,
                cancelled=self._request_cancelled,
            )
            self.start_barrier.wait()
            connected = False
            while self.running:
                if not connected:
                    if not self._wait_for_server():
                        break
                    connected = True
                    # Recovery must start from a new observation, not one captured
                    # while the server was unavailable.
                    with self.policy_control_lock:
                        self._drain_remote_observation_queue()
                        if not self._is_control_blocked():
                            self.must_go.set()
                try:
                    generation, observation, remote_observation = self.remote_observation_queue.get(timeout=0.1)
                except Empty:
                    continue
                if not self._can_accept_remote_response(generation):
                    continue
                record = None
                # The request is ended on the recorder that began it, even if 'r' swaps it meanwhile.
                recorder = self.recorder
                try:
                    self._request_generation = generation
                    if recorder is not None:
                        record = recorder.begin_request(generation, observation, remote_observation)
                    self._receive_and_enqueue_remote_actions(
                        generation, observation, remote_observation, verbose=verbose, record=record,
                    )
                except InterruptedError:
                    if record is not None:
                        record.fail("interrupted")
                    connected = False
                except Exception as exc:
                    self.logger.warning("Pi05 request failed; reconnecting with a fresh observation: %s", exc)
                    if record is not None:
                        record.fail(exc)
                    self._discard_failed_request(generation)
                    connected = False
                    self.shutdown_event.wait(RECONNECT_DELAY_S)
                finally:
                    self._request_generation = None
                    if record is not None:
                        recorder.end_request(record)
        except Exception:
            self.logger.exception("Pi05 receiver failed")
            self.shutdown_event.set()
            self.start_barrier.abort()
        finally:
            try:
                if self.remote_client is not None:
                    self.remote_client.close()
                    self.remote_client = None
            finally:
                self._receiver_done.set()

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        if self.uses_remote_zmq_backend:
            self.start_barrier.wait()

        self.logger.info("Control loop thread starting")
        # `task` is the initial ('a') instruction; the keyboard can switch it while running.
        self.log_active_task()

        performed_action = None
        captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            recorder = self.recorder

            if not self._is_control_blocked() and self.actions_available():
                performed_action = self.control_loop_action(verbose, recorder)

            if not self._is_control_blocked() and self._ready_to_send_observation():
                captured_observation = self.control_loop_remote_observation(self.active_task, verbose)

            if recorder is not None:
                recorder.tick(control_loop_start, self._is_control_blocked())

            self.logger.debug("Control loop (ms): %.2f", (time.perf_counter() - control_loop_start) * 1000)
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return captured_observation, performed_action

    def stop(self):
        # No 'r' toggle can open a new recorder once _record_stopped is set.
        self._record_stopped = True
        self.shutdown_event.set()
        self.start_barrier.abort()
        self._keyboard_stop_event.set()
        # Polling requests notice shutdown within 100 ms. Only their owner
        # thread closes the socket; never close it during a concurrent recv.
        self._receiver_done.wait()

        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=0.5)
        self._restore_terminal()

        with self.pause_state_lock:
            ready_pose_thread = self._ready_pose_thread

        if ready_pose_thread is not None and ready_pose_thread.is_alive():
            ready_pose_thread.join(timeout=0.5)
            if ready_pose_thread.is_alive():
                self.logger.warning("Ready-pose worker is still running; continuing shutdown")

        # Stop the SDK state stream and flush before the robot disconnects;
        # the robot is disconnected even if closing the recorder fails.
        try:
            with self._record_toggle_lock:
                self._close_debug_record()
        except Exception:
            self.logger.exception("Closing the debug recorder failed")
        finally:
            super().stop()


@draccus.wrap()
def async_client(cfg: DemoRobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = DemoPi05RobotClient(cfg)

    if client.start():
        action_receiver_thread = threading.Thread(target=client.receive_remote_actions, daemon=True)
        try:
            action_receiver_thread.start()
            client.start_keyboard_listener()
            client.control_loop(task=cfg.task)
        finally:
            if action_receiver_thread.ident is None:
                client._receiver_done.set()
            client.stop()
            if action_receiver_thread.ident is not None:
                action_receiver_thread.join(timeout=1.0)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


KgsPi05RobotClient = DemoPi05RobotClient
validate_kgs_backend = validate_demo_backend


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
