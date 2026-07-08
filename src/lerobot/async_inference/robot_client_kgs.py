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

"""KGS Pi0.5 async robot client with terminal pause/resume controls.

This module intentionally leaves ``robot_client.py`` unchanged. It reuses the
base ``RobotClient`` implementation and isolates RB-Y1/Pi0.5 operator controls
behind a separate entrypoint:

    python -m lerobot.async_inference.robot_client_kgs ...
"""

from __future__ import annotations

import logging
import select
import sys
import termios
import threading
import time
import tty
from contextlib import suppress
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus

from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    Observation,
    RawObservation,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)
from .robot_client import RobotClient

SUPPORTED_KGS_BACKENDS = {"pi05_zmq", "pi05_thor"}


def validate_kgs_backend(backend: str) -> None:
    if backend not in SUPPORTED_KGS_BACKENDS:
        raise ValueError(
            "robot_client_kgs only supports remote Pi0.5 backends "
            f"{sorted(SUPPORTED_KGS_BACKENDS)}, got {backend!r}"
        )


class KgsPi05RobotClient(RobotClient):
    """Pi0.5-only client that can pause, ready-pose, and resume safely."""

    prefix = "robot_client_kgs"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        validate_kgs_backend(getattr(config, "backend", "grpc"))
        super().__init__(config)
        self._init_kgs_pause_state()

    def _init_kgs_pause_state(self) -> None:
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

    def _clear_action_queue(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()

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
                generation == self.pause_generation
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
                self.resume_requested = False

                if not self.ready_pose_in_progress:
                    self.ready_pose_in_progress = True
                    start_worker = True

            self._clear_action_queue()
            drained = self._drain_remote_observation_queue()
            self.must_go.clear()

        self.logger.info(
            "Paused policy control; cleared queued actions and drained %s remote observations", drained
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
                return

        self._resume_now()

    def _resume_now(self) -> None:
        with self.policy_control_lock:
            self._clear_action_queue()
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

    def _start_ready_pose_worker(self) -> None:
        thread = threading.Thread(
            target=self._ready_pose_worker,
            daemon=True,
            name="KgsReadyPoseWorker",
        )
        with self.pause_state_lock:
            self._ready_pose_thread = thread
        thread.start()

    def _ready_pose_worker(self) -> None:
        ready_succeeded = False
        try:
            move_to_ready_pose = getattr(self.robot, "move_to_ready_pose", None)
            if callable(move_to_ready_pose):
                self.logger.info("Moving robot to ready pose")
                move_to_ready_pose()
                ready_succeeded = True
            else:
                self.logger.warning("Robot does not expose move_to_ready_pose(); staying paused")
        except Exception as exc:
            self.logger.exception("Ready-pose motion failed; staying paused until explicit resume: %s", exc)
        finally:
            self._complete_ready_pose_transition(ready_succeeded)

    def _complete_ready_pose_transition(self, ready_succeeded: bool = True) -> None:
        with self.pause_state_lock:
            self.ready_pose_in_progress = False
            self._ready_pose_thread = None
            should_resume = self.resume_requested and ready_succeeded and not self.shutdown_event.is_set()
            if not should_resume:
                self.resume_requested = False

        if should_resume:
            self._resume_now()
        elif ready_succeeded:
            self.logger.info("Ready-pose motion complete; client remains paused")
        else:
            self.logger.warning("Ready-pose motion did not complete; client remains paused")

    def start_keyboard_listener(self) -> bool:
        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            return True

        if not sys.stdin.isatty():
            self.logger.warning("stdin is not a TTY; KGS pause/resume keyboard controls are disabled")
            return False

        self._keyboard_stop_event.clear()
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop,
            daemon=True,
            name="KgsKeyboardListener",
        )
        self._keyboard_thread.start()
        self.logger.info("KGS keyboard controls enabled: 's'=pause/ready, 'f'=resume")
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
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue

                key = sys.stdin.read(1).lower()
                if key == "s":
                    self.pause_and_ready()
                elif key == "f":
                    self.resume_from_pause()

        except Exception as exc:
            self.logger.warning("KGS keyboard listener stopped: %s", exc)
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
    ) -> None:
        payload = (generation, observation, remote_observation)
        try:
            self.remote_observation_queue.put_nowait(payload)
            return
        except Full:
            pass

        with suppress(Empty):
            self.remote_observation_queue.get_nowait()
        self.remote_observation_queue.put_nowait(payload)

    def control_loop_remote_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        if not self.uses_remote_zmq_backend:
            raise RuntimeError("control_loop_remote_observation is only valid for remote ZMQ backends")

        if self._is_control_blocked():
            return None

        try:
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

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

            self._put_latest_remote_observation(generation, observation, remote_observation)

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
            return None

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        """Pop and send one action unless policy control is paused."""
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

            action = self._action_tensor_to_action_dict(timed_action.get_action())

            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()

            performed_action = self.robot.send_action(action)

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
    ) -> bool:
        request_start = time.perf_counter()
        action_dict = self.remote_client.get_action(remote_observation)
        request_time = time.perf_counter() - request_start

        if not self._can_accept_remote_response(generation):
            self.logger.info("Discarding stale %s action response for generation %s", self.remote_backend_name, generation)
            return False

        timed_actions = self._convert_remote_actions(action_dict, observation)

        with self.policy_control_lock:
            if not self._can_accept_remote_response(generation):
                self.logger.info("Discarding %s action response after conversion because state changed", self.remote_backend_name)
                return False

            self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

            if verbose and timed_actions:
                old_size, old_timesteps = self._inspect_action_queue()
                if not old_timesteps:
                    with self.latest_action_lock:
                        old_timesteps = [self.latest_action]
            else:
                old_size, old_timesteps = 0, []

            queue_update_start = time.perf_counter()
            self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
            queue_update_time = time.perf_counter() - queue_update_start

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
        if not self.uses_remote_zmq_backend:
            self.logger.debug("receive_remote_actions called for non-ZMQ backend; skipping")
            return
        if self.remote_client is None:
            raise RuntimeError(
                f"{self.remote_backend_name} client not started. "
                "Run RobotClient.start() before requesting actions."
            )

        self.start_barrier.wait()
        self.logger.info("%s action receiving thread starting", self.remote_backend_name)

        while self.running:
            try:
                generation, observation, remote_observation = self.remote_observation_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._receive_and_enqueue_remote_actions(
                    generation,
                    observation,
                    remote_observation,
                    verbose=verbose,
                )
            except Exception as exc:
                self.logger.error("Error in %s action receiving loop: %s", self.remote_backend_name, exc)

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        if self.uses_remote_zmq_backend:
            self.start_barrier.wait()

        self.logger.info("Control loop thread starting")

        performed_action = None
        captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            if not self._is_control_blocked() and self.actions_available():
                performed_action = self.control_loop_action(verbose)

            if not self._is_control_blocked() and self._ready_to_send_observation():
                captured_observation = self.control_loop_remote_observation(task, verbose)

            self.logger.debug("Control loop (ms): %.2f", (time.perf_counter() - control_loop_start) * 1000)
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return captured_observation, performed_action

    def stop(self):
        self.shutdown_event.set()
        self._keyboard_stop_event.set()

        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=0.5)
        self._restore_terminal()

        with self.pause_state_lock:
            ready_pose_thread = self._ready_pose_thread

        if ready_pose_thread is not None and ready_pose_thread.is_alive():
            ready_pose_thread.join(timeout=0.5)
            if ready_pose_thread.is_alive():
                self.logger.warning("Ready-pose worker is still running; continuing shutdown")

        super().stop()


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = KgsPi05RobotClient(cfg)

    if client.start():
        action_receiver_thread = threading.Thread(target=client.receive_remote_actions, daemon=True)
        action_receiver_thread.start()
        client.start_keyboard_listener()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join(timeout=1.0)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
