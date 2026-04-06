#!/usr/bin/env python

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

"""
RTC + Async Inference: Run Real-Time Chunking with a remote policy server.

This script combines RTC (prefix-guided action chunking with inference delay
compensation) with the async gRPC-based inference architecture, allowing the
policy to run on a separate (potentially more powerful) machine while the robot
client handles observation capture and action execution locally.

Architecture:
    ┌──────────────────────────────┐        gRPC         ┌──────────────────────────┐
    │       Robot Client           │ ◄──────────────────► │    Policy Server         │
    │                              │                      │                          │
    │  [observation_sender thread] │  SendObservations    │  SendObservations()      │
    │  - captures robot obs        │ ───────────────────► │  - enqueues observation  │
    │  - attaches RTC prefix &     │                      │                          │
    │    inference_delay            │                      │  GetActions()            │
    │                              │  GetActions           │  - runs policy with RTC  │
    │  [action_receiver thread]    │ ◄─────────────────── │  - returns original +    │
    │  - receives action chunks    │                      │    processed actions     │
    │  - ActionQueue.merge()       │                      └──────────────────────────┘
    │                              │
    │  [actor_control thread]      │
    │  - ActionQueue.get()         │
    │  - sends actions to robot    │
    └──────────────────────────────┘

Usage:

    # 1. Start the policy server (on a GPU machine) with RTC enabled:
    python -m lerobot.async_inference.policy_server \
        --host=0.0.0.0 \
        --port=8080 \
        --fps=30 \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --rtc.max_guidance_weight=5.0 \
        --rtc.prefix_attention_schedule=LINEAR

    # 2. Run this client (on the robot machine):
    python examples/rtc/eval_with_real_robot_async.py \
        --robot.type=so100_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --server_address=GPU_MACHINE_IP:8080 \
        --policy_type=pi0 \
        --pretrained_name_or_path=lerobot/pi0-so100-multitask \
        --policy_device=cuda \
        --actions_per_chunk=50 \
        --task="Pick up the object" \
        --rtc.enabled=true \
        --rtc.execution_horizon=10 \
        --duration=120
"""

import logging
import math
import pickle  # nosec
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import draccus
import grpc
import torch

from lerobot.async_inference.configs import RobotClientConfig, get_aggregate_function
from lerobot.async_inference.helpers import (
    FPSTracker,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCConfig
from lerobot.processor.factory import make_default_robot_action_processor
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    so_follower,
    unitree_g1,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

logging.basicConfig(level=logging.INFO)
logger = get_logger("rtc_async_client")


@dataclass
class RTCAsyncClientConfig(RobotClientConfig):
    """Configuration extending RobotClientConfig with RTC and demo parameters."""

    # Demo parameters
    duration: float = field(default=60.0, metadata={"help": "Duration to run the demo (seconds)"})
    interpolation_multiplier: int = field(default=1, metadata={"help": "Control rate multiplier (1=off, 2=2x, 3=3x)"})

    # RTC is inherited from RobotClientConfig.rtc

    def __post_init__(self):
        super().__post_init__()
        if self.rtc.enabled:
            # Validate RTC threshold is large enough
            if self.action_queue_size_to_get_new_actions < self.rtc.execution_horizon:
                raise ValueError(
                    f"action_queue_size_to_get_new_actions ({self.action_queue_size_to_get_new_actions}) "
                    f"must be >= rtc.execution_horizon ({self.rtc.execution_horizon})"
                )


class RTCAsyncClient:
    """Async gRPC client with RTC (Real-Time Chunking) support.

    Uses ActionQueue for RTC-style merge/replace instead of the default
    aggregate_fn blending. Tracks round-trip latency to compute inference_delay
    and sends prev_chunk_left_over to the server for prefix attention.
    """

    def __init__(self, config: RTCAsyncClientConfig):
        self.config = config

        # Robot
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        logger.info(f"Robot connected: {config.robot.type}")

        # gRPC
        self.server_address = config.server_address
        self.channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # Policy config to send to server
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
        )

        # RTC action queue (replaces the standard Queue-based approach)
        self.action_queue = ActionQueue(config.rtc)
        self.action_queue_size_history: list[int] = []

        # Latency tracker for inference_delay calculation
        self.latency_tracker = LatencyTracker(maxlen=100)

        # Robot action processor
        self.robot_action_processor = make_default_robot_action_processor()

        # Synchronization
        self.shutdown_event = threading.Event()
        self.start_barrier = threading.Barrier(3)  # observation_sender, action_receiver, actor_control

        # FPS tracking
        self.fps_tracker = FPSTracker(target_fps=config.fps)

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        """Connect to the policy server and send policy instructions."""
        try:
            start_t = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            logger.info(f"Connected to server in {time.perf_counter() - start_t:.4f}s")

            policy_config_bytes = pickle.dumps(self.policy_config)
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=policy_config_bytes)
            )
            logger.info(
                f"Sent policy instructions: type={self.policy_config.policy_type}, "
                f"model={self.policy_config.pretrained_name_or_path}"
            )
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as e:
            logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Shutdown the client."""
        self.shutdown_event.set()
        self.robot.disconnect()
        self.channel.close()
        logger.info("Client stopped")

    # ─── Observation Sender Thread ──────────────────────────────────────────

    def observation_sender(self):
        """Thread: captures observations and sends them to the policy server.

        Attaches RTC metadata (prev_chunk_left_over, inference_delay) to each
        observation so the server can pass them to predict_action_chunk().
        Sends a new observation when the action queue falls below the threshold.
        """
        self.start_barrier.wait()
        logger.info("[OBS_SENDER] Starting observation sender thread")

        fps = self.config.fps
        time_per_chunk = 1.0 / fps
        get_actions_threshold = self.config.action_queue_size_to_get_new_actions if self.config.rtc.enabled else 0
        first_obs = True

        try:
            while self.running:
                should_send = (
                    first_obs
                    or self.action_queue.qsize() <= get_actions_threshold
                )

                if not should_send:
                    time.sleep(0.05)
                    continue

                # Capture observation
                raw_obs = self.robot.get_observation()
                raw_obs["task"] = self.config.task

                # Compute RTC metadata
                prev_chunk_left_over = self.action_queue.get_left_over() if self.config.rtc.enabled else None
                inference_latency = self.latency_tracker.max() or 0.0
                inference_delay = math.ceil(inference_latency / time_per_chunk) if self.config.rtc.enabled else None

                # Build timed observation with RTC fields
                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    timestep=max(self.action_queue.get_action_index(), 0),
                    observation=raw_obs,
                    must_go=first_obs or self.action_queue.empty(),
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                )
                first_obs = False

                # Record send time for round-trip latency measurement
                send_time = time.perf_counter()
                timed_obs._send_perf_time = send_time  # stash for latency calc

                # Serialize and send
                obs_bytes = pickle.dumps(timed_obs)
                obs_iterator = send_bytes_in_chunks(
                    obs_bytes,
                    services_pb2.Observation,
                    log_prefix="[OBS_SENDER]",
                    silent=True,
                )
                try:
                    self.stub.SendObservations(obs_iterator)
                    logger.debug(
                        f"[OBS_SENDER] Sent obs #{timed_obs.get_timestep()} "
                        f"(delay={inference_delay}, prefix={'yes' if prev_chunk_left_over is not None else 'no'})"
                    )
                except grpc.RpcError as e:
                    logger.error(f"[OBS_SENDER] gRPC error sending observation: {e}")

        except Exception as e:
            logger.error(f"[OBS_SENDER] Fatal: {e}\n{traceback.format_exc()}")
            sys.exit(1)

        logger.info("[OBS_SENDER] Thread shutting down")

    # ─── Action Receiver Thread ─────────────────────────────────────────────

    def action_receiver(self):
        """Thread: receives action chunks from the server and merges into ActionQueue.

        In RTC mode, ActionQueue.merge() replaces the queue (accounting for
        inference delay). Original actions are extracted from TimedAction for
        computing leftovers on the next inference round.
        """
        self.start_barrier.wait()
        logger.info("[ACTION_RECV] Starting action receiver thread")

        fps = self.config.fps
        time_per_chunk = 1.0 / fps

        try:
            while self.running:
                try:
                    actions_response = self.stub.GetActions(services_pb2.Empty())
                    if len(actions_response.data) == 0:
                        continue  # Server returned empty (timeout)

                    receive_time = time.perf_counter()
                    timed_actions: list[TimedAction] = pickle.loads(actions_response.data)  # nosec

                    if not timed_actions:
                        continue

                    # Move to client device
                    client_device = self.config.client_device
                    for ta in timed_actions:
                        if ta.action.device.type != client_device:
                            ta.action = ta.action.to(client_device)
                        if ta.original_action is not None and ta.original_action.device.type != client_device:
                            ta.original_action = ta.original_action.to(client_device)

                    # Extract original and processed action tensors
                    processed_actions = torch.stack([ta.action for ta in timed_actions])
                    if timed_actions[0].original_action is not None:
                        original_actions = torch.stack([ta.original_action for ta in timed_actions])
                    else:
                        original_actions = processed_actions.clone()

                    # Compute real delay from round-trip latency
                    # Use the timestamp difference as a proxy for network + inference latency
                    first_ts = timed_actions[0].get_timestamp()
                    round_trip_latency = time.time() - first_ts
                    self.latency_tracker.add(round_trip_latency)
                    real_delay = math.ceil(round_trip_latency / time_per_chunk)

                    action_index_before = self.action_queue.get_action_index()

                    # Merge into ActionQueue (RTC: replace; non-RTC: append)
                    self.action_queue.merge(
                        original_actions,
                        processed_actions,
                        real_delay,
                        action_index_before,
                    )

                    logger.info(
                        f"[ACTION_RECV] Merged {len(timed_actions)} actions, "
                        f"delay={real_delay}, queue_size={self.action_queue.qsize()}"
                    )

                except grpc.RpcError as e:
                    logger.error(f"[ACTION_RECV] gRPC error: {e}")

        except Exception as e:
            logger.error(f"[ACTION_RECV] Fatal: {e}\n{traceback.format_exc()}")
            sys.exit(1)

        logger.info("[ACTION_RECV] Thread shutting down")

    # ─── Actor Control Thread ───────────────────────────────────────────────

    def actor_control(self):
        """Thread: consumes actions from ActionQueue and sends to the robot.

        Runs at a fixed frequency (cfg.fps). Uses ActionQueue.get() which
        returns None if the queue is empty (non-blocking).
        """
        self.start_barrier.wait()
        logger.info("[ACTOR] Starting actor control thread")

        action_keys = [k for k in self.robot.action_features if k.endswith(".pos")]

        action_count = 0
        interpolator = ActionInterpolator(multiplier=self.config.interpolation_multiplier)
        action_interval = interpolator.get_control_interval(self.config.fps)

        try:
            while self.running:
                start_time = time.perf_counter()

                if interpolator.needs_new_action():
                    new_action = self.action_queue.get()
                    if new_action is not None:
                        self.action_queue_size_history.append(self.action_queue.qsize())
                        interpolator.add(new_action.cpu())

                action_tensor = interpolator.get()
                if action_tensor is not None:
                    action_tensor = action_tensor.cpu()
                    action_dict = {
                        key: action_tensor[i].item()
                        for i, key in enumerate(action_keys)
                    }
                    action_processed = self.robot_action_processor((action_dict, None))
                    self.robot.send_action(action_processed)
                    action_count += 1

                dt_s = time.perf_counter() - start_time
                time.sleep(max(0, (action_interval - dt_s) - 0.001))

            logger.info(f"[ACTOR] Shutting down. Total actions: {action_count}")

        except Exception as e:
            logger.error(f"[ACTOR] Fatal: {e}\n{traceback.format_exc()}")
            sys.exit(1)

    # ─── Main Run Loop ──────────────────────────────────────────────────────

    def run(self):
        """Start all threads and run until duration elapsed or shutdown."""
        if not self.start():
            return

        obs_thread = threading.Thread(
            target=self.observation_sender, daemon=True, name="ObsSender"
        )
        recv_thread = threading.Thread(
            target=self.action_receiver, daemon=True, name="ActionRecv"
        )
        actor_thread = threading.Thread(
            target=self.actor_control, daemon=True, name="ActorCtrl"
        )

        obs_thread.start()
        recv_thread.start()
        actor_thread.start()

        logger.info(f"Running for {self.config.duration}s...")
        start_time = time.time()

        try:
            while not self.shutdown_event.is_set():
                elapsed = time.time() - start_time
                if elapsed >= self.config.duration:
                    break

                # Log queue status periodically
                if int(elapsed) % 10 == 0:
                    logger.info(
                        f"[MAIN] Elapsed: {elapsed:.0f}s, "
                        f"Queue: {self.action_queue.qsize()}, "
                        f"Max latency: {self.latency_tracker.max():.3f}s"
                    )

                time.sleep(1.0)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            self.shutdown_event.set()
            obs_thread.join(timeout=2)
            recv_thread.join(timeout=2)
            actor_thread.join(timeout=2)

            if self.config.debug_visualize_queue_size and self.action_queue_size_history:
                visualize_action_queue_size(self.action_queue_size_history)

            self.stop()
            logger.info("All threads stopped. Cleanup complete.")


@draccus.wrap()
def main(cfg: RTCAsyncClientConfig):
    """Entry point for RTC + Async inference client."""
    register_third_party_plugins()

    logger.info(f"Config: server={cfg.server_address}, policy={cfg.policy_type}, rtc={cfg.rtc.enabled}")

    client = RTCAsyncClient(cfg)
    client.run()


if __name__ == "__main__":
    main()
