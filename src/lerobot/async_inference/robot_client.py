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
Example command:
```shell
python src/lerobot/async_inference/robot_client.py         --robot.type=so100_follower         --robot.port=/dev/tty.usbmodem58760431541         --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}"         --robot.id=black         --task="dummy"         --server_address=127.0.0.1:8080         --policy_type=act         --pretrained_name_or_path=user/model         --policy_device=mps         --client_device=cpu         --actions_per_chunk=50         --chunk_size_threshold=0.5         --aggregate_fn_name=weighted_average         --debug_visualize_queue_size=True
```
"""

import logging
import math
import pickle  # nosec
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (
    RobotConfig,  # noqa: F401
    make_robot_from_config,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .groot_n16_zmq import (
    GR00TZMQClient,
    build_groot_n16_observation,
    groot_n16_action_dict_to_timed_actions,
    validate_groot_robot_compatibility,
)
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    apply_observation_crops,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)
    _MAX_PENDING_OBSERVATION_METRICS = 512

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        self.config = config
        self.backend = config.backend
        self.server_address = config.server_address

        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        self.shutdown_event = threading.Event()

        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)

        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing
        self._pending_observation_metrics_lock = threading.Lock()
        self._pending_observation_metrics: OrderedDict[tuple[int, float], dict[str, float]] = OrderedDict()

        self.policy_config: RemotePolicyConfig | None = None
        self.channel = None
        self.stub = None
        self.remote_client: GR00TZMQClient | None = None

        if self.uses_grpc_backend:
            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
            self.policy_config = RemotePolicyConfig(
                config.policy_type,
                config.pretrained_name_or_path,
                lerobot_features,
                config.actions_per_chunk,
                config.policy_device,
                config.obs_atol,
                client_image_crop_applied=bool(config.image_crop_params),
            )
            self.channel = grpc.insecure_channel(
                self.server_address,
                grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
            )
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            self.logger.info(f"Initializing gRPC client to connect to server at {self.server_address}")
        else:
            validate_groot_robot_compatibility(
                self.robot,
                front_camera_key=self.config.groot_front_camera_key,
                left_wrist_camera_key=self.config.groot_left_wrist_camera_key,
                right_wrist_camera_key=self.config.groot_right_wrist_camera_key,
            )
            self.logger.info(f"Initializing GR00T ZMQ client to connect to server at {self.server_address}")

        self.logger.info("Robot connected and ready")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def uses_grpc_backend(self) -> bool:
        return self.backend == "grpc"

    @property
    def uses_groot_backend(self) -> bool:
        return self.backend == "groot_n16_zmq"

    def start(self):
        """Start the robot client and connect to the remote inference server."""
        try:
            if self.uses_grpc_backend:
                assert self.stub is not None
                assert self.policy_config is not None

                start_time = time.perf_counter()
                self.stub.Ready(services_pb2.Empty())
                end_time = time.perf_counter()
                self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

                policy_config_bytes = pickle.dumps(self.policy_config)
                policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

                self.logger.info("Sending policy instructions to policy server")
                self.logger.debug(
                    f"Policy type: {self.policy_config.policy_type} | "
                    f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                    f"Device: {self.policy_config.device}"
                )
                self.stub.SendPolicyInstructions(policy_setup)
            else:
                self.remote_client = GR00TZMQClient(
                    server_address=self.server_address,
                    timeout_ms=self.config.zmq_timeout_ms,
                )
                if not self.remote_client.ping():
                    self.logger.error("Failed to connect to GR00T inference server")
                    self.remote_client.close()
                    self.remote_client = None
                    return False

            self.shutdown_event.clear()
            return True

        except (grpc.RpcError, ImportError, RuntimeError, TimeoutError, ValueError) as e:
            self.logger.error(f"Failed to connect to remote inference server: {e}")
            return False

    def stop(self):
        """Stop the robot client."""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        if self.channel is not None:
            self.channel.close()
            self.logger.debug("gRPC channel closed")

        if self.remote_client is not None:
            self.remote_client.close()
            self.remote_client = None
            self.logger.debug("GR00T ZMQ client closed")

        self.logger.debug("Client stopped")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.uses_grpc_backend:
            raise RuntimeError("send_observation is only valid for the gRPC backend")
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")
        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        assert self.stub is not None

        send_started_perf = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - send_started_perf
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            rpc_started_perf = time.perf_counter()
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            send_completed_perf = time.perf_counter()
            obs_timestep = obs.get_timestep()
            self._register_pending_observation_metrics(
                observation=obs,
                send_started_perf=send_started_perf,
                rpc_started_perf=rpc_started_perf,
                send_completed_perf=send_completed_perf,
                serialize_time=serialize_time,
            )
            self.logger.debug(f"Sent observation #{obs_timestep} | ")
            self.logger.debug(
                f"Observation #{obs_timestep} send RPC time: {(send_completed_perf - rpc_started_perf) * 1000:.2f}ms"
            )
            self.logger.debug(f"Sent observation #{obs_timestep} | ")
            self.logger.debug(
                f"Observation #{obs_timestep} send RPC time: {(send_completed_perf - rpc_started_perf) * 1000:.2f}ms"
            )
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    @staticmethod
    def _format_latency_metric(value_ms: float) -> str:
        if math.isnan(value_ms):
            return "n/a"
        return f"{value_ms:.2f}ms"

    def _register_pending_observation_metrics(
        self,
        observation: TimedObservation,
        send_started_perf: float,
        rpc_started_perf: float,
        send_completed_perf: float,
        serialize_time: float,
    ) -> None:
        key = (observation.get_timestep(), observation.get_timestamp())
        with self._pending_observation_metrics_lock:
            self._pending_observation_metrics[key] = {
                "send_started_perf": send_started_perf,
                "rpc_started_perf": rpc_started_perf,
                "send_completed_perf": send_completed_perf,
                "serialize_time": serialize_time,
            }
            self._pending_observation_metrics.move_to_end(key)
            while len(self._pending_observation_metrics) > self._MAX_PENDING_OBSERVATION_METRICS:
                self._pending_observation_metrics.popitem(last=False)

    def _pop_round_trip_metrics(
        self,
        first_action: TimedAction,
        receive_perf: float,
    ) -> dict[str, float] | None:
        key = (first_action.get_timestep(), first_action.get_timestamp())
        with self._pending_observation_metrics_lock:
            pending_metrics = self._pending_observation_metrics.pop(key, None)

        if pending_metrics is None:
            return None

        return {
            "observation_to_action_rtt_ms": max(0.0, (receive_perf - pending_metrics["send_started_perf"]) * 1000),
            "send_rpc_ms": max(
                0.0,
                (pending_metrics["send_completed_perf"] - pending_metrics["rpc_started_perf"]) * 1000,
            ),
            "wait_after_send_ms": max(0.0, (receive_perf - pending_metrics["send_completed_perf"]) * 1000),
            "serialize_ms": pending_metrics["serialize_time"] * 1000,
        }

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Find the same-timestep actions in the queue and aggregate them using aggregate_fn."""
        if aggregate_fn is None:
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            if new_action.get_timestep() <= latest_action:
                continue
            if new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()],
                        new_action.get_action(),
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def _move_actions_to_client_device(self, timed_actions: list[TimedAction]) -> list[TimedAction]:
        client_device = self.config.client_device
        if client_device == "cpu":
            self.logger.debug(f"Actions kept on device: {client_device}")
            return timed_actions

        for timed_action in timed_actions:
            if timed_action.get_action().device.type != client_device:
                timed_action.action = timed_action.get_action().to(client_device)
        self.logger.debug(f"Converted actions to device: {client_device}")
        return timed_actions

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the gRPC policy server."""
        if not self.uses_grpc_backend:
            self.logger.debug("receive_actions called for non-gRPC backend; nothing to do")
            return

        assert self.stub is not None

        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()
                receive_perf = time.perf_counter()
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                timed_actions = self._move_actions_to_client_device(timed_actions)
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                if len(timed_actions) > 0:
                    first_action = timed_actions[0]
                    round_trip_metrics = self._pop_round_trip_metrics(first_action, receive_perf)
                    if round_trip_metrics is None:
                        round_trip_metrics = {
                            "observation_to_action_rtt_ms": max(
                                0.0, (receive_time - first_action.get_timestamp()) * 1000
                            ),
                            "send_rpc_ms": float("nan"),
                            "wait_after_send_ms": float("nan"),
                            "serialize_ms": float("nan"),
                        }
                    self.logger.info(
                        f"Obs #{first_action.get_timestep()} -> Action RTT: "
                        f"{self._format_latency_metric(round_trip_metrics['observation_to_action_rtt_ms'])} | "
                        f"Send RPC: {self._format_latency_metric(round_trip_metrics['send_rpc_ms'])} | "
                        f"Post-send wait: {self._format_latency_metric(round_trip_metrics['wait_after_send_ms'])} | "
                        f"Obs serialize: {self._format_latency_metric(round_trip_metrics['serialize_ms'])} | "
                        f"Action deserialize: {self._format_latency_metric(deserialize_time * 1000)}"
                    )

                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]

                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                    first_action_timestep = timed_actions[0].get_timestep()
                    observation_to_action_rtt = (receive_time - timed_actions[0].get_timestamp()) * 1000
                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Observation-to-action RTT: {observation_to_action_rtt:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()

                if verbose and len(timed_actions) > 0:
                    new_size, new_timesteps = self._inspect_action_queue()
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue."""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Read and perform an action from the local queue."""
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        performed_action = self.robot.send_action(self._action_tensor_to_action_dict(timed_action.get_action()))
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )
            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return performed_action

    def _ready_to_send_observation(self):
        """Flag when the client is ready to send a new observation."""
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()

        if self.action_chunk_size <= 0:
            return True
        return queue_size / self.action_chunk_size <= self._chunk_size_threshold

    def _capture_timed_observation(self, task: str) -> tuple[TimedObservation, RawObservation, int, float]:
        start_time = time.perf_counter()
        raw_observation: RawObservation = self.robot.get_observation()
        raw_observation = apply_observation_crops(raw_observation, self.config.image_crop_params)
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

        return observation, raw_observation, current_queue_size, obs_capture_time

    def _log_observation_capture(self, observation: TimedObservation, obs_capture_time: float, verbose: bool) -> None:
        if not verbose:
            return

        fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
        self.logger.info(
            f"Obs #{observation.get_timestep()} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
            f"Target: {fps_metrics['target_fps']:.2f}"
        )
        self.logger.debug(
            f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
        )

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        try:
            observation, raw_observation, current_queue_size, obs_capture_time = self._capture_timed_observation(task)
            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                self.must_go.clear()

            self._log_observation_capture(observation, obs_capture_time, verbose)
            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
            return None
    def control_loop_groot_inference(self, task: str, verbose: bool = False) -> RawObservation | None:
        if not self.uses_groot_backend:
            raise RuntimeError("control_loop_groot_inference is only valid for the GR00T backend")
        if self.remote_client is None:
            raise RuntimeError("GR00T client not started. Run RobotClient.start() before requesting actions.")

        try:
            observation, raw_observation, current_queue_size, obs_capture_time = self._capture_timed_observation(task)
            groot_observation = build_groot_n16_observation(
                raw_observation,
                front_camera_key=self.config.groot_front_camera_key,
                left_wrist_camera_key=self.config.groot_left_wrist_camera_key,
                right_wrist_camera_key=self.config.groot_right_wrist_camera_key,
                image_size=self.config.groot_image_size,
            )

            request_start = time.perf_counter()
            action_dict = self.remote_client.get_action(groot_observation)
            request_time = time.perf_counter() - request_start

            timed_actions = groot_n16_action_dict_to_timed_actions(
                action_dict,
                timestamp=observation.get_timestamp(),
                timestep=observation.get_timestep(),
                environment_dt=self.config.environment_dt,
                client_device=self.config.client_device,
            )
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
                f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go}) | "
                f"GR00T request time: {request_time * 1000:.2f}ms"
            )
            self._log_observation_capture(observation, obs_capture_time, verbose)

            if verbose and timed_actions:
                new_size, new_timesteps = self._inspect_action_queue()
                incoming_timesteps = [a.get_timestep() for a in timed_actions]
                self.logger.info(
                    f"Received GR00T action chunk for step #{incoming_timesteps[0]} | "
                    f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                    f"Request time: {request_time * 1000:.2f}ms"
                )
                self.logger.debug(
                    f"Queue update complete ({queue_update_time:.6f}s) | "
                    f"Before: {old_size} items ({old_timesteps[:1] + old_timesteps[-1:] if old_timesteps else []}) | "
                    f"After: {new_size} items ({new_timesteps[:1] + new_timesteps[-1:] if new_timesteps else []})"
                )

            return raw_observation
        except Exception as e:
            self.logger.error(f"Error in GR00T inference loop: {e}")
            return None

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and running remote inference."""
        if self.uses_grpc_backend:
            self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        performed_action = None
        captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            if self.actions_available():
                performed_action = self.control_loop_action(verbose)
            if self._ready_to_send_observation():
                if self.uses_grpc_backend:
                    captured_observation = self.control_loop_observation(task, verbose)
                else:
                    captured_observation = self.control_loop_groot_inference(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return captured_observation, performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = RobotClient(cfg)
    action_receiver_thread = None

    if client.start():
        if client.uses_grpc_backend:
            client.logger.info("Starting action receiver thread...")
            action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
            action_receiver_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            if action_receiver_thread is not None:
                action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
