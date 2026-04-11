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

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
    DEFAULT_ZMQ_TIMEOUT_MS,
    SUPPORTED_BACKENDS,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}

AUTO_AGGREGATE_FN_NAME = "auto"
AUTO_TRTC_POLICIES = {"pi0", "pi05"}
logger = logging.getLogger(__name__)


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


def resolve_aggregate_function_name(
    aggregate_fn_name: str,
    policy_type: str,
    pretrained_name_or_path: str,
) -> str:
    if aggregate_fn_name != AUTO_AGGREGATE_FN_NAME:
        return aggregate_fn_name

    if not pretrained_name_or_path:
        return "weighted_average"

    if policy_type not in AUTO_TRTC_POLICIES:
        return "weighted_average"

    try:
        policy_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        if getattr(policy_config, "use_action_prefix_conditioning", False):
            resolved_name = "latest_only"
        else:
            resolved_name = "weighted_average"
        logger.info(
            "Resolved aggregate_fn_name=auto to '%s' using policy config from %s",
            resolved_name,
            pretrained_name_or_path,
        )
        return resolved_name
    except Exception as exc:  # noqa: BLE001
        resolved_name = "latest_only" if policy_type in AUTO_TRTC_POLICIES else "weighted_average"
        logger.warning(
            "Falling back to aggregate_fn_name='%s' because policy config for %s could not be loaded: %s",
            resolved_name,
            pretrained_name_or_path,
            exc,
        )
        return resolved_name


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )
    logging: bool = field(
        default=False,
        metadata={
            "help": "Whether to save raw images received by the gRPC server and action chunks sent to the client"
        },
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
            "logging": self.logging,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Remote inference backend configuration
    backend: str = field(
        default="grpc",
        metadata={"help": f"Remote backend to use. Options: {SUPPORTED_BACKENDS}"},
    )

    # Policy configuration (required for grpc backend only)
    policy_type: str = field(default="", metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(default="", metadata={"help": "Pretrained model name or path"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    obs_atol: float = field(
        default=1.0,
        metadata={"help": "State-space absolute tolerance for server-side observation similarity filtering"},
    )
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    image_crop_params: dict[str, tuple[int, int, int, int]] = field(
        default_factory=dict,
        metadata={
            "help": "Optional per-camera crop parameters as (top, left, height, width), applied on the client before sending observations"
        },
    )
    zmq_timeout_ms: int = field(
        default=DEFAULT_ZMQ_TIMEOUT_MS,
        metadata={"help": "ZMQ send/recv timeout in milliseconds for the GR00T backend"},
    )
    groot_front_camera_key: str = field(
        default="front",
        metadata={"help": "Robot observation key mapped to GR00T cam_front_head"},
    )
    groot_right_wrist_camera_key: str = field(
        default="right",
        metadata={"help": "Robot observation key mapped to GR00T cam_right_wrist"},
    )
    groot_left_wrist_camera_key: str = field(
        default="left",
        metadata={"help": "Robot observation key mapped to GR00T cam_left_wrist"},
    )
    groot_image_size: tuple[int, int] = field(
        default=(480, 640),
        metadata={"help": "Final (height, width) of GR00T camera observations after crop/resize"},
    )

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default=AUTO_AGGREGATE_FN_NAME,
        metadata={
            "help": f"Name of aggregate function to use. Options: {[AUTO_AGGREGATE_FN_NAME, *list(AGGREGATE_FUNCTIONS.keys())]}"
        },
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}")

        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if self.backend == "grpc":
            if not self.policy_type:
                raise ValueError("policy_type cannot be empty when backend='grpc'")

            if not self.pretrained_name_or_path:
                raise ValueError("pretrained_name_or_path cannot be empty when backend='grpc'")

            if not self.policy_device:
                raise ValueError("policy_device cannot be empty when backend='grpc'")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.obs_atol < 0:
            raise ValueError(f"obs_atol must be non-negative, got {self.obs_atol}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.zmq_timeout_ms <= 0:
            raise ValueError(f"zmq_timeout_ms must be positive, got {self.zmq_timeout_ms}")

        if not self.groot_front_camera_key:
            raise ValueError("groot_front_camera_key cannot be empty")

        if not self.groot_right_wrist_camera_key:
            raise ValueError("groot_right_wrist_camera_key cannot be empty")

        if not self.groot_left_wrist_camera_key:
            raise ValueError("groot_left_wrist_camera_key cannot be empty")

        if len(self.groot_image_size) != 2:
            raise ValueError(
                f"groot_image_size must be a pair of integers (height, width), got {self.groot_image_size}"
            )
        groot_height, groot_width = (int(v) for v in self.groot_image_size)
        if groot_height <= 0 or groot_width <= 0:
            raise ValueError(f"groot_image_size dimensions must be positive, got {self.groot_image_size}")
        self.groot_image_size = (groot_height, groot_width)

        normalized_crop_params = {}
        for key, value in self.image_crop_params.items():
            if len(value) != 4:
                raise ValueError(
                    f"image_crop_params['{key}'] must have four values (top, left, height, width), got {value}"
                )
            top, left, height, width = (int(v) for v in value)
            if top < 0 or left < 0:
                raise ValueError(
                    f"image_crop_params['{key}'] must use non-negative top/left offsets, got {value}"
                )
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"image_crop_params['{key}'] must use positive height/width, got {value}"
                )
            normalized_crop_params[key] = (top, left, height, width)
        self.image_crop_params = normalized_crop_params
        self.aggregate_fn_name = resolve_aggregate_function_name(
            self.aggregate_fn_name,
            policy_type=self.policy_type,
            pretrained_name_or_path=self.pretrained_name_or_path,
        )
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "backend": self.backend,
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "obs_atol": self.obs_atol,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "image_crop_params": self.image_crop_params,
            "zmq_timeout_ms": self.zmq_timeout_ms,
            "groot_front_camera_key": self.groot_front_camera_key,
            "groot_left_wrist_camera_key": self.groot_left_wrist_camera_key,
            "groot_right_wrist_camera_key": self.groot_right_wrist_camera_key,
            "groot_image_size": self.groot_image_size,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
        }
