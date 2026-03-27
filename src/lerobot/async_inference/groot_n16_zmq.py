# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .helpers import TimedAction

GROOT_N16_ACTION_KEYS = [f"right_arm_{i}" for i in range(7)] + ["right_gripper_0"]
GROOT_N16_ARM_KEYS = [f"right_arm_{i}" for i in range(7)]


def _import_zmq_dependencies():
    try:
        import msgpack
        import zmq
    except ImportError as exc:  # pragma: no cover - exercised in environments without optional deps
        raise ImportError(
            "The 'groot_n16_zmq' backend requires both 'msgpack' and 'pyzmq' to be installed."
        ) from exc

    return msgpack, zmq


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        msgpack, _ = _import_zmq_dependencies()
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes):
        msgpack, _ = _import_zmq_dependencies()
        return msgpack.unpackb(
            data,
            object_hook=MsgSerializer._decode,
            strict_map_key=False,
        )

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Non-serializable type: {type(obj)}")

    @staticmethod
    def _decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


class GR00TZMQClient:
    def __init__(self, server_address: str, timeout_ms: int):
        _, zmq = _import_zmq_dependencies()
        self._zmq = zmq
        self.address = normalize_zmq_server_address(server_address)
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)

    def _call(self, endpoint: str, data: dict[str, Any] | None = None):
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data

        try:
            self.sock.send(MsgSerializer.to_bytes(request))
            raw = self.sock.recv()
        except self._zmq.Again as exc:
            raise TimeoutError(f"Server response timeout (endpoint={endpoint})") from exc
        except self._zmq.ZMQError as exc:
            raise RuntimeError(f"ZMQ error: {exc}") from exc

        response = MsgSerializer.from_bytes(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            return bool(self._call("ping"))
        except Exception:
            return False

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self._call("get_action", {"observation": observation, "options": None})
        if isinstance(response, (list, tuple)) and len(response) >= 1:
            return dict(tuple(response)[0])
        if isinstance(response, dict):
            return response
        raise RuntimeError(f"Unexpected GR00T response type: {type(response)}")

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._call("reset", {"options": options})

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


def normalize_zmq_server_address(server_address: str) -> str:
    if server_address.startswith("tcp://"):
        return server_address

    host, sep, port = server_address.rpartition(":")
    if not sep or not host or not port:
        raise ValueError(
            f"GR00T backend expects server_address in 'host:port' format, got {server_address!r}"
        )

    return f"tcp://{host}:{port}"


def validate_groot_robot_compatibility(
    robot: Any,
    *,
    front_camera_key: str,
    right_wrist_camera_key: str,
) -> None:
    action_keys = list(robot.action_features)
    if action_keys != GROOT_N16_ACTION_KEYS:
        raise ValueError(
            "The 'groot_n16_zmq' backend currently supports only right_arm_0..6 + right_gripper_0. "
            f"Received action features: {action_keys}"
        )

    missing_keys = [
        key
        for key in [*GROOT_N16_ACTION_KEYS, front_camera_key, right_wrist_camera_key]
        if key not in robot.observation_features
    ]
    if missing_keys:
        raise ValueError(
            "Robot observation features are missing keys required by the GR00T backend: "
            f"{missing_keys}"
        )


def _resize_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == image_size:
        return np.ascontiguousarray(image)

    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    resized = torch.nn.functional.interpolate(
        image_tensor,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return np.ascontiguousarray(
        resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )


def ensure_uint8_hwc_image(image: Any, image_size: tuple[int, int]) -> np.ndarray:
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[2] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3), got {image_arr.shape}")

    if image_arr.dtype != np.uint8:
        if np.issubdtype(image_arr.dtype, np.floating):
            image_arr = np.clip(image_arr, 0.0, 1.0 if image_arr.max(initial=0.0) <= 1.0 else 255.0)
            if image_arr.max(initial=0.0) <= 1.0:
                image_arr = image_arr * 255.0
        image_arr = image_arr.astype(np.uint8)

    return _resize_image(np.ascontiguousarray(image_arr), image_size)


def build_groot_n16_observation(
    raw_observation: dict[str, Any],
    *,
    front_camera_key: str,
    right_wrist_camera_key: str,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    missing_keys = [
        key
        for key in [*GROOT_N16_ARM_KEYS, "right_gripper_0", front_camera_key, right_wrist_camera_key]
        if key not in raw_observation
    ]
    if missing_keys:
        raise KeyError(f"Raw observation missing keys required by GR00T backend: {missing_keys}")

    front_image = ensure_uint8_hwc_image(raw_observation[front_camera_key], image_size)
    right_image = ensure_uint8_hwc_image(raw_observation[right_wrist_camera_key], image_size)
    right_arm = np.asarray([raw_observation[key] for key in GROOT_N16_ARM_KEYS], dtype=np.float32)
    right_gripper = np.asarray([raw_observation["right_gripper_0"]], dtype=np.float32)
    task = str(raw_observation.get("task", ""))

    return {
        "video": {
            "cam_front_head": front_image[np.newaxis, np.newaxis],
            "cam_right_wrist": right_image[np.newaxis, np.newaxis],
        },
        "state": {
            "right_arm": right_arm[np.newaxis, np.newaxis],
            "right_gripper": right_gripper[np.newaxis, np.newaxis],
        },
        "language": {
            "annotation.human.task_description": [[task]],
        },
    }


def groot_n16_action_dict_to_timed_actions(
    action_dict: dict[str, Any],
    *,
    timestamp: float,
    timestep: int,
    environment_dt: float,
    client_device: str = "cpu",
) -> list[TimedAction]:
    right_arm = np.asarray(action_dict.get("right_arm"), dtype=np.float32)
    right_gripper = np.asarray(action_dict.get("right_gripper"), dtype=np.float32)

    if right_arm.shape[:1] != (1,) or right_arm.ndim != 3 or right_arm.shape[2] != 7:
        raise ValueError(f"Expected right_arm shape (1, T, 7), got {right_arm.shape}")
    if right_gripper.shape[:1] != (1,) or right_gripper.ndim != 3 or right_gripper.shape[2] != 1:
        raise ValueError(f"Expected right_gripper shape (1, T, 1), got {right_gripper.shape}")
    if right_arm.shape[1] != right_gripper.shape[1]:
        raise ValueError(
            "GR00T action chunk lengths do not match: "
            f"right_arm={right_arm.shape[1]}, right_gripper={right_gripper.shape[1]}"
        )

    device = torch.device(client_device)
    timed_actions = []
    for i in range(right_arm.shape[1]):
        action_tensor = torch.from_numpy(
            np.concatenate([right_arm[0, i], right_gripper[0, i]], axis=0)
        ).to(device=device, dtype=torch.float32)
        timed_actions.append(
            TimedAction(
                timestamp=timestamp + i * environment_dt,
                timestep=timestep + i,
                action=action_tensor,
            )
        )

    return timed_actions
