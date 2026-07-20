from __future__ import annotations

import importlib.machinery
import sys
import threading
import types
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.utils import import_utils


@pytest.fixture()
def async_import_stubs(monkeypatch):
    monkeypatch.setitem(import_utils._require_package_cache, "grpc", True)

    if "grpc" not in sys.modules:
        grpc_stub = types.ModuleType("grpc")
        grpc_stub.__spec__ = importlib.machinery.ModuleSpec("grpc", loader=None)
        grpc_stub.insecure_channel = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "grpc", grpc_stub)

    if "lerobot.transport.services_pb2_grpc" not in sys.modules:
        services_pb2_grpc_stub = types.ModuleType("lerobot.transport.services_pb2_grpc")
        services_pb2_grpc_stub.AsyncInferenceStub = lambda channel: None
        monkeypatch.setitem(sys.modules, "lerobot.transport.services_pb2_grpc", services_pb2_grpc_stub)


def test_supported_names_include_pi05_zmq_and_pi05_thor(async_import_stubs):
    from lerobot.async_inference.constants import SUPPORTED_BACKENDS

    assert "pi05_zmq" in SUPPORTED_BACKENDS
    assert "pi05_thor" in SUPPORTED_BACKENDS


def test_registry_keeps_both_keys_and_labels(async_import_stubs):
    from lerobot.async_inference.robot_client import REMOTE_ZMQ_BACKENDS

    assert REMOTE_ZMQ_BACKENDS["pi05_zmq"]["label"] == "Pi05"
    assert REMOTE_ZMQ_BACKENDS["pi05_thor"]["label"] == "Pi05 Thor"


def test_pi05thorclient_aliases_pi05zmqclient(async_import_stubs):
    from lerobot.async_inference.policy.pi05_thor import Pi05ThorClient
    from lerobot.async_inference.policy.pi05_zmq import Pi05ZMQClient

    assert Pi05ThorClient is Pi05ZMQClient


def test_all_legacy_thor_symbols_remain_importable(async_import_stubs):
    import lerobot.async_inference.policy.pi05_thor as thor

    for name in [
        "ARM_DOF",
        "GRIPPER_DOF",
        "ACTION_DOF",
        "RIGHT_ARM_KEYS",
        "LEFT_ARM_KEYS",
        "GRIPPER_KEYS",
        "PI05_ACTION_KEYS",
        "PI05_STATE_KEYS",
        "MsgSerializer",
        "Pi05ThorClient",
        "normalize_zmq_server_address",
        "validate_pi05_robot_compatibility",
        "ensure_uint8_hwc_image",
        "build_pi05_observation",
        "parse_pi05_actions",
        "get_pi05_action_part",
        "squeeze_optional_batch",
        "actions_to_timed_actions",
        "pi05_action_dict_to_timed_actions",
    ]:
        assert hasattr(thor, name)


def test_both_registry_entries_use_canonical_behavior(async_import_stubs):
    from lerobot.async_inference.policy.pi05_zmq import (
        Pi05ZMQClient,
        build_pi05_observation,
        pi05_action_dict_to_timed_actions,
        validate_pi05_robot_compatibility,
    )
    from lerobot.async_inference.robot_client import REMOTE_ZMQ_BACKENDS

    assert REMOTE_ZMQ_BACKENDS["pi05_zmq"]["client_cls"] is Pi05ZMQClient
    assert REMOTE_ZMQ_BACKENDS["pi05_thor"]["client_cls"] is Pi05ZMQClient
    assert REMOTE_ZMQ_BACKENDS["pi05_zmq"]["validator"] is validate_pi05_robot_compatibility
    assert REMOTE_ZMQ_BACKENDS["pi05_thor"]["validator"] is validate_pi05_robot_compatibility
    assert REMOTE_ZMQ_BACKENDS["pi05_zmq"]["build_observation"] is build_pi05_observation
    assert REMOTE_ZMQ_BACKENDS["pi05_thor"]["build_observation"] is build_pi05_observation
    assert REMOTE_ZMQ_BACKENDS["pi05_zmq"]["convert_actions"] is pi05_action_dict_to_timed_actions
    assert REMOTE_ZMQ_BACKENDS["pi05_thor"]["convert_actions"] is pi05_action_dict_to_timed_actions


def test_protocol_golden_fixtures_match_for_both_aliases(async_import_stubs):
    from lerobot.async_inference.policy import pi05_thor, pi05_zmq

    raw_observation = {
        **{key: float(i) for i, key in enumerate(pi05_zmq.PI05_STATE_KEYS)},
        "front": np.ones((2, 3, 3), dtype=np.uint8),
        "left": np.ones((3, 2, 3), dtype=np.float32),
        "right": np.ones((3, 2, 2), dtype=np.uint8),
        "task": "pick",
    }

    zmq_observation = pi05_zmq.build_pi05_observation(
        raw_observation,
        front_camera_key="front",
        left_wrist_camera_key="left",
        right_wrist_camera_key="right",
    )
    thor_observation = pi05_thor.build_pi05_observation(
        raw_observation,
        front_camera_key="front",
        left_wrist_camera_key="left",
        right_wrist_camera_key="right",
    )

    assert pi05_thor.MsgSerializer.to_bytes(zmq_observation) == pi05_zmq.MsgSerializer.to_bytes(
        thor_observation
    )

    actions = np.arange(32, dtype=np.float32).reshape(2, 16)
    action_dict = {"actions": actions}
    np.testing.assert_array_equal(
        pi05_thor.parse_pi05_actions(action_dict),
        pi05_zmq.parse_pi05_actions(action_dict),
    )

    thor_timed = pi05_thor.pi05_action_dict_to_timed_actions(
        action_dict,
        timestamp=10.0,
        timestep=7,
        environment_dt=0.1,
    )
    zmq_timed = pi05_zmq.pi05_action_dict_to_timed_actions(
        action_dict,
        timestamp=10.0,
        timestep=7,
        environment_dt=0.1,
    )

    assert [action.get_timestep() for action in thor_timed] == [7, 8]
    assert [action.get_timestamp() for action in thor_timed] == [10.0, 10.1]
    for thor_action, zmq_action in zip(thor_timed, zmq_timed, strict=True):
        torch.testing.assert_close(thor_action.get_action(), zmq_action.get_action())


class ScriptedRemoteClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def get_action(self, remote_observation):
        self.calls.append(remote_observation)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _make_remote_loop_client():
    from lerobot.async_inference.robot_client import RobotClient

    client = object.__new__(RobotClient)
    client.backend = "pi05_zmq"
    client.remote_observation_queue = Queue(maxsize=1)
    client.action_queue = Queue()
    client.action_queue_lock = nullcontext()
    client.latest_action_lock = nullcontext()
    client.latest_action = -1
    client.action_chunk_size = -1
    client.chunks_received = 0
    client.shutdown_event = threading.Event()
    client.must_go = SimpleNamespace(set=lambda: None)
    client.config = SimpleNamespace(
        environment_dt=0.1,
        client_device="cpu",
        aggregate_fn=lambda _old, new: new,
    )
    client.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    client.start_barrier = SimpleNamespace(wait=lambda: None)
    return client


def _remote_observation(timestep: int):
    from lerobot.async_inference.helpers import TimedObservation

    observation = TimedObservation(timestamp=float(timestep), timestep=timestep, observation={})
    return observation, {"observation_id": timestep}


def test_remote_loop_drops_failed_call_and_accepts_newest_observation(
    async_import_stubs, monkeypatch
):
    from lerobot.async_inference.robot_client import RobotClient

    client = _make_remote_loop_client()
    client.remote_client = ScriptedRemoteClient(
        [
            TimeoutError("drop this request"),
            {"actions": torch.ones((2, 16)).numpy()},
        ]
    )
    client.remote_observation_queue.put(_remote_observation(1))

    original_aggregate = RobotClient._aggregate_action_queues

    def stop_after_success(self, timed_actions, aggregate_fn=None):
        original_aggregate(self, timed_actions, aggregate_fn)
        self.shutdown_event.set()

    monkeypatch.setattr(RobotClient, "_aggregate_action_queues", stop_after_success)

    def enqueue_latest_after_error(*args, **kwargs):
        client.remote_observation_queue.put(_remote_observation(3))

    client.logger.error = enqueue_latest_after_error

    client.receive_remote_actions()

    assert client.remote_client.calls == [{"observation_id": 1}, {"observation_id": 3}]
    assert [action.get_timestep() for action in client.action_queue.queue] == [3, 4]


def test_remote_loop_does_not_retry_inside_same_iteration(async_import_stubs):
    client = _make_remote_loop_client()
    client.remote_client = ScriptedRemoteClient([TimeoutError("single failure")])
    client.remote_observation_queue.put(_remote_observation(5))
    client.logger.error = lambda *args, **kwargs: client.shutdown_event.set()

    client.receive_remote_actions()

    assert client.remote_client.calls == [{"observation_id": 5}]
    assert client.action_queue.empty()
