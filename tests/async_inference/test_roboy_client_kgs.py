from __future__ import annotations

import logging
import sys
import threading
import time
import types
from queue import Queue
from types import SimpleNamespace

import pytest
import torch


def allow_async_import_without_grpcio():
    try:
        import grpc  # noqa: F401

        return
    except ImportError:
        pass

    grpc = types.ModuleType("grpc")
    grpc.__version__ = "1.73.1"
    grpc.RpcError = RuntimeError

    grpc_utilities = types.ModuleType("grpc._utilities")
    grpc_utilities.first_version_is_lower = lambda current, minimum: False
    grpc._utilities = grpc_utilities

    sys.modules["grpc"] = grpc
    sys.modules["grpc._utilities"] = grpc_utilities

    from lerobot.utils import import_utils

    import_utils._require_package_cache["grpc"] = True


class FakeRobot:
    def __init__(self):
        self.ready_calls = 0
        self.action_features = ["a", "b"]
        self.is_connected = True

    def move_to_ready_pose(self):
        self.ready_calls += 1

    def get_observation(self):
        return {"state": torch.zeros(2)}

    def send_action(self, action):
        return action

    def disconnect(self):
        self.is_connected = False


class FakeRemoteClient:
    def __init__(self, callback=None):
        self.callback = callback

    def get_action(self, remote_observation):
        if self.callback is not None:
            self.callback()
        return {"action": remote_observation}


class FakeTimedAction:
    def __init__(self, action=None, timestep=1, timestamp=0.0):
        self.action = torch.tensor(action or [1.0, 2.0])
        self.timestep = timestep
        self.timestamp = timestamp

    def get_action(self):
        return self.action

    def get_timestep(self):
        return self.timestep

    def get_timestamp(self):
        return self.timestamp


def make_client():
    allow_async_import_without_grpcio()

    from lerobot.async_inference.roboy_client_kgs import KgsPi05RobotClient

    client = object.__new__(KgsPi05RobotClient)
    client.config = SimpleNamespace(environment_dt=0.01, aggregate_fn=None)
    client.robot = FakeRobot()
    client.backend = "pi05_zmq"
    client.remote_client = FakeRemoteClient()
    client.server_address = "localhost:1"
    client.shutdown_event = threading.Event()
    client.latest_action_lock = threading.Lock()
    client.latest_action = 0
    client.action_chunk_size = 10
    client._chunk_size_threshold = 0.5
    client.action_queue = Queue()
    client.remote_observation_queue = Queue(maxsize=1)
    client.action_queue_lock = threading.Lock()
    client.action_queue_size = []
    client.start_barrier = threading.Barrier(2)
    client.must_go = threading.Event()
    client.must_go.set()
    client.channel = None
    client.logger = logging.getLogger("test_roboy_client_kgs")
    client._init_kgs_pause_state()
    return client


def test_pause_transition_clears_queues_and_starts_one_worker(monkeypatch):
    client = make_client()
    client.action_queue.put(object())
    client.remote_observation_queue.put((0, object(), object()))
    started = []

    monkeypatch.setattr(client, "_start_ready_pose_worker", lambda: started.append("worker"))

    client.pause_and_ready()

    assert client.pause_event.is_set()
    assert client.pause_generation == 1
    assert client.action_queue.empty()
    assert client.remote_observation_queue.empty()
    assert not client.must_go.is_set()
    assert client.ready_pose_in_progress is True
    assert started == ["worker"]

    client.pause_and_ready()

    assert client.pause_generation == 2
    assert started == ["worker"]


def test_resume_during_ready_pose_is_deferred_until_worker_completes():
    client = make_client()
    client.pause_event.set()
    client.ready_pose_in_progress = True
    client.pause_generation = 1
    client.must_go.clear()
    client.action_queue.put(object())
    client.remote_observation_queue.put((1, object(), object()))

    client.resume_from_pause()

    assert client.resume_requested is True
    assert client.pause_event.is_set()
    assert not client.must_go.is_set()
    assert not client.action_queue.empty()
    assert not client.remote_observation_queue.empty()

    client._complete_ready_pose_transition()

    assert client.ready_pose_in_progress is False
    assert client.resume_requested is False
    assert not client.pause_event.is_set()
    assert client.must_go.is_set()
    assert client.pause_generation == 2
    assert client.action_queue.empty()
    assert client.remote_observation_queue.empty()


def test_deferred_resume_after_failed_ready_pose_stays_paused():
    client = make_client()
    client.pause_event.set()
    client.ready_pose_in_progress = True
    client.resume_requested = True
    client.pause_generation = 3
    client.must_go.clear()

    client._complete_ready_pose_transition(ready_succeeded=False)

    assert client.ready_pose_in_progress is False
    assert client.resume_requested is False
    assert client.pause_event.is_set()
    assert not client.must_go.is_set()
    assert client.pause_generation == 3


def test_control_loop_action_does_not_send_when_paused(monkeypatch):
    client = make_client()
    client.pause_event.set()
    client.action_queue.put(FakeTimedAction())
    sent = []

    monkeypatch.setattr(client.robot, "send_action", lambda action: sent.append(action))

    assert client.control_loop_action() is None
    assert sent == []
    assert not client.action_queue.empty()


def test_pause_waits_for_inflight_action_before_setting_paused(monkeypatch):
    client = make_client()
    client.action_queue.put(FakeTimedAction())
    send_started = threading.Event()
    release_send = threading.Event()
    worker_started = []

    def blocking_send(action):
        send_started.set()
        assert release_send.wait(timeout=1.0)
        return action

    monkeypatch.setattr(client.robot, "send_action", blocking_send)
    monkeypatch.setattr(client, "_start_ready_pose_worker", lambda: worker_started.append("worker"))

    action_thread = threading.Thread(target=client.control_loop_action)
    action_thread.start()
    assert send_started.wait(timeout=1.0)

    pause_thread = threading.Thread(target=client.pause_and_ready)
    pause_thread.start()
    time.sleep(0.05)

    assert not client.pause_event.is_set()
    assert not client.ready_pose_in_progress

    release_send.set()
    action_thread.join(timeout=1.0)
    pause_thread.join(timeout=1.0)

    assert not action_thread.is_alive()
    assert not pause_thread.is_alive()
    assert client.pause_event.is_set()
    assert client.ready_pose_in_progress
    assert worker_started == ["worker"]


def test_stale_generation_response_is_discarded_before_processing(monkeypatch):
    client = make_client()
    client.pause_generation = 1
    client.must_go.clear()

    def advance_generation():
        with client.pause_state_lock:
            client.pause_generation = 2

    client.remote_client = FakeRemoteClient(callback=advance_generation)
    calls = {"convert": 0, "aggregate": 0}

    def convert(*args, **kwargs):
        calls["convert"] += 1
        return []

    def aggregate(*args, **kwargs):
        calls["aggregate"] += 1

    monkeypatch.setattr(client, "_convert_remote_actions", convert)
    monkeypatch.setattr(client, "_aggregate_action_queues", aggregate)

    accepted = client._receive_and_enqueue_remote_actions(1, SimpleNamespace(get_timestep=lambda: 0), {})

    assert accepted is False
    assert calls == {"convert": 0, "aggregate": 0}
    assert client.action_chunk_size == 10
    assert not client.must_go.is_set()


def test_generation_changed_after_conversion_discards_before_queue_mutation(monkeypatch):
    client = make_client()
    client.pause_generation = 1
    client.must_go.clear()
    calls = {"convert": 0, "aggregate": 0}

    def convert(*args, **kwargs):
        calls["convert"] += 1
        with client.pause_state_lock:
            client.pause_generation = 2
        return [FakeTimedAction()] * 20

    def aggregate(*args, **kwargs):
        calls["aggregate"] += 1

    monkeypatch.setattr(client, "_convert_remote_actions", convert)
    monkeypatch.setattr(client, "_aggregate_action_queues", aggregate)

    accepted = client._receive_and_enqueue_remote_actions(1, SimpleNamespace(get_timestep=lambda: 0), {})

    assert accepted is False
    assert calls == {"convert": 1, "aggregate": 0}
    assert client.action_chunk_size == 10
    assert not client.must_go.is_set()


@pytest.mark.parametrize("paused,readying", [(True, False), (False, True)])
def test_paused_or_readying_response_is_discarded_before_processing(monkeypatch, paused, readying):
    client = make_client()
    client.pause_generation = 1
    client.must_go.clear()
    if paused:
        client.pause_event.set()
    client.ready_pose_in_progress = readying
    calls = {"convert": 0, "aggregate": 0}

    monkeypatch.setattr(client, "_convert_remote_actions", lambda *args, **kwargs: calls.__setitem__("convert", 1))
    monkeypatch.setattr(client, "_aggregate_action_queues", lambda *args, **kwargs: calls.__setitem__("aggregate", 1))

    accepted = client._receive_and_enqueue_remote_actions(1, SimpleNamespace(get_timestep=lambda: 0), {})

    assert accepted is False
    assert calls == {"convert": 0, "aggregate": 0}
    assert not client.must_go.is_set()


def test_non_tty_keyboard_startup_disables_listener(monkeypatch, caplog):
    allow_async_import_without_grpcio()

    import lerobot.async_inference.roboy_client_kgs as kgs

    client = make_client()
    fake_stdin = SimpleNamespace(isatty=lambda: False)
    monkeypatch.setattr(kgs.sys, "stdin", fake_stdin)

    with caplog.at_level(logging.WARNING):
        enabled = client.start_keyboard_listener()

    assert enabled is False
    assert client._keyboard_thread is None
    assert "not a TTY" in caplog.text


def test_terminal_restore_clears_saved_terminal_state(monkeypatch):
    allow_async_import_without_grpcio()

    import lerobot.async_inference.roboy_client_kgs as kgs

    client = make_client()
    calls = []
    monkeypatch.setattr(kgs.termios, "tcsetattr", lambda fd, when, attrs: calls.append((fd, when, attrs)))

    client._terminal_fd = 7
    client._terminal_attrs = ["saved"]
    client._restore_terminal()

    assert calls == [(7, kgs.termios.TCSADRAIN, ["saved"])]
    assert client._terminal_fd is None
    assert client._terminal_attrs is None


@pytest.mark.parametrize("backend", ["pi05_zmq", "pi05_thor"])
def test_kgs_backend_allowlist_accepts_pi05_backends(backend):
    allow_async_import_without_grpcio()

    from lerobot.async_inference.roboy_client_kgs import validate_kgs_backend

    validate_kgs_backend(backend)


@pytest.mark.parametrize("backend", ["grpc", "groot_zmq"])
def test_kgs_backend_allowlist_rejects_other_backends(backend):
    allow_async_import_without_grpcio()

    from lerobot.async_inference.roboy_client_kgs import validate_kgs_backend

    with pytest.raises(ValueError, match="only supports remote Pi0.5"):
        validate_kgs_backend(backend)
