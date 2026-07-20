# ruff: noqa: E402

from __future__ import annotations

import threading
from typing import Any

import pytest

pytest.importorskip("msgpack")
zmq = pytest.importorskip("zmq")

from lerobot.utils import import_utils

import_utils._require_package_cache["grpc"] = True

from lerobot.async_inference.policy import pi05_zmq
from lerobot.async_inference.policy.pi05_zmq import MsgSerializer, Pi05ZMQClient


class FakeSocket:
    def __init__(self, *, recv_result: Any = True, send_error: BaseException | None = None):
        self.recv_result = recv_result
        self.send_error = send_error
        self.sent = []
        self.options = []
        self.connected = []
        self.close_calls = 0

    def setsockopt(self, option, value):
        self.options.append((option, value))

    def connect(self, address):
        self.connected.append(address)

    def send(self, payload):
        self.sent.append(MsgSerializer.from_bytes(payload))
        if self.send_error is not None:
            raise self.send_error

    def recv(self):
        if isinstance(self.recv_result, BaseException):
            raise self.recv_result
        return MsgSerializer.to_bytes(self.recv_result)

    def close(self):
        self.close_calls += 1

    @property
    def closed(self):
        return self.close_calls > 0


class FakeContext:
    def __init__(self, sockets):
        self.sockets = list(sockets)
        self.created = []
        self.term_calls = 0

    def socket(self, socket_type):
        assert socket_type == zmq.REQ
        sock = self.sockets.pop(0)
        self.created.append(sock)
        return sock

    def term(self):
        self.term_calls += 1

    @property
    def terminated(self):
        return self.term_calls > 0


class FakeZMQ:
    REQ = zmq.REQ
    RCVTIMEO = zmq.RCVTIMEO
    SNDTIMEO = zmq.SNDTIMEO
    LINGER = zmq.LINGER
    Again = zmq.Again
    ZMQError = zmq.ZMQError

    def __init__(self, context):
        self._context = context

    def Context(self):  # noqa: N802
        return self._context


@pytest.fixture()
def fake_client(monkeypatch):
    def make_client(*sockets):
        context = FakeContext(sockets)
        fake_zmq = FakeZMQ(context)
        monkeypatch.setattr(pi05_zmq, "_import_zmq_dependencies", lambda: (__import__("msgpack"), fake_zmq))
        client = Pi05ZMQClient("127.0.0.1:5555", timeout_ms=25)
        return client, context

    return make_client


def _linger_zero_count(socket: FakeSocket) -> int:
    return socket.options.count((zmq.LINGER, 0))


def test_timeout_resets_req_before_raising_timeout_error(fake_client):
    old_socket = FakeSocket(recv_result=zmq.Again())
    new_socket = FakeSocket(recv_result=True)
    client, context = fake_client(old_socket, new_socket)

    with pytest.raises(TimeoutError, match="endpoint=reset"):
        client.reset()

    assert client.sock is new_socket
    assert context.created == [old_socket, new_socket]
    assert old_socket.sent == [{"endpoint": "reset", "data": {"options": None}}]
    assert old_socket.closed is True
    assert new_socket.connected == ["tcp://127.0.0.1:5555"]


def test_zmq_error_resets_req_before_raising_runtime_error(fake_client):
    old_socket = FakeSocket(send_error=zmq.ZMQError("transport failed"))
    new_socket = FakeSocket(recv_result=True)
    client, _ = fake_client(old_socket, new_socket)

    with pytest.raises(RuntimeError, match="ZMQ error"):
        client.reset()

    assert client.sock is new_socket
    assert old_socket.sent == [{"endpoint": "reset", "data": {"options": None}}]
    assert old_socket.closed is True


def test_timeout_sets_linger_zero_and_closes_old_socket(fake_client):
    old_socket = FakeSocket(recv_result=zmq.Again())
    new_socket = FakeSocket(recv_result=True)
    client, _ = fake_client(old_socket, new_socket)

    with pytest.raises(TimeoutError):
        client.reset()

    assert _linger_zero_count(old_socket) >= 2
    assert old_socket.closed is True


@pytest.mark.parametrize(
    "method_name,args,expected_request",
    [
        ("ping", (), {"endpoint": "ping"}),
        ("reset", (), {"endpoint": "reset", "data": {"options": None}}),
        (
            "get_action",
            ({"observation/state": [1.0]},),
            {
                "endpoint": "get_action",
                "data": {"observation": {"observation/state": [1.0]}, "options": None},
            },
        ),
    ],
)
def test_failed_ping_reset_and_get_action_each_send_once(
    fake_client, method_name, args, expected_request
):
    old_socket = FakeSocket(recv_result=zmq.Again())
    new_socket = FakeSocket(recv_result=True)
    client, _ = fake_client(old_socket, new_socket)

    if method_name == "ping":
        assert client.ping(*args) is False
    else:
        with pytest.raises(TimeoutError):
            getattr(client, method_name)(*args)

    assert old_socket.sent == [expected_request]
    assert new_socket.sent == []


def test_next_call_after_timeout_uses_new_socket_and_succeeds(fake_client):
    timed_out_socket = FakeSocket(recv_result=zmq.Again())
    recovered_socket = FakeSocket(recv_result={"ok": True})
    client, _ = fake_client(timed_out_socket, recovered_socket)

    with pytest.raises(TimeoutError):
        client.reset()

    assert client.reset() == {"ok": True}
    assert recovered_socket.sent == [{"endpoint": "reset", "data": {"options": None}}]


def test_close_closes_current_socket_and_context_once(fake_client):
    old_socket = FakeSocket(recv_result=zmq.Again())
    current_socket = FakeSocket(recv_result=True)
    client, context = fake_client(old_socket, current_socket)

    with pytest.raises(TimeoutError):
        client.reset()

    client.close()
    client.close()

    assert current_socket.closed is True
    assert context.terminated is True
    assert current_socket.close_calls == 1
    assert context.term_calls == 1
    assert _linger_zero_count(current_socket) >= 2


def test_local_tcp_timeout_then_fresh_request_succeeds_without_efsm():
    ctx = zmq.Context()
    server = ctx.socket(zmq.ROUTER)
    server.setsockopt(zmq.RCVTIMEO, 2000)
    server.setsockopt(zmq.LINGER, 0)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    received = []
    done = threading.Event()

    def server_loop():
        try:
            first_frames = server.recv_multipart()
            first = MsgSerializer.from_bytes(first_frames[-1])
            received.append(first)
            second_frames = server.recv_multipart()
            second = MsgSerializer.from_bytes(second_frames[-1])
            received.append(second)
            server.send_multipart(
                [*second_frames[:-1], MsgSerializer.to_bytes({"ok": second["endpoint"]})]
            )
        finally:
            server.close()
            ctx.term()
            done.set()

    thread = threading.Thread(target=server_loop)
    thread.start()

    client = Pi05ZMQClient(f"127.0.0.1:{port}", timeout_ms=500)
    try:
        with pytest.raises(TimeoutError):
            client.reset({"attempt": 1})

        assert client.reset({"attempt": 2}) == {"ok": "reset"}
    finally:
        client.close()
        assert done.wait(timeout=2.0)
        thread.join(timeout=2.0)

    assert received == [
        {"endpoint": "reset", "data": {"options": {"attempt": 1}}},
        {"endpoint": "reset", "data": {"options": {"attempt": 2}}},
    ]
