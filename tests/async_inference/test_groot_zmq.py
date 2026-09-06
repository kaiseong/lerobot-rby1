import threading
import time

import msgpack
import numpy as np
import pytest
import zmq

from lerobot.async_inference.policy.groot_zmq import GR00TZMQClient, MsgSerializer


def _msgpack_numpy_payload(array: np.ndarray) -> dict:
    return {
        b"nd": True,
        b"type": array.dtype.str,
        b"kind": b"",
        b"shape": array.shape,
        b"data": array.tobytes(),
    }


def test_msg_serializer_decodes_groot_n17_numpy_actions() -> None:
    expected = np.arange(28, dtype=np.float32).reshape(1, 4, 7)
    response = [{"left_arm": _msgpack_numpy_payload(expected)}, {}]

    decoded = MsgSerializer.from_bytes(msgpack.packb(response))

    assert isinstance(decoded[0]["left_arm"], np.ndarray)
    np.testing.assert_array_equal(decoded[0]["left_arm"], expected)


def test_msg_serializer_preserves_legacy_numpy_round_trip() -> None:
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)

    decoded = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"action": expected}))

    np.testing.assert_array_equal(decoded["action"], expected)


def test_msg_serializer_rejects_object_dtype_payload() -> None:
    payload = {
        b"nd": True,
        b"type": "|O",
        b"kind": b"O",
        b"shape": [1],
        b"data": b"not-a-safe-object-array",
    }

    with pytest.raises(ValueError, match="object-dtype"):
        MsgSerializer.from_bytes(msgpack.packb(payload))


def test_client_recovers_after_response_timeout() -> None:
    server_context = zmq.Context()
    server = server_context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    port = server.bind_to_random_port("tcp://127.0.0.1")

    def serve() -> None:
        first_request = MsgSerializer.from_bytes(server.recv())
        assert first_request["endpoint"] == "first"
        time.sleep(0.1)
        server.send(MsgSerializer.to_bytes({"late": True}))

        second_request = MsgSerializer.from_bytes(server.recv())
        assert second_request["endpoint"] == "second"
        server.send(MsgSerializer.to_bytes({"ok": True}))

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    client = GR00TZMQClient(f"127.0.0.1:{port}", timeout_ms=40)
    original_socket = client.sock

    try:
        with pytest.raises(TimeoutError, match=r"endpoint=first"):
            client._call("first")

        assert client.sock is not original_socket
        assert client.sock.getsockopt(zmq.RCVTIMEO) == 40
        assert client.sock.getsockopt(zmq.SNDTIMEO) == 40
        assert client.sock.getsockopt(zmq.LINGER) == 0

        time.sleep(0.08)
        assert client._call("second") == {"ok": True}
    finally:
        client.close()
        server_thread.join(timeout=1)
        server.close()
        server_context.term()

    assert not server_thread.is_alive()


def test_client_recovers_socket_and_preserves_zmq_error() -> None:
    client = GR00TZMQClient("127.0.0.1:1", timeout_ms=25)
    original_socket = client.sock

    class BrokenSocket:
        closed = False

        def send(self, _request):
            raise zmq.ZMQError(zmq.EFSM)

        def close(self, linger=None):
            assert linger == 0
            self.closed = True

    broken_socket = BrokenSocket()
    original_socket.close(linger=0)
    client.sock = broken_socket

    try:
        with pytest.raises(RuntimeError, match="ZMQ error:.*current state"):
            client._call("get_action")

        assert broken_socket.closed
        assert client.sock is not broken_socket
        assert client.sock.getsockopt(zmq.RCVTIMEO) == 25
        assert client.sock.getsockopt(zmq.SNDTIMEO) == 25
        assert client.sock.getsockopt(zmq.LINGER) == 0
    finally:
        client.close()
