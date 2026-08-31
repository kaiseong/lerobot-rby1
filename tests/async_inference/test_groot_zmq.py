import msgpack
import numpy as np
import pytest

from lerobot.async_inference.policy.groot_zmq import MsgSerializer


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
