import io

import pytest

from msgpack import BufferFull, ExtType, OutOfData, packb
from msgpack.fallback import Unpacker


def test_checkpoint_rollback_and_commit():
    data = packb(1) + packb("two") + packb(3)
    unpacker = Unpacker()
    unpacker.feed(data)

    checkpoint = unpacker.checkpoint()
    assert unpacker.unpack() == 1
    assert unpacker.unpack() == "two"
    assert unpacker.tell() == len(packb(1) + packb("two"))

    checkpoint.rollback()
    assert unpacker.tell() == 0
    assert unpacker.unpack() == 1

    checkpoint = unpacker.checkpoint()
    assert unpacker.unpack() == "two"
    checkpoint.commit()
    assert unpacker.unpack() == 3


def test_nested_checkpoints_are_lifo():
    data = packb(1) + packb(2) + packb(3)
    unpacker = Unpacker()
    unpacker.feed(data)

    outer = unpacker.checkpoint()
    assert unpacker.unpack() == 1
    inner = unpacker.checkpoint()
    assert unpacker.unpack() == 2

    with pytest.raises(RuntimeError):
        outer.rollback()
    inner.rollback()
    assert unpacker.tell() == len(packb(1))
    assert unpacker.unpack() == 2

    inner = unpacker.checkpoint()
    inner.commit()
    outer.rollback()
    assert unpacker.tell() == 0
    assert unpacker.unpack() == 1


def test_checkpoint_context_manager():
    unpacker = Unpacker()
    unpacker.feed(packb(1) + packb(2))

    with pytest.raises(OutOfData):
        with unpacker.checkpoint():
            unpacker.unpack()
            unpacker.unpack()
            unpacker.unpack()

    assert unpacker.tell() == 0
    assert unpacker.unpack() == 1

    with unpacker.checkpoint():
        assert unpacker.unpack() == 2
    assert unpacker.tell() == 2


def test_position_entry_points_roll_back():
    data = packb([1, 2]) + packb(3)
    unpacker = Unpacker()
    unpacker.feed(data)

    checkpoint = unpacker.checkpoint()
    assert unpacker.read_array_header() == 2
    assert unpacker.unpack() == 1
    assert unpacker.unpack() == 2
    assert unpacker.tell() == len(packb([1, 2]))
    checkpoint.rollback()
    assert unpacker.tell() == 0

    checkpoint = unpacker.checkpoint()
    unpacker.skip()
    assert unpacker.tell() == len(packb([1, 2]))
    checkpoint.rollback()
    checkpoint = unpacker.checkpoint()
    assert unpacker.read_bytes(1) == data[:1]
    checkpoint.rollback()
    assert unpacker.unpack() == [1, 2]


def test_map_header_entry_point_rolls_back():
    data = packb({"a": 1}) + packb(2)
    unpacker = Unpacker()
    unpacker.feed(data)

    checkpoint = unpacker.checkpoint()
    assert unpacker.read_map_header() == 1
    assert unpacker.unpack() == "a"
    assert unpacker.unpack() == 1
    checkpoint.rollback()

    assert unpacker.tell() == 0
    assert unpacker.unpack() == {"a": 1}
    assert unpacker.unpack() == 2


def test_checkpoint_retains_fed_bytes_within_buffer_limit():
    unpacker = Unpacker(max_buffer_size=5)
    unpacker.feed(b"abc")
    checkpoint = unpacker.checkpoint()
    assert unpacker.unpack() == ord("a")

    unpacker.feed(b"de")
    with pytest.raises(BufferFull):
        unpacker.feed(b"f")

    checkpoint.rollback()
    assert unpacker.tell() == 0
    assert [unpacker.unpack() for _ in range(5)] == [ord(c) for c in "abcde"]


def test_cross_feed_rollback_matches_fresh_unpacker():
    # array16(16), str16("0123456789"), followed by 15 positive ints.
    data = b"\xdc\x00\x10\xda\x00\x0a" + b"0123456789" + bytes(range(15))

    # Split so OutOfData happens inside the container header, then the
    # long-str header, then the long-str payload.
    chunks = [data[:1], data[1:2], data[2:4], data[4:5], data[5:9]]

    trial = Unpacker()
    reference = Unpacker()
    checkpoint = trial.checkpoint()

    for chunk in chunks:
        trial.feed(chunk)
        with pytest.raises(OutOfData):
            trial.unpack()

    checkpoint.rollback()
    remaining = data[9:]
    trial.feed(remaining)

    reference.feed(data)
    assert trial.unpack() == reference.unpack()
    assert trial.tell() == reference.tell()


def test_filelike_checkpoint_rollback():
    data = packb(1) + packb(2)
    unpacker = Unpacker(io.BytesIO(data), read_size=1)

    checkpoint = unpacker.checkpoint()
    assert unpacker.unpack() == 1
    assert unpacker.tell() == 1
    checkpoint.rollback()

    assert unpacker.tell() == 0
    assert unpacker.unpack() == 1
    checkpoint = unpacker.checkpoint()
    assert unpacker.unpack() == 2
    checkpoint.commit()


def test_ext_hook_result_is_reused_after_rollback():
    calls = []

    def ext_hook(code, data):
        calls.append((code, data))
        return {"code": code, "data": data}

    packed = packb(ExtType(7, b"x"))
    unpacker = Unpacker(ext_hook=ext_hook)
    unpacker.feed(packed + packed)

    outer = unpacker.checkpoint()
    first = unpacker.unpack()
    inner = unpacker.checkpoint()
    second = unpacker.unpack()
    inner.rollback()
    assert unpacker.unpack() is second
    outer.rollback()

    assert first == {"code": 7, "data": b"x"}
    assert second == {"code": 7, "data": b"x"}
    assert unpacker.unpack() is first
    assert unpacker.unpack() is second
    assert calls == [(7, b"x"), (7, b"x")]


def test_ext_hook_exception_is_reused_after_rollback():
    def ext_hook(code, data):
        calls.append(data)
        raise RuntimeError("callback side effect")

    calls = []
    unpacker = Unpacker(ext_hook=ext_hook)
    unpacker.feed(packb(ExtType(7, b"x")))

    checkpoint = unpacker.checkpoint()
    with pytest.raises(RuntimeError, match="callback side effect"):
        unpacker.unpack()
    checkpoint.rollback()

    with pytest.raises(RuntimeError, match="callback side effect"):
        unpacker.unpack()
    assert len(calls) == 1
