import gc
import io
import weakref

from pytest import raises

from msgpack import packb, unpackb
from msgpack.fallback import Unpacker


def test_bin_view_returns_readonly_memoryview():
    unpacker = Unpacker(bin_view=True)
    unpacker.feed(packb(b"abc"))
    result = unpacker.unpack()

    assert isinstance(result, memoryview)
    assert result.readonly
    assert result == b"abc"
    assert isinstance(result.obj, bytearray)
    with raises(TypeError):
        result[0] = ord(b"x")


def test_default_mode_continues_returning_bytes():
    unpacker = Unpacker()
    unpacker.feed(packb(b"abc"))
    result = unpacker.unpack()

    assert result == b"abc"
    assert type(result) is bytes


def test_unpackb_supports_bin_view():
    result = unpackb(packb(b"abc"), bin_view=True)

    assert isinstance(result, memoryview)
    assert result.readonly
    assert result == b"abc"
    assert isinstance(result.obj, bytearray)


def test_bin_view_from_mutable_input_is_independent():
    source = bytearray(packb(b"abc"))
    unpacker = Unpacker(bin_view=True)
    unpacker.feed(source)
    result = unpacker.unpack()

    source[1] = ord(b"X")

    assert result == b"abc"
    assert isinstance(result.obj, bytearray)
    assert result.obj is not source


def test_large_backing_buffer_is_copied():
    packed = packb(b"abc") + packb(b"x" * 100)
    unpacker = Unpacker(bin_view=True, max_bin_view_buffer_size=32)
    unpacker.feed(packed)
    result = unpacker.unpack()

    assert result == b"abc"
    assert type(result.obj) is bytes
    assert result.obj is not unpacker._buffer


def test_view_release_allows_backing_storage_reclamation():
    unpacker = Unpacker(bin_view=True, max_bin_view_buffer_size=32)
    unpacker.feed(packb(b"abc"))
    first_view = unpacker.unpack()
    first_storage = weakref.ref(first_view.obj)

    unpacker.feed(packb(b"y" * 100))
    assert first_view == b"abc"
    assert first_view.readonly
    assert first_storage() is not None

    first_view.release()
    unpacker.feed(packb(b"z"))
    gc.collect()
    assert first_storage() is None


def test_retained_view_survives_feeds_unpacks_and_compaction():
    unpacker = Unpacker(bin_view=True, max_bin_view_buffer_size=32)
    unpacker.feed(packb(b"abc"))
    first_view = unpacker.unpack()
    first_storage = weakref.ref(first_view.obj)

    for index in range(10):
        payload = bytes([65 + index]) * 30
        unpacker.feed(packb(payload))
        result = unpacker.unpack()
        assert result == payload
        assert first_view == b"abc"
        assert first_view.readonly
        assert first_storage() is not None

    first_view.release()
    unpacker.feed(packb(b"final"))
    gc.collect()
    assert first_storage() is None
    assert unpacker.unpack() == b"final"


def test_nested_bin_view_survives_parent_message_compaction():
    unpacker = Unpacker(bin_view=True, max_bin_view_buffer_size=64)
    unpacker.feed(packb([b"abc", b"def"]))
    first_view, second_view = unpacker.unpack()

    assert first_view == b"abc"
    assert second_view == b"def"
    unpacker.feed(packb(b"x" * 100))
    assert first_view == b"abc"
    assert second_view == b"def"


def test_zero_length_bin_view():
    unpacker = Unpacker(bin_view=True)
    unpacker.feed(packb(b""))
    result = unpacker.unpack()

    assert isinstance(result, memoryview)
    assert result.readonly
    assert len(result) == 0
    unpacker.feed(packb(b"x" * 100))
    assert result == b""


def test_bin_view_file_stream_survives_refill_compaction():
    stream = io.BytesIO(packb(b"abc") + packb(b"x" * 100))
    unpacker = Unpacker(
        stream,
        read_size=4,
        bin_view=True,
        max_bin_view_buffer_size=32,
    )
    first_view = unpacker.unpack()
    first_storage = weakref.ref(first_view.obj)

    assert first_view == b"abc"
    assert first_storage() is not None
    assert unpacker.unpack() == b"x" * 100
    assert first_view == b"abc"
    assert first_storage() is not None

    first_view.release()
    gc.collect()
    assert first_storage() is None


def test_bin_view_map_keys_remain_hashable_bytes():
    result = unpackb(packb({b"key": b"value"}), bin_view=True, raw=True)
    key = next(iter(result))

    assert key == b"key"
    assert type(key) is bytes
    assert isinstance(result[key], memoryview)


def test_invalid_bin_view_buffer_size():
    with raises(ValueError):
        Unpacker(bin_view=True, max_bin_view_buffer_size=-1)
