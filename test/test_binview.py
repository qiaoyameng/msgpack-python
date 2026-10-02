import gc
import platform
import sys

import pytest

from msgpack import ExtType, packb
from msgpack.fallback import Unpacker, unpackb


def test_bin_view_mode_returns_readonly_view():
    packed = packb({"payload": b"\x01\x02\x03", "name": "abc"})
    obj = unpackb(packed, bin_views=True)
    payload = obj["payload"]
    assert isinstance(payload, memoryview)
    assert payload.readonly
    assert bytes(payload) == b"\x01\x02\x03"
    assert obj["name"] == "abc"
    with pytest.raises(TypeError):
        payload[0] = 0


def test_default_mode_still_returns_bytes():
    obj = unpackb(packb([b"abc", b"def"]))
    assert type(obj[0]) is bytes
    assert type(obj[1]) is bytes


def test_views_share_snapshot_within_retain_limit():
    unpacker = Unpacker(bin_views=True)
    unpacker.feed(packb([b"aaaa", b"bbbb", 1]))
    a, b, _ = unpacker.unpack()
    assert bytes(a) == b"aaaa"
    assert bytes(b) == b"bbbb"
    # consecutive fields share one snapshot instead of copying per field
    assert a.obj is b.obj


def test_view_immune_to_fed_bytearray_mutation():
    buf = bytearray(packb({"blob": b"\x01\x02\x03\x04"}))
    unpacker = Unpacker(bin_views=True)
    unpacker.feed(buf)
    view = unpacker.unpack()["blob"]
    buf[:] = b"\x00" * len(buf)
    assert bytes(view) == b"\x01\x02\x03\x04"


def test_max_view_retain_caps_retained_buffer():
    msg = packb({"blob": b"\x01" * 16})
    unpacker = Unpacker(bin_views=True, max_view_retain=64)
    unpacker.feed(msg + b"\xc0" * 10000)  # long tail stays buffered
    view = unpacker.unpack()["blob"]
    assert bytes(view) == b"\x01" * 16
    # only the field itself is retained, not the whole buffered stream
    assert len(view.obj) == 16


def test_max_view_retain_must_be_non_negative():
    with pytest.raises(ValueError):
        Unpacker(bin_views=True, max_view_retain=-1)


def test_raw_and_ext_are_unaffected():
    unpacker = Unpacker(bin_views=True, raw=True)
    unpacker.feed(packb("hello") + packb(ExtType(1, b"ab")))
    assert type(unpacker.unpack()) is bytes
    ext = unpacker.unpack()
    assert type(ext.data) is bytes
    assert ext.data == b"ab"


def test_view_lifecycle_across_stream_compaction():
    first_blob = bytes(range(256)) * 4
    unpacker = Unpacker(bin_views=True)
    unpacker.feed(packb({"seq": 0, "blob": first_blob}))
    view = unpacker.unpack()["blob"]
    base = view.obj

    # Keep feeding and unpacking; every feed compacts the consumed prefix
    # of the internal buffer.
    for i in range(1, 50):
        unpacker.feed(packb({"seq": i, "blob": b"\x00" * 2048}))
        msg = unpacker.unpack()
        assert msg["seq"] == i
        assert bytes(msg["blob"]) == b"\x00" * 2048

    # The view of the first message is still intact and read-only.
    assert bytes(view) == first_blob
    assert view.readonly
    with pytest.raises(TypeError):
        view[0] = 0

    # Releasing the view lets the underlying storage be reclaimed.
    view.release()
    del view
    gc.collect()
    if platform.python_implementation() == "CPython":
        # only this local reference (and the getrefcount argument) remain
        assert sys.getrefcount(base) == 2
    del base
    gc.collect()


def test_empty_bin_view():
    obj = unpackb(packb(b""), bin_views=True)
    assert isinstance(obj, memoryview)
    assert len(obj) == 0
    assert bytes(obj) == b""
