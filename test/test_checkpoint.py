"""Tests for explicit, bounded Unpacker checkpoints (pure Python fallback)."""

import io

import pytest

from msgpack import packb, ExtType
from msgpack.exceptions import BufferFull, OutOfData
from msgpack.fallback import Unpacker


def _drain(unpacker):
    out = []
    while True:
        try:
            out.append(unpacker.unpack())
        except OutOfData:
            return out


def test_checkpoint_rollback_allows_trial_read_across_feeds():
    data = packb([1, "abcdefghij", {"k": b"v" * 50}]) + packb("z" * 200)

    # The same split point, with and without a trial read, must agree.
    saw_short_read = False
    for split in range(1, len(data)):
        tried = Unpacker()
        ref = Unpacker()
        tried.feed(data[:split])
        cp = tried.checkpoint()
        try:
            tried.unpack()
            incomplete = False
        except OutOfData:
            incomplete = True
            saw_short_read = True
        # Bytes fed during the checkpoint survive the rollback.
        tried.rollback(cp)
        assert tried.tell() == 0
        tried.feed(data[split:])

        ref.feed(data)
        assert _drain(tried) == _drain(ref)
        assert tried.tell() == ref.tell() == len(data)
        if not incomplete:
            # The first object fit entirely in the prefix; the split still
            # has to leave the *second* object short to exercise OutOfData.
            continue
    assert saw_short_read


def test_trial_read_succeeds_then_is_rolled_back():
    data = packb(1) + packb("two") + packb([3])
    unpacker = Unpacker()
    unpacker.feed(data)

    cp = unpacker.checkpoint()
    assert unpacker.unpack() == 1
    assert unpacker.tell() == 1
    unpacker.rollback(cp)
    assert unpacker.tell() == 0
    assert _drain(unpacker) == [1, "two", [3]]


def test_checkpoint_commit_keeps_position():
    unpacker = Unpacker()
    unpacker.feed(packb(1) + packb(2))
    cp = unpacker.checkpoint()
    assert unpacker.unpack() == 1
    unpacker.commit(cp)
    assert unpacker.tell() == 1
    assert unpacker.unpack() == 2
    assert unpacker.tell() == 2


def test_nested_checkpoints_scope_and_lifo():
    unpacker = Unpacker()
    unpacker.feed(packb(1) + packb(2) + packb(3))

    outer = unpacker.checkpoint()
    assert unpacker.unpack() == 1
    inner = unpacker.checkpoint()
    assert unpacker.unpack() == 2

    # Releasing the outer checkpoint while the inner one is alive is invalid.
    with pytest.raises(RuntimeError):
        unpacker.commit(outer)

    unpacker.rollback(inner)
    assert unpacker.tell() == 1  # back to the outer checkpoint's position
    assert unpacker.unpack() == 2
    unpacker.rollback(outer)
    assert unpacker.tell() == 0
    assert _drain(unpacker) == [1, 2, 3]


def test_released_checkpoint_cannot_be_reused():
    unpacker = Unpacker()
    cp = unpacker.checkpoint()
    unpacker.rollback(cp)
    with pytest.raises(RuntimeError):
        unpacker.rollback(cp)
    with pytest.raises(RuntimeError):
        unpacker.commit(cp)


def test_checkpoint_context_manager():
    unpacker = Unpacker()
    unpacker.feed(packb(1) + packb(2))

    with pytest.raises(RuntimeError, match="boom"):
        with unpacker.checkpoint():
            assert unpacker.unpack() == 1
            raise RuntimeError("boom")
    assert unpacker.tell() == 0
    assert unpacker.unpack() == 1

    with unpacker.checkpoint():
        assert unpacker.unpack() == 2
    assert unpacker.tell() == 2


def test_tell_skip_and_headers_obey_checkpoint():
    data = packb([1, 2, 3]) + packb({"a": 1}) + packb("tail")
    unpacker = Unpacker()
    unpacker.feed(data)

    cp = unpacker.checkpoint()
    assert unpacker.read_array_header() == 3
    assert unpacker.tell() == 1
    unpacker.skip()
    unpacker.skip()
    unpacker.skip()
    assert unpacker.tell() == 4
    assert unpacker.read_map_header() == 1
    unpacker.skip()
    unpacker.skip()
    unpacker.rollback(cp)
    assert unpacker.tell() == 0
    assert _drain(unpacker) == [[1, 2, 3], {"a": 1}, "tail"]


def test_read_bytes_respects_checkpoint():
    unpacker = Unpacker()
    unpacker.feed(b"\x01\x02\x03\x04")
    cp = unpacker.checkpoint()
    assert unpacker.unpack() == 1
    assert unpacker.read_bytes(2) == b"\x02\x03"
    assert unpacker.tell() == 3
    unpacker.rollback(cp)
    assert unpacker.tell() == 0
    assert unpacker.read_bytes(4) == b"\x01\x02\x03\x04"
    assert unpacker.tell() == 4


def test_incomplete_header_and_long_string_across_many_feeds():
    long_str = "q" * 300
    data = packb([long_str]) + packb(123456789)

    tried = Unpacker()
    ref = Unpacker()
    cp = tried.checkpoint()

    fed = 0
    short_reads = 0
    chunks = [data[i : i + 3] for i in range(0, len(data), 3)]
    for chunk in chunks:
        tried.feed(chunk)
        fed += 1
        try:
            tried.unpack()
            break
        except OutOfData:
            short_reads += 1
    assert short_reads > 0

    tried.rollback(cp)
    assert tried.tell() == 0
    tried.feed(b"".join(chunks[fed:]))  # only chunks not fed yet
    ref.feed(data)
    assert _drain(tried) == _drain(ref)
    assert fed > 1


def test_retained_window_is_bounded_not_the_whole_stream():
    unpacker = Unpacker(max_buffer_size=8)
    cp = unpacker.checkpoint()
    unpacker.feed(b"\xa5hello")  # six bytes retained by the checkpoint
    assert unpacker.unpack() == "hello"

    # The overflow is reported at the feed boundary; nothing is mutated.
    with pytest.raises(BufferFull):
        unpacker.feed(b"XXXXX")
    assert unpacker.tell() == 6
    unpacker.rollback(cp)
    assert unpacker.tell() == 0

    # Six unread bytes: exactly two more fit (8 total) while retained.
    unpacker.feed(b"01")
    # While the checkpoint is alive the window stays pinned at 8 bytes.
    with pytest.raises(BufferFull):
        unpacker.feed(b"2")

    # The rollback already released the checkpoint; a fresh checkpoint
    # commits normally and the pending tail keeps flowing.
    fresh = unpacker.checkpoint()
    unpacker.commit(fresh)
    assert unpacker.unpack() == "hello"
    assert unpacker.unpack() == 0x30
    assert unpacker.unpack() == 0x31
    unpacker.feed(packb(42))
    assert unpacker.unpack() == 42


def test_ext_hook_not_reinvoked_after_rollback():
    calls = []

    def hook(code, data):
        calls.append((code, bytes(data)))
        return {"code": code, "data": bytes(data)}

    unpacker = Unpacker(ext_hook=hook)
    unpacker.feed(packb(ExtType(7, b"xy")))
    cp = unpacker.checkpoint()
    first = unpacker.unpack()
    unpacker.rollback(cp)
    second = unpacker.unpack()

    assert first == second == {"code": 7, "data": b"xy"}
    assert first is second  # cached result is reused
    assert calls == [(7, b"xy")]


def test_ext_hook_dedup_inside_container_with_nested_checkpoint():
    calls = []
    unpacker = Unpacker(
        ext_hook=lambda c, d: calls.append((c, bytes(d))) or (c, bytes(d))
    )
    ext = ExtType(9, b"zz")
    unpacker.feed(packb([ext, ext]) + packb(7))

    outer = unpacker.checkpoint()
    inner = unpacker.checkpoint()
    assert unpacker.unpack() == [(9, b"zz"), (9, b"zz")]
    assert len(calls) == 2
    unpacker.commit(inner)

    unpacker.rollback(outer)
    assert unpacker.unpack() == [(9, b"zz"), (9, b"zz")]
    assert calls == [(9, b"zz"), (9, b"zz")]


def test_ext_cache_dropped_when_bytes_are_compacted():
    unpacker = Unpacker(ext_hook=lambda c, d: (c, bytes(d)))
    unpacker.feed(packb(ExtType(3, b"q")) + packb(b"pad"))
    cp = unpacker.checkpoint()
    unpacker.unpack()
    unpacker.commit(cp)
    assert unpacker._ext_cache
    unpacker.feed(b"")  # triggers buffer compaction
    assert not unpacker._ext_cache


def test_file_like_checkpoint_trial_and_rollback():
    data = packb([1, "abcdefghij", {"k": b"v" * 50}]) + packb("z" * 200)

    tried = Unpacker(io.BytesIO(data), read_size=4, max_buffer_size=1024)
    ref = Unpacker(io.BytesIO(data), read_size=4, max_buffer_size=1024)

    cp = tried.checkpoint()
    first = tried.unpack()
    tried.rollback(cp)
    assert tried.tell() == 0
    assert tried.unpack() == first
    assert tried.unpack() == "z" * 200
    assert tried.tell() == len(data)

    assert ref.unpack() == first
    assert ref.unpack() == "z" * 200
    assert ref.tell() == len(data)
