import pytest

from msgpack import packb
from msgpack.exceptions import BufferFull, OutOfData, StackError
from msgpack.fallback import Unpacker


def make_unpacker(data, **kwargs):
    unpacker = Unpacker(**kwargs)
    unpacker.feed(data)
    return unpacker


def test_cursor_iterates_items_one_by_one():
    items = [1, "two", [3, 3], {"four": 4}, None]
    unpacker = make_unpacker(packb(items) + packb("tail"))
    cursor = unpacker.array_cursor()
    assert cursor.remaining == 5
    got = []
    for item in cursor:
        got.append(item)
        assert cursor.remaining == 5 - len(got)
    assert got == items
    assert cursor.closed
    # Cursor detached itself once exhausted; the unpacker is usable again.
    assert unpacker.unpack() == "tail"


def test_cursor_resumes_after_feed():
    data = packb(["hello", "world"])
    unpacker = Unpacker()
    unpacker.feed(data[:2])  # header + start of first str
    cursor = unpacker.array_cursor()
    with pytest.raises(OutOfData):
        next(cursor)
    assert cursor.remaining == 2
    unpacker.feed(data[2:7])
    assert next(cursor) == "hello"
    with pytest.raises(OutOfData):
        next(cursor)
    unpacker.feed(data[7:])
    assert next(cursor) == "world"
    with pytest.raises(StopIteration):
        next(cursor)


def test_cursor_rejects_other_operations():
    unpacker = make_unpacker(packb([1, 2, 3]))
    cursor = unpacker.array_cursor()
    with pytest.raises(RuntimeError):
        unpacker.unpack()
    with pytest.raises(RuntimeError):
        unpacker.skip()
    with pytest.raises(RuntimeError):
        next(iter(unpacker))
    with pytest.raises(RuntimeError):
        unpacker.read_array_header()
    with pytest.raises(RuntimeError):
        unpacker.read_map_header()
    with pytest.raises(RuntimeError):
        unpacker.read_bytes(1)
    with pytest.raises(RuntimeError):
        unpacker.array_cursor()
    # The cursor itself still works.
    assert next(cursor) == 1
    assert list(cursor) == [2, 3]
    assert unpacker.tell() == 4


def test_close_drains_then_reads_next_message():
    items = ["\x91\x92", [1, [2, 3]], {"k": "v"}, b"\xc0\xff", 5]
    next_msg = {"next": [1, 2, 3]}
    unpacker = make_unpacker(packb(items) + packb(next_msg))
    cursor = unpacker.array_cursor()
    assert next(cursor) == "\x91\x92"
    assert next(cursor) == [1, [2, 3]]
    cursor.close()
    assert cursor.closed
    # Remaining items were skipped by real parsing, so the boundary is exact.
    assert unpacker.unpack() == next_msg
    with pytest.raises(OutOfData):
        unpacker.unpack()


def test_close_drain_waits_for_more_data():
    data = packb([1, 2, 3, 4]) + packb("next")
    unpacker = Unpacker()
    unpacker.feed(data[:3])  # header + two items
    cursor = unpacker.array_cursor()
    assert next(cursor) == 1
    with pytest.raises(OutOfData):
        cursor.close()
    assert not cursor.closed
    unpacker.feed(data[3:])
    cursor.close()
    assert cursor.closed
    assert unpacker.unpack() == "next"


def test_close_without_drain_requires_reset():
    data = packb([1, 2, 3]) + packb("next")
    unpacker = make_unpacker(data)
    cursor = unpacker.array_cursor()
    assert next(cursor) == 1
    cursor.close(drain=False)
    with pytest.raises(RuntimeError):
        unpacker.unpack()
    with pytest.raises(RuntimeError):
        unpacker.array_cursor()
    unpacker.reset()
    assert unpacker.tell() == 0
    unpacker.feed(packb("next"))
    assert unpacker.unpack() == "next"


def test_context_manager_drains_by_default():
    unpacker = make_unpacker(packb([1, 2, 3]) + packb("next"))
    with unpacker.array_cursor() as cursor:
        assert next(cursor) == 1
    assert cursor.closed
    assert unpacker.unpack() == "next"


def test_context_manager_reset_policy():
    unpacker = make_unpacker(packb([1, 2, 3]) + packb("next"))
    with unpacker.array_cursor(drain_on_close=False) as cursor:
        assert next(cursor) == 1
    with pytest.raises(RuntimeError):
        unpacker.unpack()
    unpacker.reset()
    unpacker.feed(packb("next"))
    assert unpacker.unpack() == "next"


def test_close_drain_respects_depth_limit():
    nested = b"\x91" * 2000 + b"\xc0"
    unpacker = make_unpacker(b"\x92\x01" + nested)
    cursor = unpacker.array_cursor()
    assert next(cursor) == 1
    with pytest.raises(StackError):
        cursor.close()


def test_cursor_item_respects_depth_limit():
    nested = b"\x91" * 2000 + b"\xc0"
    unpacker = make_unpacker(b"\x91" + nested)
    cursor = unpacker.array_cursor()
    with pytest.raises(StackError):
        next(cursor)


def test_cursor_respects_buffer_limit():
    unpacker = make_unpacker(packb([1, 2, 3]), max_buffer_size=4)
    cursor = unpacker.array_cursor()
    with pytest.raises(BufferFull):
        unpacker.feed(b"x" * 100)
    assert next(cursor) == 1


def test_empty_array_cursor():
    unpacker = make_unpacker(packb([]) + packb(42))
    cursor = unpacker.array_cursor()
    assert cursor.remaining == 0
    with pytest.raises(StopIteration):
        next(cursor)
    assert unpacker.unpack() == 42


def test_array_cursor_on_non_array():
    unpacker = make_unpacker(packb({"a": 1}))
    with pytest.raises(ValueError):
        unpacker.array_cursor()
