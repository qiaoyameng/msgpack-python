import pytest

from msgpack import packb
from msgpack.exceptions import BufferFull, OutOfData, StackError
from msgpack.fallback import Unpacker


def test_cursor_iterates_items_one_by_one():
    records = [1, "two", [3], {"four": 4}, None, b"\x00\x01"]
    unpacker = Unpacker()
    unpacker.feed(packb(records))
    cursor = unpacker.array_cursor()
    assert cursor.remaining == len(records)
    out = []
    for item in cursor:
        out.append(item)
        assert cursor.remaining == len(records) - len(out)
    assert out == records
    assert cursor.closed
    # Unpacker is usable again once the array is exhausted.
    with pytest.raises(OutOfData):
        unpacker.unpack()


def test_cursor_feed_incrementally():
    records = [1, "hello", [1, 2], {"k": "v"}]
    data = packb(records)
    unpacker = Unpacker()
    cursor = None
    out = []
    for i in range(len(data)):
        unpacker.feed(data[i : i + 1])
        if cursor is None:
            try:
                cursor = unpacker.array_cursor()
            except OutOfData:
                continue
        while True:
            try:
                out.append(next(cursor))
            except OutOfData:
                break
            except StopIteration:
                break
    assert out == records


def test_cursor_rejects_other_operations_while_alive():
    unpacker = Unpacker()
    unpacker.feed(packb([1, 2, 3]))
    cursor = unpacker.array_cursor()
    assert next(cursor) == 1
    with pytest.raises(RuntimeError):
        unpacker.unpack()
    with pytest.raises(RuntimeError):
        unpacker.skip()
    with pytest.raises(RuntimeError):
        unpacker.read_array_header()
    with pytest.raises(RuntimeError):
        unpacker.read_map_header()
    with pytest.raises(RuntimeError):
        unpacker.read_bytes(1)
    with pytest.raises(RuntimeError):
        next(unpacker)
    with pytest.raises(RuntimeError):
        unpacker.array_cursor()
    cursor.close()
    # After draining close the unpacker coordinates again.
    with pytest.raises(OutOfData):
        unpacker.unpack()


def test_cursor_not_an_array():
    unpacker = Unpacker()
    unpacker.feed(packb(42))
    with pytest.raises(ValueError):
        unpacker.array_cursor()


def test_close_drain_then_read_next_message():
    first = packb([1, 2, 3, 4, 5])
    second = packb({"next": "message"})
    unpacker = Unpacker()
    unpacker.feed(first + second)
    cursor = unpacker.array_cursor()
    assert [next(cursor), next(cursor)] == [1, 2]
    cursor.close()
    assert cursor.closed
    assert unpacker.tell() == len(first)
    assert unpacker.unpack() == {"next": "message"}


def test_close_without_drain_requires_reset():
    first = packb([1, 2, 3, 4, 5])
    second = packb("next")
    unpacker = Unpacker()
    unpacker.feed(first + second)
    cursor = unpacker.array_cursor()
    assert [next(cursor), next(cursor)] == [1, 2]
    cursor.close(drain=False)
    assert cursor.closed
    with pytest.raises(RuntimeError):
        unpacker.unpack()
    with pytest.raises(RuntimeError):
        unpacker.skip()
    with pytest.raises(RuntimeError):
        unpacker.array_cursor()
    unpacker.reset()
    unpacker.feed(second)
    assert unpacker.unpack() == "next"


def test_close_drain_waits_for_more_input():
    first = packb([1, 2, 3, 4])
    second = packb("tail")
    unpacker = Unpacker()
    unpacker.feed(first[:3])  # header + 2 items
    cursor = unpacker.array_cursor()
    assert [next(cursor), next(cursor)] == [1, 2]
    with pytest.raises(OutOfData):
        cursor.close()
    assert not cursor.closed  # retriable after feeding more data
    unpacker.feed(first[3:] + second)
    cursor.close()
    assert cursor.closed
    assert unpacker.unpack() == "tail"


def test_close_drain_respects_depth_limit():
    # Outer array of 2 items; the second is nested far beyond the
    # recursion limit, so draining it must raise StackError.
    data = b"\x92\x01" + b"\x91" * 3000 + b"\xc0"
    unpacker = Unpacker()
    unpacker.feed(data)
    cursor = unpacker.array_cursor()
    assert next(cursor) == 1
    with pytest.raises(StackError):
        cursor.close()
    # The buffer position is no longer reliable: reset is required.
    with pytest.raises(RuntimeError):
        unpacker.unpack()
    unpacker.reset()
    unpacker.feed(packb("ok"))
    assert unpacker.unpack() == "ok"


def test_cursor_respects_buffer_limit():
    unpacker = Unpacker(max_buffer_size=8)
    cursor_data = packb([1, 2, 3])
    unpacker.feed(cursor_data)
    cursor = unpacker.array_cursor()
    assert next(cursor) == 1
    with pytest.raises(BufferFull):
        unpacker.feed(b"x" * 8)
    cursor.close()


def test_cursor_exact_boundaries():
    records = [
        "str",
        b"\xde\xad",
        [1, [2, 3]],
        {"a": (1, 2)},
        -42,
        1.5,
        True,
    ]
    first = packb(records)
    second = packb([["exact"], {"boundaries": 1}])
    unpacker = Unpacker()
    unpacker.feed(first + second)
    with unpacker.array_cursor() as cursor:
        assert next(cursor) == "str"
        cursor.skip()  # skip b"\xde\xad" without constructing it
        assert next(cursor) == [1, [2, 3]]
    # Context-manager exit drained the remaining items exactly.
    assert unpacker.tell() == len(first)
    assert unpacker.unpack() == [["exact"], {"boundaries": 1}]
    assert unpacker.tell() == len(first) + len(second)


def test_exhausted_cursor_detaches_automatically():
    unpacker = Unpacker()
    unpacker.feed(packb([1]) + packb(2))
    cursor = unpacker.array_cursor()
    assert next(cursor) == 1
    assert cursor.closed
    assert unpacker.unpack() == 2
