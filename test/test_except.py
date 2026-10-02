#!/usr/bin/env python

import datetime
import gc
import tracemalloc

from pytest import raises

from msgpack import (
    ExtType,
    FormatError,
    OutOfData,
    StackError,
    UnpackErrorInfo,
    Unpacker,
    packb,
    unpackb,
)
from msgpack import fallback as pure


class DummyException(Exception):
    pass


def test_raise_on_find_unsupported_value():
    with raises(TypeError):
        packb(datetime.datetime.now())


def test_raise_from_object_hook():
    def hook(obj):
        raise DummyException

    raises(DummyException, unpackb, packb({}), object_hook=hook)
    raises(DummyException, unpackb, packb({"fizz": "buzz"}), object_hook=hook)
    raises(DummyException, unpackb, packb({"fizz": "buzz"}), object_pairs_hook=hook)
    raises(DummyException, unpackb, packb({"fizz": {"buzz": "spam"}}), object_hook=hook)
    raises(
        DummyException,
        unpackb,
        packb({"fizz": {"buzz": "spam"}}),
        object_pairs_hook=hook,
    )

    up = Unpacker(object_hook=hook)

    def up_unpack(x):
        up.feed(x)
        return up.unpack()

    raises(DummyException, up_unpack, packb({}))
    raises(DummyException, up_unpack, packb({"fizz": "buzz"}))
    raises(DummyException, up_unpack, packb({"fizz": "buzz"}))
    raises(DummyException, up_unpack, packb({"fizz": {"buzz": "spam"}}))
    raises(
        DummyException,
        up_unpack,
        packb({"fizz": {"buzz": "spam"}}),
    )


def test_pure_python_structured_error_after_two_records():
    first = pure.Packer().pack({"a": 1})
    second = pure.Packer().pack([10, 20])
    bad = b"\x81\xa3bad\x92\x01\x81\xa1k\xc1"
    data = first + second + bad

    for chunk_size in range(1, len(data) + 1):
        unpacker = pure.Unpacker()
        records = []
        error = None
        for start in range(0, len(data), chunk_size):
            unpacker.feed(data[start : start + chunk_size])
            while True:
                try:
                    records.append(unpacker.unpack())
                except OutOfData:
                    break
                except FormatError as exc:
                    error = exc
                    break
            if error is not None:
                break

        assert records == [{"a": 1}, [10, 20]]
        assert isinstance(error.msgpack_error, UnpackErrorInfo)
        assert error.position == len(first) + len(second) + len(bad) - 1
        assert error.offset == error.position
        assert error.absolute_offset == error.position
        assert error.object_start == len(first) + len(second)
        assert error.object_start_offset == error.object_start
        assert error.path == ("bad", 1, "k")
        assert data[error.position] == 0xC1


def test_pure_python_structured_error_with_buffer_compaction():
    first = packb("x" * 20)
    second = packb("y" * 20)
    bad = b"\x81\xa3bad\x92\x01\x81\xa1k\xc1"
    data = first + second + bad

    unpacker = pure.Unpacker()
    unpacker.feed(data[: len(first) + len(second) + 1])
    assert unpacker.unpack() == "x" * 20
    unpacker.feed(data[len(first) + len(second) + 1 :])
    assert unpacker.unpack() == "y" * 20

    with raises(FormatError) as exc_info:
        unpacker.unpack()

    exc = exc_info.value
    assert exc.position == len(data) - 1
    assert exc.object_start == len(first) + len(second)
    assert exc.path == ("bad", 1, "k")


def test_pure_python_structured_out_of_data_is_recoverable():
    unpacker = pure.Unpacker()
    unpacker.feed(b"\xa3ab")

    with raises(OutOfData) as exc_info:
        unpacker.unpack()

    exc = exc_info.value
    assert exc.position == 3
    assert exc.object_start == 0
    assert exc.path == ()

    unpacker.feed(b"c")
    assert unpacker.unpack() == "abc"
    assert unpacker.tell() == 4


def test_pure_python_structured_errors_are_bounded():
    long_key = "z" * 100
    packed = packb({long_key: b"\xc1"}, use_bin_type=True)
    unpacker = pure.Unpacker(raw=False, max_str_len=10)
    unpacker.feed(packed)

    with raises(ValueError) as exc_info:
        unpacker.unpack()

    path = exc_info.value.path
    assert path == ("<map key>",)

    unpacker = pure.Unpacker(raw=False, max_str_len=200)
    packed_nested = packb({long_key: b""})[:-2] + b"\x91\xc1"
    unpacker.feed(packed_nested)
    with raises(FormatError) as nested_info:
        unpacker.unpack()

    nested_path = nested_info.value.path
    assert nested_path == ("z" * 32 + "...", 0)
    assert len(nested_path[0]) == 35

    nested = b"\x91" * 5000 + b"\xc1"
    with raises(StackError) as stack_info:
        pure.unpackb(nested)

    assert stack_info.value.position == 65
    assert stack_info.value.object_start == 0
    assert len(stack_info.value.path) == 65
    assert stack_info.value.path[-1] == "..."


def test_pure_python_structured_invalid_utf8_and_hooks():
    packed = b"\x81\xa1k\xa2\xc1x"
    with raises(UnicodeDecodeError) as exc_info:
        pure.unpackb(packed)
    exc = exc_info.value
    assert exc.position == 4
    assert exc.object_start == 0
    assert exc.path == ("k",)

    def hook(value):
        raise DummyException

    with raises(DummyException) as hook_info:
        pure.unpackb(packb({"k": [1]}), list_hook=hook)

    assert hook_info.value.position == 3
    assert hook_info.value.object_start == 0
    assert hook_info.value.path == ("k",)


def test_pure_python_structured_hook_exception_without_settable_fields():
    class ReadOnly:
        def __set__(self, instance, value):
            raise TypeError

    class SlottedHookError(Exception):
        msgpack_error = ReadOnly()
        position = ReadOnly()

    def hook(value):
        raise SlottedHookError("boom")

    with raises(SlottedHookError) as exc_info:
        pure.unpackb(packb({"k": [1]}), list_hook=hook)

    assert "msgpack_error" not in exc_info.value.__dict__


def test_pure_python_structured_invalid_map_key_path():
    packed = b"\x81\xa5outer\x81\x01\x01"
    with raises(ValueError) as exc_info:
        pure.unpackb(packed)

    exc = exc_info.value
    assert exc.position == 8
    assert exc.object_start == 0
    assert exc.path == ("outer", "<map key>")


def test_raise_from_list_hook():
    def hook(lst: list) -> list:
        raise DummyException

    with raises(DummyException):
        unpackb(packb([1, 2, 3]), list_hook=hook)

    with raises(DummyException):
        unpacker = Unpacker(list_hook=hook)
        unpacker.feed(packb([1, 2, 3]))
        unpacker.unpack()


def test_raise_from_ext_hook():
    def hook(code: int, data: bytes) -> ExtType:
        raise DummyException

    packed = packb(ExtType(42, b"hello"))

    with raises(DummyException):
        unpackb(packed, ext_hook=hook)

    with raises(DummyException):
        unpacker = Unpacker(ext_hook=hook)
        unpacker.feed(packed)
        unpacker.unpack()


def test_invalidvalue():
    incomplete = b"\xd9\x97#DL_"  # raw8 - length=0x97
    with raises(ValueError):
        unpackb(incomplete)

    with raises(OutOfData):
        unpacker = Unpacker()
        unpacker.feed(incomplete)
        unpacker.unpack()

    with raises(FormatError):
        unpackb(b"\xc1")  # (undefined tag)

    with raises(FormatError):
        unpackb(b"\x91\xc1")  # fixarray(len=1) [ (undefined tag) ]

    with raises(StackError):
        unpackb(b"\x91" * 3000)  # nested fixarray(len=1)

    with raises(StackError):
        unpacker = Unpacker()
        unpacker.feed(b"\x91" * 3000)
        unpacker.skip()


def test_no_memory_leak_on_nested_invalid_tag() -> None:
    """Regression test: unpacking nested arrays containing an invalid tag must not leak objects."""

    kwargs: dict = {
        "raw": False,
        "strict_map_key": False,
        "max_array_len": 1 << 20,
        "max_map_len": 1 << 20,
    }
    n = 1000

    for depth in range(1, 15):
        data = bytes([0x91] * depth + [0xC1])

        gc.collect()
        tracemalloc.start()
        s1 = tracemalloc.take_snapshot()

        for _ in range(n):
            try:
                unpackb(data, **kwargs)
            except Exception:
                pass

        gc.collect()
        s2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        leaked = sum(s.count_diff for s in s2.compare_to(s1, "lineno") if s.count_diff > 0)
        per_call = leaked / n
        assert per_call < 1.0, f"depth={depth}: {per_call:.2f} leaked objects/call (expected < 1)"


def test_strict_map_key():
    valid = {"unicode": 1, b"bytes": 2}
    packed = packb(valid, use_bin_type=True)
    assert valid == unpackb(packed, raw=False, strict_map_key=True)

    invalid = {42: 1}
    packed = packb(invalid, use_bin_type=True)
    with raises(ValueError):
        unpackb(packed, raw=False, strict_map_key=True)


def test_strict_map_key_with_object_pairs_hook():
    # strict_map_key should be enforced even when object_pairs_hook is set
    invalid = {42: "value"}
    packed = packb(invalid, use_bin_type=True)
    with raises(ValueError):
        unpackb(packed, raw=False, strict_map_key=True, object_pairs_hook=list)

    # valid keys (str/bytes) should still work with object_pairs_hook
    valid = {"key": "value"}
    packed = packb(valid, use_bin_type=True)
    result = unpackb(packed, raw=False, strict_map_key=True, object_pairs_hook=list)
    assert result == [("key", "value")]


def test_unpacker_should_not_crash_after_exception():
    up = Unpacker()  # default: strict_map_key=True
    up.feed(b"\x83\x73\xc4\x00")  # fixmap(3): int key (rejected) + empty bin8
    try:
        up.unpack()  # ValueError: int is not allowed for map key ...
    except Exception:
        pass
    up.skip()  # SIGSEGV (resumes from a corrupt parser context)
