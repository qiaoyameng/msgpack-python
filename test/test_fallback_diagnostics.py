import pytest

from msgpack import ExtType, FormatError, OutOfData, packb
from msgpack import fallback
from msgpack.exceptions import UnpackDiagnostic


def check_diagnostic(exc, offset, object_start, path):
    diagnostic = exc.unpack_diagnostic
    assert isinstance(diagnostic, UnpackDiagnostic)
    assert diagnostic.offset == offset
    assert exc.absolute_offset == offset
    assert diagnostic.object_start == object_start
    assert exc.object_start == object_start
    assert diagnostic.path == path
    assert exc.path == path
    assert diagnostic.path_text
    assert diagnostic.path_text in exc.__notes__[-1]


def test_invalid_type_diagnostic():
    packed = packb({"a": [None]})[:-1] + b"\xc1"

    with pytest.raises(FormatError) as exc_info:
        fallback.unpackb(packed)

    check_diagnostic(exc_info.value, 4, 0, ("a", 0))


def test_truncation_is_recoverable_with_diagnostic():
    complete = b"\x81\xa1a\xdb\x00\x00\x00\x03abc"
    packed = complete[:-1]

    unpacker = fallback.Unpacker()
    out_of_data_offsets = []
    for chunk in (complete[:5], complete[5:-1], complete[-1:]):
        unpacker.feed(chunk)
        try:
            result = unpacker.unpack()
        except OutOfData as exc:
            out_of_data_offsets.append(exc.absolute_offset)
        else:
            assert result == {"a": "abc"}

    assert out_of_data_offsets == [5, 10]

    with pytest.raises(ValueError) as exc_info:
        fallback.unpackb(packed)

    check_diagnostic(exc_info.value, 10, 0, ("a",))


def test_length_limit_diagnostic():
    packed = packb({"a": "ok"})

    with pytest.raises(ValueError) as exc_info:
        fallback.unpackb(packed, max_str_len=1)

    check_diagnostic(exc_info.value, 3, 0, ("a",))


def test_invalid_utf8_diagnostic():
    packed = b"\x81\xa1a\xda\x00\x03\xffabc"

    with pytest.raises(UnicodeDecodeError) as exc_info:
        fallback.unpackb(packed)

    check_diagnostic(exc_info.value, 6, 0, ("a",))


class HookFailure(Exception):
    pass


def test_hook_failure_preserves_type_and_diagnostic():
    def hook(obj):
        raise HookFailure("hook failed")

    packed = packb({"a": {}})

    with pytest.raises(HookFailure) as exc_info:
        fallback.unpackb(packed, object_hook=hook)

    check_diagnostic(exc_info.value, 3, 0, ("a",))


def fail_hook(*args):
    raise HookFailure("hook failed")


def test_list_hook_failure_diagnostic():
    with pytest.raises(HookFailure) as exc_info:
        fallback.unpackb(packb({"a": [1]}), list_hook=fail_hook)

    check_diagnostic(exc_info.value, 3, 0, ("a",))


def test_ext_hook_failure_diagnostic():
    with pytest.raises(HookFailure) as exc_info:
        fallback.unpackb(packb({"a": ExtType(1, b"x")}), ext_hook=fail_hook)

    check_diagnostic(exc_info.value, 3, 0, ("a",))


def test_object_pairs_hook_failure_diagnostic():
    with pytest.raises(HookFailure) as exc_info:
        fallback.unpackb(packb({"a": {}}), object_pairs_hook=fail_hook)

    check_diagnostic(exc_info.value, 0, 0, ())


def test_path_bounds_long_key_and_deep_object():
    long_key = "k" * 1000
    packed = bytearray(packb({long_key: b"xxx"}))
    packed[-5] = 0xC1
    packed = bytes(packed)

    with pytest.raises(FormatError) as exc_info:
        fallback.unpackb(packed)

    path = exc_info.value.unpack_diagnostic.path
    assert len(path) == 1
    assert len(path[0]) < 100

    deep_packed = b"\x91" * 65 + b"\xc1"
    with pytest.raises(FormatError) as exc_info:
        fallback.unpackb(
            deep_packed,
            max_array_len=100,
        )

    diagnostic = exc_info.value.unpack_diagnostic
    assert len(diagnostic.path) == 64
    assert diagnostic.path[0] == "..."
    assert all(part == 0 for part in diagnostic.path[1:])


@pytest.mark.parametrize("chunk_size", [*range(1, 25), 100])
def test_stream_diagnostics_stable_across_chunks_and_compaction(chunk_size):
    first = packb({"first": 1})
    second = packb(["second"])
    bad = packb({"bad": [None]})[:-1] + b"\xc1"
    packed = first + second + bad

    unpacker = fallback.Unpacker()
    position = 0
    decoded = []

    while position < len(packed):
        unpacker.feed(packed[position : position + chunk_size])
        position += chunk_size
        while True:
            try:
                decoded.append(unpacker.unpack())
            except OutOfData:
                break
            except FormatError as exc_info:
                assert decoded == [{"first": 1}, ["second"]]
                bad_offset = len(first) + len(second)
                check_diagnostic(
                    exc_info,
                    22,
                    bad_offset,
                    ("bad", 0),
                )
                assert packed[exc_info.absolute_offset] == 0xC1
                return

    pytest.fail("invalid record was not detected")
