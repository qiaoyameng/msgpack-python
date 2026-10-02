import struct

import pytest

import msgpack
from msgpack import (
    DuplicateMapKeyError,
    ExtType,
    deleteb,
    editb,
    insertb,
    replaceb,
    unpackb,
)
from msgpack.fallback import Packer


def test_replacement_preserves_untouched_noncanonical_and_ext_bytes():
    untouched = b"\xa9untouched\x92\xcc\x00\xc7\x05\x7f\xc1\xcc\x00\xd4\x7f"
    packed = b"\x82" + untouched + b"\xa1b\xa1x"

    edited = replaceb(packed, (b"b",), b"z" * 300, raw=True)

    assert edited.startswith(b"\x82" + untouched)
    assert edited[: 1 + len(untouched)] == b"\x82" + untouched
    decoded = unpackb(edited, raw=True, strict_map_key=False)
    assert decoded[b"b"] == b"z" * 300
    assert decoded[b"untouched"] == [0, ExtType(127, b"\xc1\xcc\x00\xd4\x7f")]


def test_fixmap_insertion_rewrites_map_header():
    pairs = [(f"key{i:02d}".encode(), i) for i in range(15)]
    packed = Packer().pack_map_pairs(pairs)
    assert packed[0] == 0x8F

    edited = insertb(packed, (), b"new", 15, raw=True)

    assert edited[:3] == b"\xde\x00\x10"
    assert edited[3 : 3 + len(packed) - 1] == packed[1:]
    decoded = unpackb(edited, raw=True, strict_map_key=False)
    assert len(decoded) == 16
    assert decoded[b"new"] == 15


def test_map16_deletion_rewrites_to_fixmap_header():
    pairs = [(f"key{i:02d}".encode(), i) for i in range(16)]
    packed = Packer().pack_map_pairs(pairs)
    assert packed[:3] == b"\xde\x00\x10"

    edited = deleteb(packed, (b"key15",), raw=True)

    assert edited[0] == 0x8F
    assert edited[1 : 1 + len(packed[3:-8])] == packed[3:-8]
    decoded = unpackb(edited, raw=True, strict_map_key=False)
    assert b"key15" not in decoded
    assert len(decoded) == 15


def test_nested_map_insertion_rewrites_only_target_header():
    prefix = b"\x81\xa5outer\x8f"
    pairs = tuple((f"k{i:02d}", i) for i in range(15))
    inner_pairs = b"".join(Packer().pack(x) for item in pairs for x in item)
    packed = prefix + inner_pairs

    edited = insertb(packed, ("outer",), "new", True)

    assert edited[:10] == b"\x81\xa5outer" + b"\xde\x00\x10"
    assert edited[10 : 10 + len(inner_pairs)] == inner_pairs
    assert unpackb(edited)["outer"]["new"] is True


def test_array_deletion_rewrites_header_boundary():
    packed = Packer().pack_array_header(16) + b"".join(Packer().pack(i) for i in range(16))
    assert packed[:3] == b"\xdc\x00\x10"

    edited = deleteb(packed, (15,))

    assert edited[0] == 0x9F
    assert unpackb(edited, use_list=True) == list(range(15))


def test_duplicate_target_key_is_ambiguous():
    packed = Packer().pack_map_pairs([(b"dup", 1), (b"dup", 2), (b"other", 3)])

    with pytest.raises(DuplicateMapKeyError):
        replaceb(packed, (b"dup",), 4, raw=True)


def test_semantically_duplicate_noncanonical_key_is_ambiguous():
    packed = b"\x82\x00\x01\xcc\x00\x02"

    with pytest.raises(DuplicateMapKeyError):
        replaceb(packed, (0,), 3)


def test_map16_insertion_rewrites_to_map32_header():
    entries = b"".join(
        struct.pack(">BH", 0xCD, index) + b"\x00" for index in range(0xFFFF)
    )
    packed = b"\xde\xff\xff" + entries

    edited = insertb(packed, (), 0xFFFF, 1)

    assert edited[:5] == b"\xdf\x00\x01\x00\x00"
    assert edited[5 : 5 + len(entries)] == entries
    decoded = unpackb(edited, strict_map_key=False)
    assert decoded[0xFFFF] == 1
    assert len(decoded) == 0x10000


def test_duplicate_key_in_unvisited_branch_does_not_make_edit_ambiguous():
    duplicate = Packer().pack_map_pairs([(b"dup", 1), (b"dup", 2)])
    packed = Packer().pack_map_pairs(
        [(b"target", 0), (b"other", msgpack.ExtType(99, duplicate))]
    )

    edited = replaceb(packed, (b"target",), 7, raw=True)

    assert unpackb(edited, raw=True, strict_map_key=False)[b"target"] == 7


def test_ext_payload_is_opaque_and_can_be_replaced():
    opaque_ext = b"\xd4\x7f\xc1"
    packed = b"\x82\xa3ext" + opaque_ext + b"\xa1n\x01"

    edited = replaceb(packed, ("n",), 2)

    assert bytes(edited[5:8]) == opaque_ext
    assert unpackb(edited, strict_map_key=False)["n"] == 2


@pytest.mark.parametrize(
    "packed",
    [
        b"",
        b"\xc1",
        b"\x81\xa1k",
        b"\xc7\x03\x7fab",
        b"\x01\x02",
    ],
)
def test_corrupt_input_raises_without_result(packed):
    with pytest.raises(Exception):
        editb(packed, (), 1)


def test_insert_existing_key_fails():
    packed = Packer().pack_map_pairs([(b"k", 1)])
    with pytest.raises(ValueError):
        insertb(packed, (), b"k", 2, raw=True)


def test_root_cannot_be_deleted():
    with pytest.raises(ValueError):
        deleteb(b"\x01", ())


def test_path_into_scalar_fails():
    with pytest.raises(TypeError):
        replaceb(b"\x01", ("x",), 2)
