import pytest

import msgpack
from msgpack import ExtType
from msgpack.binary_edit import (
    AmbiguousKeyError,
    DuplicateKeyError,
    InvalidPathError,
    PathNotFoundError,
    delete_at,
    insert_at,
    replace_at,
)
from msgpack.exceptions import FormatError


def unpack(data):
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def test_replace_preserves_untouched_handwritten_bytes():
    untouched = (
        b"\xa1a"  # "a"
        b"\xce\x00\x00\x00\x2a"  # 42 as uint32 (non-shortest encoding)
        b"\xa1b"  # "b"
        b"\xc7\x04\x63\xde\xad\xbe\xef"  # unknown ext type 0x63, opaque payload
    )
    doc = b"\x83" + untouched + b"\xa1c\x81\xa1d\x01"
    out = replace_at(doc, ["c", "d"], 255)
    assert untouched in out
    assert out.startswith(b"\x83" + untouched)
    assert unpack(out) == {
        "a": 42,
        "b": ExtType(0x63, b"\xde\xad\xbe\xef"),
        "c": {"d": 255},
    }


def test_non_shortest_ancestor_header_preserved():
    doc = b"\xde\x00\x02\xa1a\x01\xa1b\x02"  # map16 header for 2 entries
    out = replace_at(doc, ["a"], 9)
    assert out == b"\xde\x00\x02\xa1a\x09\xa1b\x02"


def test_ext_payload_is_opaque():
    payload = b"\x81\xa1a\x01"  # looks like {"a": 1} but must not be traversed
    doc = b"\x82\xa1e\xc7\x04\x09" + payload + b"\xa1b\x02"
    out = replace_at(doc, ["b"], 3)
    assert payload in out
    assert unpack(out) == {"e": ExtType(9, payload), "b": 3}
    with pytest.raises(InvalidPathError):
        replace_at(doc, ["e", "a"], 9)


def test_insert_rewrites_fixmap_to_map16():
    base = {"k%02d" % i: i for i in range(15)}
    doc = msgpack.packb(base)
    assert doc[0] == 0x8F
    out = insert_at(doc, [], "k15", 15)
    assert out[:3] == b"\xde\x00\x10"
    assert unpack(out) == dict(base, k15=15)
    back = delete_at(out, ["k15"])
    assert back[0] == 0x8F
    assert unpack(back) == base


def test_nested_header_rewrite_keeps_outer_header():
    inner = {"k%02d" % i: i for i in range(15)}
    doc = msgpack.packb({"items": inner, "other": [1, 2, 3]})
    assert doc[0] == 0x82
    out = insert_at(doc, ["items"], "k15", 15)
    assert out[0] == 0x82  # outer fixmap header untouched
    assert unpack(out) == {"items": dict(inner, k15=15), "other": [1, 2, 3]}


def test_delete_rewrites_array16_to_fixarray():
    doc = msgpack.packb(list(range(16)))
    assert doc[0] == 0xDC
    out = delete_at(doc, [0])
    assert out[0] == 0x9F
    assert unpack(out) == list(range(1, 16))


def test_delete_map_entry():
    doc = msgpack.packb({"a": 1, "b": 2, "c": 3})
    out = delete_at(doc, ["b"])
    assert out[0] == 0x82
    assert unpack(out) == {"a": 1, "c": 3}


def test_delete_array_element():
    out = delete_at(msgpack.packb([1, 2, 3]), [1])
    assert unpack(out) == [1, 3]


def test_duplicate_keys_are_ambiguous():
    doc = b"\x82\xa1a\x01\xa1a\x02"
    with pytest.raises(AmbiguousKeyError):
        replace_at(doc, ["a"], 3)
    with pytest.raises(AmbiguousKeyError):
        delete_at(doc, ["a"])


def test_nested_duplicate_keys_are_ambiguous():
    doc = b"\x81\xa1x\x82\xa1a\x01\xa1a\x02"
    with pytest.raises(AmbiguousKeyError):
        replace_at(doc, ["x", "a"], 3)


def test_insert_existing_key_raises():
    with pytest.raises(DuplicateKeyError):
        insert_at(msgpack.packb({"a": 1}), [], "a", 2)


def test_insert_into_nested_map():
    doc = msgpack.packb({"outer": {"a": 1}})
    out = insert_at(doc, ["outer"], "b", 2)
    assert unpack(out) == {"outer": {"a": 1, "b": 2}}


def test_insert_into_non_map_raises():
    with pytest.raises(InvalidPathError):
        insert_at(msgpack.packb([1]), [], "k", 1)


def test_missing_key_raises():
    with pytest.raises(PathNotFoundError):
        replace_at(msgpack.packb({"a": 1}), ["b"], 2)


def test_index_out_of_range_raises():
    with pytest.raises(PathNotFoundError):
        delete_at(msgpack.packb([1, 2]), [5])


def test_negative_index():
    out = replace_at(msgpack.packb([1, 2, 3]), [-1], 9)
    assert unpack(out) == [1, 2, 9]


def test_path_through_scalar_raises():
    with pytest.raises(InvalidPathError):
        replace_at(msgpack.packb({"a": 1}), ["a", "b"], 2)


def test_replace_root_and_delete_root():
    out = replace_at(msgpack.packb({"a": 1}), [], [1, 2])
    assert unpack(out) == [1, 2]
    with pytest.raises(InvalidPathError):
        delete_at(msgpack.packb({"a": 1}), [])


def test_integer_map_keys():
    doc = msgpack.packb({1: "a", 2: "b"})
    out = replace_at(doc, [2], "z")
    assert unpack(out) == {1: "a", 2: "z"}


def test_deeply_nested_replace():
    doc = msgpack.packb({"a": [{"b": [1, 2, {"c": "old"}]}]})
    out = replace_at(doc, ["a", 0, "b", 2, "c"], "new")
    assert unpack(out) == {"a": [{"b": [1, 2, {"c": "new"}]}]}


@pytest.mark.parametrize(
    "doc",
    [
        b"",  # empty
        b"\x82\xa1a",  # truncated map
        b"\xc1",  # invalid marker
        b"\x01\xff",  # trailing garbage
        b"\xa5ab",  # truncated str
        b"\xc7\x0a\x01\x00",  # truncated ext payload
    ],
)
def test_corrupt_input_raises(doc):
    with pytest.raises(FormatError):
        replace_at(doc, ["a"], 1)
    with pytest.raises(FormatError):
        delete_at(doc, ["a"])
    with pytest.raises(FormatError):
        insert_at(doc, [], "k", 1)
