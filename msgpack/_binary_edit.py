"""Structural, in-place-byte-splicing MessagePack editing."""

import importlib
import struct
from collections import namedtuple

from .exceptions import DuplicateMapKeyError, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp


def _fallback():
    return importlib.import_module("msgpack.fallback")

REPLACE = "replace"
DELETE = "delete"
INSERT = "insert"

_SCALAR = 0
_ARRAY = 1
_MAP = 2

_MISSING_KEY = object()

_Span = namedtuple(
    "_Span", ("start", "body_start", "kind", "count", "end"), defaults=(None,)
)


class _ExtKey(namedtuple("_ExtKey", "code data")):
    pass


def _view(data):
    view = memoryview(data)
    if view.itemsize != 1:
        raise ValueError("cannot edit a multi-byte object")
    if not view.contiguous:
        view = memoryview(view.tobytes())
    return view


def _require(view, pos, size):
    if pos < 0 or pos + size > len(view):
        raise ValueError("incomplete MessagePack data")


def _ext_hook(code, data):
    if code >= 0:
        return ExtType(code, data)
    if code == -1:
        try:
            return Timestamp.from_bytes(data)
        except ValueError:
            return _ExtKey(code, data)
    return _ExtKey(code, data)


def _check_depth(depth):
    if depth > 1024:
        raise StackError("recursion limit exceeded")


def _read_header(view, pos):
    _require(view, pos, 1)
    marker = view[pos]

    if marker <= 0x7F or marker >= 0xE0:
        return _Span(pos, pos + 1, _SCALAR, 0, pos + 1)
    if marker <= 0x8F:
        return _Span(pos, pos + 1, _MAP, marker & 0x0F)
    if marker <= 0x9F:
        return _Span(pos, pos + 1, _ARRAY, marker & 0x0F)
    if marker <= 0xBF:
        end = pos + 1 + (marker & 0x1F)
        _require(view, pos + 1, marker & 0x1F)
        return _Span(pos, pos + 1, _SCALAR, 0, end)

    if marker in (0xC0, 0xC2, 0xC3):
        return _Span(pos, pos + 1, _SCALAR, 0, pos + 1)
    if marker == 0xC1:
        raise FormatError("invalid MessagePack marker: 0xc1")

    if 0xCA <= marker <= 0xD3:
        sizes = {
            0xCA: 4,
            0xCB: 8,
            0xCC: 1,
            0xCD: 2,
            0xCE: 4,
            0xCF: 8,
            0xD0: 1,
            0xD1: 2,
            0xD2: 4,
            0xD3: 8,
        }
        size = sizes[marker]
        _require(view, pos + 1, size)
        return _Span(pos, pos + 1, _SCALAR, 0, pos + 1 + size)

    if 0xD4 <= marker <= 0xD8:
        size = {0xD4: 1, 0xD5: 2, 0xD6: 4, 0xD7: 8, 0xD8: 16}[marker]
        _require(view, pos + 1, size + 1)
        return _Span(pos, pos + 1, _SCALAR, 0, pos + 2 + size)

    if 0xC4 <= marker <= 0xC6 or 0xD9 <= marker <= 0xDB:
        size = {0xC4: 1, 0xC5: 2, 0xC6: 4, 0xD9: 1, 0xDA: 2, 0xDB: 4}[marker]
        _require(view, pos + 1, size)
        length = struct.unpack_from({1: ">B", 2: ">H", 4: ">I"}[size], view, pos + 1)[0]
        _require(view, pos + 1 + size, length)
        return _Span(pos, pos + 1, _SCALAR, 0, pos + 1 + size + length)

    if 0xDC <= marker <= 0xDD:
        size = 2 if marker == 0xDC else 4
        _require(view, pos + 1, size)
        count = struct.unpack_from(">H" if size == 2 else ">I", view, pos + 1)[0]
        kind = _ARRAY
    elif 0xDE <= marker <= 0xDF:
        size = 2 if marker == 0xDE else 4
        _require(view, pos + 1, size)
        count = struct.unpack_from(">H" if size == 2 else ">I", view, pos + 1)[0]
        kind = _MAP
    elif 0xC7 <= marker <= 0xC9:
        size = {0xC7: 1, 0xC8: 2, 0xC9: 4}[marker]
        _require(view, pos + 1, size)
        length = struct.unpack_from({1: ">B", 2: ">H", 4: ">I"}[size], view, pos + 1)[0]
        _require(view, pos + 1 + size, length + 1)
        return _Span(pos, pos + 1, _SCALAR, 0, pos + 1 + size + 1 + length)
    else:
        raise FormatError("unknown MessagePack marker: 0x%02x" % marker)

    return _Span(pos, pos + 1 + size, kind, count)


def _end_of_object(view, pos=0):
    root = _read_header(view, pos)
    if root.kind == _SCALAR:
        return root.end

    stack = [[root, root.count * (2 if root.kind == _MAP else 1)]]
    pos = root.body_start
    while stack:
        frame = stack[-1]
        if frame[1] == 0:
            stack.pop()
            if not stack:
                return pos
            stack[-1][1] -= 1
            continue

        obj = _read_header(view, pos)
        if obj.kind == _SCALAR:
            pos = obj.end
            frame[1] -= 1
        else:
            _check_depth(len(stack) + 1)
            stack.append([obj, obj.count * (2 if obj.kind == _MAP else 1)])
            pos = obj.body_start


def _validate_one(view):
    if not len(view):
        raise ValueError("incomplete MessagePack data")
    end = _end_of_object(view, 0)
    if end != len(view):
        raise ExtraData(None, bytes(view[end:]))
    return end


def _canonical(value):
    if isinstance(value, Timestamp):
        value = ExtType(-1, value.to_bytes())
    if isinstance(value, _ExtKey):
        return ("ext", value.code, value.data)

    value_type = type(value)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        return ("float", value)
    if value_type is str:
        return ("str", value)
    if value_type is bytes or value_type is bytearray:
        return ("bytes", bytes(value))
    if value is None:
        return ("none", None)
    if isinstance(value, ExtType):
        return ("ext", value.code, value.data)
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_canonical(item) for item in value))
    if hasattr(value, "items"):
        pairs = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        return ("map", tuple(sorted(pairs, key=lambda pair: repr(pair))))
    raise TypeError(f"unsupported MessagePack path key: {value!r}")


def _decode_key(view, start, end, raw, strict_map_key):
    return _fallback().unpackb(
        bytes(view[start:end]),
        raw=raw,
        strict_map_key=strict_map_key,
        ext_hook=_ext_hook,
    )


def _find_map(view, span, key, raw, strict_map_key):
    wanted = _canonical(key)
    pos = span.body_start
    match = None

    for _ in range(span.count):
        key_start = pos
        key_end = _end_of_object(view, key_start)
        try:
            decoded = _decode_key(view, key_start, key_end, raw, strict_map_key)
        except UnicodeDecodeError:
            decoded = _MISSING_KEY

        value_start = key_end
        value_end = _end_of_object(view, value_start)
        if decoded is not _MISSING_KEY and _canonical(decoded) == wanted:
            if match is not None:
                raise DuplicateMapKeyError(repr(key))
            match = (key_start, key_end, value_start, value_end)
        pos = value_end

    return match, pos


def _step(view, span, span_end, component, raw, strict_map_key):
    if span.kind == _MAP:
        found, container_end = _find_map(view, span, component, raw, strict_map_key)
        if found is None:
            raise KeyError(component)
        key_start, _key_end, value_start, value_end = found
        child = _read_header(view, value_start)
        return child, value_end, span, container_end, key_start, value_end

    if span.kind == _ARRAY:
        if type(component) is not int:
            raise TypeError("array path components must be integers")
        if component < 0:
            raise IndexError("array path components must be non-negative")
        if component >= span.count:
            raise IndexError(component)

        pos = span.body_start
        for index in range(component + 1):
            item_start = pos
            item_end = _end_of_object(view, item_start)
            if index == component:
                child = _read_header(view, item_start)
                return child, item_end, span, span_end, item_start, item_end
            pos = item_end

    raise TypeError(f"cannot descend into {component!r}")


def _validate_path(path):
    if not isinstance(path, (tuple, list)):
        raise TypeError("path must be a tuple or list")
    return tuple(path)


def _pack_header(kind, count):
    packer = _fallback().Packer()
    if kind == _MAP:
        return packer.pack_map_header(count)
    return packer.pack_array_header(count)


def editb(
    packed,
    path,
    value=_MISSING_KEY,
    *,
    op=REPLACE,
    key=_MISSING_KEY,
    raw=False,
    strict_map_key=False,
    **pack_options,
):
    """Edit one MessagePack value by path without unpacking the whole document.

    ``path`` addresses array items by index and map entries by key.  The
    ``raw`` and ``strict_map_key`` options control only how existing map keys
    are decoded while following that path.  ``REPLACE`` replaces the addressed
    value, ``DELETE`` removes an array item or map pair, and ``INSERT`` appends
    a new key/value pair to the addressed map.

    A new bytes object is returned.  Duplicate map keys on the path raise
    ``DuplicateMapKeyError`` instead of choosing one occurrence.
    """
    path = _validate_path(path)
    view = _view(packed)
    root_end = _validate_one(view)
    root = _read_header(view, 0)
    packer = _fallback().Packer(**pack_options)

    parent = None
    parent_end = None
    entry_start = None
    entry_end = None
    target = root
    target_end = root_end

    for component in path:
        target, target_end, parent, parent_end, entry_start, entry_end = _step(
            view, target, target_end, component, raw, strict_map_key
        )

    if op == REPLACE:
        if value is _MISSING_KEY:
            raise TypeError("replace requires a value")
        packed_value = packer.pack(value)
        parts = [view[: target.start], packed_value, view[target_end:root_end]]

    elif op == DELETE:
        if not path:
            raise ValueError("cannot delete the MessagePack root")
        if parent.kind not in (_MAP, _ARRAY):
            raise TypeError("can only delete a map entry or array item")
        packed_header = _pack_header(parent.kind, parent.count - 1)
        parts = [
            view[: parent.start],
            packed_header,
            view[parent.body_start:entry_start],
            view[entry_end:parent_end],
            view[parent_end:root_end],
        ]

    elif op == INSERT:
        if key is _MISSING_KEY or value is _MISSING_KEY:
            raise TypeError("insert requires a key and value")
        if target.kind != _MAP:
            raise TypeError("insert is only supported for maps")
        found, target_end = _find_map(view, target, key, raw, strict_map_key)
        if found is not None:
            raise ValueError(f"map key already exists: {key!r}")
        packed_pair = packer.pack(key) + packer.pack(value)
        packed_header = _pack_header(_MAP, target.count + 1)
        parts = [
            view[: target.start],
            packed_header,
            view[target.body_start:target_end],
            packed_pair,
            view[target_end:root_end],
        ]

    else:
        raise ValueError(f"unknown operation: {op!r}")

    result = b"".join(parts)
    result_view = memoryview(result)
    edited_end = _end_of_object(result_view, 0)
    if edited_end != len(result):
        raise ValueError("editing produced invalid MessagePack data")
    return result


def replaceb(packed, path, value, raw=False, strict_map_key=False, **pack_options):
    """Replace the value at ``path`` without repacking untouched branches."""
    return editb(
        packed,
        path,
        value,
        op=REPLACE,
        raw=raw,
        strict_map_key=strict_map_key,
        **pack_options,
    )


def deleteb(packed, path, raw=False, strict_map_key=False):
    """Delete the array item or map pair at ``path``."""
    return editb(
        packed,
        path,
        op=DELETE,
        raw=raw,
        strict_map_key=strict_map_key,
    )


def insertb(packed, path, key, value, raw=False, strict_map_key=False, **pack_options):
    """Insert ``key`` and ``value`` into the map located at ``path``."""
    return editb(
        packed,
        path,
        value,
        op=INSERT,
        key=key,
        raw=raw,
        strict_map_key=strict_map_key,
        **pack_options,
    )
