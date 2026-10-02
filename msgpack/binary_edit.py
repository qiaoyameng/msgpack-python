"""Structural path-based editing of packed MessagePack documents.

These functions edit the packed byte representation directly.  The
document is never fully unpacked and repacked: only the containers on
the edited path have their headers rewritten (choosing a new header
format when the element count crosses the fixmap/map16/map32 or
fixarray/array16/array32 boundaries), untouched branches keep their
original bytes verbatim, and ext payloads are treated as opaque.

All operations are atomic: they either return the complete edited
document or raise an exception, never producing partial output.
"""

import struct

from .exceptions import FormatError
from .fallback import Packer, unpackb

__all__ = [
    "AmbiguousKeyError",
    "DuplicateKeyError",
    "InvalidPathError",
    "PathEditError",
    "PathNotFoundError",
    "delete_at",
    "insert_at",
    "replace_at",
]


class PathEditError(Exception):
    """Base class for path editing errors."""


class PathNotFoundError(PathEditError):
    """The path does not exist in the document."""


class AmbiguousKeyError(PathEditError):
    """A map on the path contains the same key more than once."""


class DuplicateKeyError(PathEditError):
    """The target map of insert_at already contains the key."""


class InvalidPathError(PathEditError, TypeError):
    """The path cannot be followed (e.g. it continues through a scalar)."""


def _read_uint(buf, pos, size):
    if pos + size > len(buf):
        raise FormatError("truncated msgpack document")
    return int.from_bytes(buf[pos : pos + size], "big")


def _skip(buf, pos):
    """Return the offset just past the object starting at *pos*.

    Validates the structure iteratively without building any Python
    objects.  Ext payloads are skipped as opaque byte ranges.
    """
    end = len(buf)
    remaining = 1
    while remaining:
        if pos >= end:
            raise FormatError("truncated msgpack document")
        marker = buf[pos]
        pos += 1
        remaining -= 1
        if marker <= 0x7F or marker >= 0xE0 or marker in (0xC0, 0xC2, 0xC3):
            continue
        if marker <= 0x8F:  # fixmap
            remaining += 2 * (marker & 0x0F)
            continue
        if marker <= 0x9F:  # fixarray
            remaining += marker & 0x0F
            continue
        if marker <= 0xBF:  # fixstr
            pos += marker & 0x1F
        elif marker == 0xC1:
            raise FormatError("0xc1 is not a valid msgpack marker")
        elif marker in (0xC4, 0xD9):  # bin 8 / str 8
            pos += 1 + _read_uint(buf, pos, 1)
        elif marker in (0xC5, 0xDA):  # bin 16 / str 16
            pos += 2 + _read_uint(buf, pos, 2)
        elif marker in (0xC6, 0xDB):  # bin 32 / str 32
            pos += 4 + _read_uint(buf, pos, 4)
        elif marker == 0xC7:  # ext 8
            pos += 2 + _read_uint(buf, pos, 1)
        elif marker == 0xC8:  # ext 16
            pos += 3 + _read_uint(buf, pos, 2)
        elif marker == 0xC9:  # ext 32
            pos += 5 + _read_uint(buf, pos, 4)
        elif marker == 0xCA:  # float 32
            pos += 4
        elif marker == 0xCB:  # float 64
            pos += 8
        elif marker in (0xCC, 0xD0):  # uint 8 / int 8
            pos += 1
        elif marker in (0xCD, 0xD1):  # uint 16 / int 16
            pos += 2
        elif marker in (0xCE, 0xD2):  # uint 32 / int 32
            pos += 4
        elif marker in (0xCF, 0xD3):  # uint 64 / int 64
            pos += 8
        elif marker == 0xD4:  # fixext 1
            pos += 2
        elif marker == 0xD5:  # fixext 2
            pos += 3
        elif marker == 0xD6:  # fixext 4
            pos += 5
        elif marker == 0xD7:  # fixext 8
            pos += 9
        elif marker == 0xD8:  # fixext 16
            pos += 17
        elif marker == 0xDC:  # array 16
            remaining += _read_uint(buf, pos, 2)
            pos += 2
        elif marker == 0xDD:  # array 32
            remaining += _read_uint(buf, pos, 4)
            pos += 4
        elif marker == 0xDE:  # map 16
            remaining += 2 * _read_uint(buf, pos, 2)
            pos += 2
        else:  # marker == 0xDF, map 32
            remaining += 2 * _read_uint(buf, pos, 4)
            pos += 4
        if pos > end:
            raise FormatError("truncated msgpack document")
    return pos


def _container_header(buf, pos, want):
    """Return (count, offset_of_first_element) for a map/array header."""
    marker = buf[pos]
    if want == "map":
        if 0x80 <= marker <= 0x8F:
            return marker & 0x0F, pos + 1
        if marker == 0xDE:
            return _read_uint(buf, pos + 1, 2), pos + 3
        if marker == 0xDF:
            return _read_uint(buf, pos + 1, 4), pos + 5
    else:
        if 0x90 <= marker <= 0x9F:
            return marker & 0x0F, pos + 1
        if marker == 0xDC:
            return _read_uint(buf, pos + 1, 2), pos + 3
        if marker == 0xDD:
            return _read_uint(buf, pos + 1, 4), pos + 5
    raise InvalidPathError(
        "path continues through a non-%s object at offset %d" % (want, pos)
    )


def _pack_map_header(count):
    if count < 16:
        return bytes((0x80 | count,))
    if count <= 0xFFFF:
        return b"\xde" + struct.pack(">H", count)
    return b"\xdf" + struct.pack(">I", count)


def _pack_array_header(count):
    if count < 16:
        return bytes((0x90 | count,))
    if count <= 0xFFFF:
        return b"\xdc" + struct.pack(">H", count)
    return b"\xdd" + struct.pack(">I", count)


def _decode_key(buf, start, end):
    return unpackb(bytes(buf[start:end]), raw=False, strict_map_key=False)


def _edit(buf, start, path, op):
    if not path:
        action = op[0]
        if action == "replace":
            return op[1]
        if action == "insert":
            return _insert_into_map(buf, start, op[1], op[2])
        raise InvalidPathError("cannot delete the root object")
    marker = buf[start]
    if 0x80 <= marker <= 0x8F or marker in (0xDE, 0xDF):
        return _edit_map(buf, start, path, op)
    if 0x90 <= marker <= 0x9F or marker in (0xDC, 0xDD):
        return _edit_array(buf, start, path, op)
    raise InvalidPathError(
        "path continues through a non-container object at offset %d" % start
    )


def _edit_map(buf, start, path, op):
    count, pos = _container_header(buf, start, "map")
    wanted = path[0]
    entries = []  # (key_start, key_end, value_start, value_end)
    match = None
    for i in range(count):
        key_start = pos
        key_end = _skip(buf, key_start)
        value_start = key_end
        value_end = _skip(buf, value_start)
        entries.append((key_start, key_end, value_start, value_end))
        if _decode_key(buf, key_start, key_end) == wanted:
            if match is not None:
                raise AmbiguousKeyError(
                    "map at offset %d contains key %r more than once"
                    % (start, wanted)
                )
            match = i
        pos = value_end
    if match is None:
        raise PathNotFoundError(
            "key %r not found in map at offset %d" % (wanted, start)
        )
    action = op[0] if len(path) == 1 else "descend"
    if action == "delete":
        segments = [_pack_map_header(count - 1)]
        for i, (key_start, _, _, value_end) in enumerate(entries):
            if i != match:
                segments.append(buf[key_start:value_end])
        return b"".join(segments)
    segments = [buf[start : entries[0][0]]]  # keep original header bytes
    for i, (key_start, key_end, value_start, value_end) in enumerate(entries):
        segments.append(buf[key_start:key_end])
        if i == match:
            if action == "replace":
                segments.append(op[1])
            else:
                segments.append(_edit(buf, value_start, path[1:], op))
        else:
            segments.append(buf[value_start:value_end])
    return b"".join(segments)


def _edit_array(buf, start, path, op):
    count, pos = _container_header(buf, start, "array")
    index = path[0]
    if isinstance(index, bool) or not isinstance(index, int):
        raise InvalidPathError("array index must be an int, got %r" % (index,))
    if index < 0:
        index += count
    if not 0 <= index < count:
        raise PathNotFoundError(
            "index %r out of range for array of %d elements at offset %d"
            % (path[0], count, start)
        )
    spans = []
    for _ in range(count):
        span_start = pos
        pos = _skip(buf, span_start)
        spans.append((span_start, pos))
    action = op[0] if len(path) == 1 else "descend"
    if action == "delete":
        segments = [_pack_array_header(count - 1)]
        for i, (span_start, span_end) in enumerate(spans):
            if i != index:
                segments.append(buf[span_start:span_end])
        return b"".join(segments)
    segments = [buf[start : spans[0][0]]]  # keep original header bytes
    for i, (span_start, span_end) in enumerate(spans):
        if i == index:
            if action == "replace":
                segments.append(op[1])
            else:
                segments.append(_edit(buf, span_start, path[1:], op))
        else:
            segments.append(buf[span_start:span_end])
    return b"".join(segments)


def _insert_into_map(buf, start, packed_key, packed_value):
    count, pos = _container_header(buf, start, "map")
    wanted = unpackb(packed_key, raw=False, strict_map_key=False)
    segments = [_pack_map_header(count + 1)]
    for _ in range(count):
        key_start = pos
        key_end = _skip(buf, key_start)
        if _decode_key(buf, key_start, key_end) == wanted:
            raise DuplicateKeyError(
                "map at offset %d already contains key %r" % (start, wanted)
            )
        value_end = _skip(buf, key_end)
        segments.append(buf[key_start:value_end])
        pos = value_end
    segments.append(packed_key)
    segments.append(packed_value)
    return b"".join(segments)


def _pack_value(value):
    return Packer().pack(value)


def _prepare(data, path):
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes-like, not %s" % type(data).__name__)
    if isinstance(path, (str, bytes)) or not isinstance(path, (list, tuple)):
        raise TypeError("path must be a list or tuple of keys/indices")
    path = list(path)
    if not data:
        raise FormatError("empty input")
    # Validate the whole document up front so corrupt input never
    # yields partial output.
    if _skip(data, 0) != len(data):
        raise FormatError("trailing bytes after the root object")
    return data, path


def replace_at(data, path, value):
    """Return *data* with the object at *path* replaced by *value*.

    *path* is a list of map keys and array indices locating the target;
    an empty path replaces the whole document.  *value* is packed with
    the default :class:`Packer` options.
    """
    data, path = _prepare(data, path)
    return _edit(data, 0, path, ("replace", _pack_value(value)))


def delete_at(data, path):
    """Return *data* with the map entry or array element at *path* removed.

    Ancestor container headers are rewritten when the element count
    crosses a header format boundary.
    """
    data, path = _prepare(data, path)
    if not path:
        raise InvalidPathError("cannot delete the root object")
    return _edit(data, 0, path, ("delete",))


def insert_at(data, path, key, value):
    """Return *data* with ``key: value`` appended to the map at *path*.

    Raises :class:`DuplicateKeyError` if the map already contains *key*.
    """
    data, path = _prepare(data, path)
    return _edit(data, 0, path, ("insert", _pack_value(key), _pack_value(value)))
