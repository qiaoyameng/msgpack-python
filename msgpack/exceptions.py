class UnpackException(Exception):
    """Base class for some exceptions raised while unpacking.

    NOTE: unpack may raise exception other than subclass of
    UnpackException.  If you want to catch all error, catch
    Exception instead.
    """


class UnpackDiagnostic:
    """Position and path information captured when unpacking fails."""

    __slots__ = ("offset", "object_start", "path", "path_text")

    def __init__(self, offset, object_start, path, path_text):
        self.offset = offset
        self.object_start = object_start
        self.path = path
        self.path_text = path_text

    def __str__(self):
        return (
            f"offset={self.offset}, object_start={self.object_start}, "
            f"path={self.path_text!r}"
        )


class BufferFull(UnpackException):
    pass


class OutOfData(UnpackException):
    pass


class FormatError(ValueError, UnpackException):
    """Invalid msgpack format"""


class StackError(ValueError, UnpackException):
    """Too nested"""


# Deprecated.  Use ValueError instead
UnpackValueError = ValueError


class ExtraData(UnpackValueError):
    """ExtraData is raised when there is trailing data.

    This exception is raised while only one-shot (not streaming)
    unpack.
    """

    def __init__(self, unpacked, extra):
        self.unpacked = unpacked
        self.extra = extra

    def __str__(self):
        return "unpack(b) received extra data."


# Deprecated.  Use Exception instead to catch all exception during packing.
PackException = Exception
PackValueError = ValueError
PackOverflowError = OverflowError
