from collections import namedtuple


class UnpackException(Exception):
    """Base class for some exceptions raised while unpacking.

    NOTE: unpack may raise exception other than subclass of
    UnpackException.  If you want to catch all error, catch
    Exception instead.
    """


class BufferFull(UnpackException):
    pass


class OutOfData(UnpackException):
    """More bytes are required to finish unpacking the current object."""
    pass


class FormatError(ValueError, UnpackException):
    """Invalid msgpack format"""


class StackError(ValueError, UnpackException):
    """Too nested"""


#: Structured details attached to exceptions raised by the pure Python unpacker.
#:
#: ``position`` is the absolute byte offset where unpacking failed,
#: ``object_start`` is the absolute byte offset where the current record started,
#: and ``path`` is a bounded tuple of array indices and bounded map keys.
UnpackErrorInfo = namedtuple("UnpackErrorInfo", "position object_start path")


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
