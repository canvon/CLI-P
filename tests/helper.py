import warnings
import contextlib
import sys
from io import StringIO
from pathlib import Path

# Low-level access to C internas provided with the help of:
# * https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# * https://stackoverflow.com/questions/52219393/how-do-i-capture-stderr-in-python
# * https://github.com/minrk/wurlitzer/blob/master/wurlitzer.py
# * Various experimentation, finally https://docs.python.org/3/library/ctypes.html
#   (the official documentation) helped out quite a bit!
class C:
    types = None
    libc = None
    stderr_p = None

    @classmethod
    def init_c_level_access(cls):
        # Import foreign-function-call machinery.
        # Do it only now that some code is actually needing the functionality.
        import ctypes
        C.types = ctypes

        # Open libc. The use_errno=True is necessary to let ctypes.get_errno() actually work.
        C.libc = C.types.CDLL(None, use_errno=True)

        # Setup pointers to global state.
        # This would be one possibility:
        #if sys.platform == 'darwin':
        # But it seemed the wurlitzer code would have the more robust:
        try:
            C.stderr_p = C.types.c_void_p.in_dll(C.libc, 'stderr')
        except ValueError:
            C.stderr_p = C.types.c_void_p.in_dll(C.libc, '__stderrp')

        # Setup various functions...
        # Setting the types is quite important, as size of void pointer (8) != size of int (4), often, nowadays.
        C.strerror = C.libc.strerror
        C.strerror.argtypes = [C.types.c_int]
        C.strerror.restype = C.types.c_char_p
        #
        C.fopen = C.libc.fopen
        C.fopen.argtypes = [C.types.c_char_p, C.types.c_char_p]
        C.fopen.restype = C.types.c_void_p
        #
        C.fflush = C.libc.fflush
        C.fflush.argtypes = [C.types.c_void_p]
        C.fflush.restype = C.types.c_int
        #
        C.fclose = C.libc.fclose
        C.fclose.argtypes = [C.types.c_void_p]
        C.fclose.restype = C.types.c_int

    @classmethod
    def get_error(cls):
        c_errno = C.types.get_errno()
        c_errstr = C.strerror(c_errno)
        return None if c_errstr is None else c_errstr.decode()


def setupWarningFilters():
    # Not our bugs. (Or: De-clutter test suite terminal output.)
    warnings.filterwarnings("ignore", module="clip", category=ResourceWarning, message="unclosed file")
    warnings.filterwarnings("ignore", module="distutils", category=DeprecationWarning, message="the imp module is deprecated in favour of importlib")


def capture_stdout(code):
    ret = None
    with contextlib.closing(StringIO()) as f:
        with contextlib.redirect_stdout(f):
            ret = code()
        return f.getvalue(), ret

def capture_stdout_stderr(code):
    ret = None
    with contextlib.closing(StringIO()) as out, contextlib.closing(StringIO()) as err:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            ret = code()
        return out.getvalue(), err.getvalue(), ret

def capture_stdout_cstderr(code):
    ret = None
    if C.types is None:
        C.init_c_level_access()
    tmpfile = Path("tests/captured_output.tmp.log")
    c_tmpfile = C.fopen(str(tmpfile).encode(), b'w')
    if not c_tmpfile:
        raise RuntimeError(f"C-level opening temporary file {tmpfile!r} failed: {C.get_error()}")
    with contextlib.closing(StringIO()) as out:
        with contextlib.redirect_stdout(out):
            errout_value = None
            c_stderr_prev = C.stderr_p.value
            try:
                sys.stderr.flush()
                if C.fflush(C.stderr_p.value) != 0:
                    raise RuntimeError(f"C-level error flushing previous stderr: {C.get_error()}")
                C.stderr_p.value = c_tmpfile

                # Run the wrapped code,
                # under normal python stdout redirection
                # and C-level stderr redirection.
                ret = code()
            finally:
                sys.stderr.flush()
                C.stderr_p.value = c_stderr_prev
                if C.fflush(c_tmpfile) != 0:
                    raise RuntimeError(f"C-level error flushing temporary file: {C.get_error()}")
                if C.fclose(c_tmpfile) != 0:
                    raise RuntimeError(f"C-level error closing temporary file: {C.get_error()}")
                c_tmpfile = None
                errout_value = tmpfile.read_text()
                tmpfile.unlink()
        return out.getvalue(), errout_value, ret
