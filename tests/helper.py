import warnings
import contextlib
from io import StringIO

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
