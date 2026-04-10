"""Root conftest – shim the root __init__.py so pytest can import it."""
import sys
import types
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Pre-register a dummy __init__ module at the key pytest will use,
# so the relative-import inside __init__.py never fires during test setup.
_init_path = str(Path(__file__).resolve().parent / "__init__.py")
_mod_name = "__init__"
if _mod_name not in sys.modules:
    _dummy = types.ModuleType(_mod_name)
    _dummy.__file__ = _init_path
    _dummy.__package__ = ""
    sys.modules[_mod_name] = _dummy

collect_ignore = ["__init__.py"]
