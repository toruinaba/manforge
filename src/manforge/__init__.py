from importlib.metadata import version as _get_version
try:
    __version__ = _get_version("manforge")
except Exception:
    __version__ = "0.1.0"
