"""OpenVINS ov_eval compatibility layer."""

from . import api as _api

__all__ = list(_api.__all__)

globals().update({name: getattr(_api, name) for name in __all__})
