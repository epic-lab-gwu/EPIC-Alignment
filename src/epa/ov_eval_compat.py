from __future__ import annotations

import sys
import types

from epa.compat.ov_eval import api as _api

# Compatibility shim for the historical import path and console-script targets.
__all__ = list(_api.__all__)

globals().update({name: getattr(_api, name) for name in __all__})


class _CompatModule(types.ModuleType):
    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if not name.startswith("__"):
            setattr(_api, name, value)


sys.modules[__name__].__class__ = _CompatModule


if __name__ == "__main__":
    raise SystemExit(_api.main())
