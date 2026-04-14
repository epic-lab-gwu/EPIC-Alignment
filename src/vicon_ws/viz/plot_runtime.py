from __future__ import annotations

import sys


def _backend_token(backend: str) -> str:
    text = str(backend or "").strip().lower()
    if not text:
        return ""
    return text.split(".")[-1]


def _is_interactive_backend(matplotlib_mod, backend: str) -> bool:
    token = _backend_token(backend)
    if not token:
        return False
    try:
        names = {str(x).strip().lower() for x in matplotlib_mod.rcsetup.interactive_bk}
    except Exception:
        names = set()
    return token in names


def configure_plot_runtime(*, plot_interactive: bool, plot_backend: str = "") -> tuple[bool, str]:
    import matplotlib
    import matplotlib.pyplot as plt

    requested = str(plot_backend or "").strip()
    if requested:
        try:
            plt.switch_backend(requested)
        except Exception as exc:
            print(f"[plot] Failed to switch backend to '{requested}': {exc}")

    if bool(plot_interactive):
        current = str(matplotlib.get_backend())
        if not _is_interactive_backend(matplotlib, current):
            candidates = ["qtagg", "tkagg", "qt5agg", "wxagg"]
            for name in candidates:
                try:
                    plt.switch_backend(name)
                    current = str(matplotlib.get_backend())
                    break
                except Exception:
                    continue
        current = str(matplotlib.get_backend())
        if not _is_interactive_backend(matplotlib, current):
            return False, current
        return True, current

    return False, str(matplotlib.get_backend())


def should_enable_interactive_plot(*, plot: bool, plot_interactive: bool) -> bool:
    if bool(plot_interactive):
        return True
    if not bool(plot):
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def show_plots() -> None:
    import matplotlib.pyplot as plt

    try:
        plt.show()
    except Exception as exc:
        print(f"[plot] Interactive show failed: {exc}")
