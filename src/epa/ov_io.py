from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from epa.core.math_utils import normalize_quat_array


def load_ov_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Standard TUM/OpenVINS text files are regular whitespace-separated numeric
    # tables.  Let NumPy parse that common case in C, while retaining the
    # permissive line parser below for comma-separated or ragged legacy files.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
            rows = np.loadtxt(
                path,
                dtype=float,
                comments="#",
                usecols=range(8),
                ndmin=2,
            )
    except (OSError, ValueError):
        rows = np.empty((0, 8), dtype=float)
    else:
        if rows.shape[0] > 0:
            return rows[:, 0], rows[:, 1:4], normalize_quat_array(rows[:, 4:8])

    times: list[float] = []
    pos: list[list[float]] = []
    quat: list[list[float]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 8:
                continue
            vals = [float(x) for x in parts[:8]]
            times.append(vals[0])
            pos.append(vals[1:4])
            quat.append(vals[4:8])

    if not times:
        raise ValueError(f"Could not parse any trajectory samples from: {path}")

    t = np.asarray(times, dtype=float)
    p = np.asarray(pos, dtype=float)
    q = normalize_quat_array(np.asarray(quat, dtype=float))
    return t, p, q


def load_ov_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    pos: list[list[float]] = []
    quat: list[list[float]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x for x in line.split(",") if x != ""]
            if len(parts) < 8:
                continue
            vals = [float(x) for x in parts[:8]]
            times.append(vals[0] * 1e-9)
            pos.append(vals[1:4])
            quat.append([vals[5], vals[6], vals[7], vals[4]])

    if not times:
        raise ValueError(f"Could not parse any CSV samples from: {path}")

    t = np.asarray(times, dtype=float)
    p = np.asarray(pos, dtype=float)
    q = normalize_quat_array(np.asarray(quat, dtype=float))
    return t, p, q


def load_pose_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".csv":
        return load_ov_csv(path)
    return load_ov_txt(path)
