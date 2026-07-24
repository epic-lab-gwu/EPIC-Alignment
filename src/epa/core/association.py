from __future__ import annotations

import numpy as np

from .time_alignment import matching_time_indices


def _match_nearest_timestamps(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
):
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1) + float(offset_est_s)
    idx_ref = []
    idx_est = []
    for i, t_ref in enumerate(s_ref):
        diffs = np.abs(s_est - t_ref)
        j = int(np.argmin(diffs))
        if float(diffs[j]) <= float(max_diff_s):
            idx_ref.append(int(i))
            idx_est.append(int(j))
    return np.asarray(idx_ref, dtype=int), np.asarray(idx_est, dtype=int)


def _associate_gt_est(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
):
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1)

    if s_ref.size == 0 or s_est.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    est_longer = s_est.size > s_ref.size
    if est_longer:
        ids_short, ids_long = _match_nearest_timestamps(
            s_ref,
            s_est,
            max_diff_s=float(max_diff_s),
            offset_est_s=float(offset_est_s),
        )
        ids_ref = ids_short
        ids_est = ids_long
    else:
        ids_short, ids_long = _match_nearest_timestamps(
            s_est,
            s_ref,
            max_diff_s=float(max_diff_s),
            offset_est_s=-float(offset_est_s),
        )
        ids_ref = ids_long
        ids_est = ids_short

    return np.asarray(ids_ref, dtype=int), np.asarray(ids_est, dtype=int)


def _offset_match_diagnostics(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
    overlap_gate_min_pairs: int | None = None,
):
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1)
    pair_cap_global = max(1, min(s_ref.size, s_est.size))
    if overlap_gate_min_pairs is None:
        overlap_gate_min_pairs = max(80, int(0.1 * float(pair_cap_global)))
    overlap_gate_min_pairs = max(1, int(overlap_gate_min_pairs))

    ids_ref, ids_est = matching_time_indices(
        s_ref,
        s_est,
        max_diff=float(max_diff_s),
        offset_2=float(offset_est_s),
    )
    match_count = int(len(ids_ref))
    ratio_global = min(1.0, float(match_count) / float(pair_cap_global))

    s_est_shift = s_est + float(offset_est_s)
    overlap_start = max(float(s_ref[0]), float(s_est_shift[0]))
    overlap_end = min(float(s_ref[-1]), float(s_est_shift[-1]))
    overlap_cap = 0
    ratio_overlap = ratio_global
    if overlap_end >= overlap_start:
        n_ref_overlap = int(np.sum((s_ref >= overlap_start) & (s_ref <= overlap_end)))
        n_est_overlap = int(np.sum((s_est_shift >= overlap_start) & (s_est_shift <= overlap_end)))
        overlap_cap = int(min(n_ref_overlap, n_est_overlap))
        if overlap_cap > 0:
            ratio_overlap = min(1.0, float(match_count) / float(overlap_cap))

    ratio_gate = ratio_global
    if overlap_cap >= overlap_gate_min_pairs:
        ratio_gate = max(ratio_global, ratio_overlap)

    return {
        "match_count": match_count,
        "ratio_global": float(ratio_global),
        "ratio_overlap": float(ratio_overlap),
        "ratio_gate": float(ratio_gate),
        "pair_cap_global": int(pair_cap_global),
        "pair_cap_overlap": int(overlap_cap),
        "overlap_gate_min_pairs": int(overlap_gate_min_pairs),
        "ids_ref": ids_ref,
        "ids_est": ids_est,
    }
