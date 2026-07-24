from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate

from .association import _associate_gt_est, _offset_match_diagnostics
from .calibration import solve_world_alignment
from .math_utils import rmse
from .time_alignment import compute_psr, get_angular_velocity_norm, matching_time_indices


def _offset_grid(min_offset_s: float, max_offset_s: float, step_s: float):
    if step_s <= 0.0:
        raise ValueError("offset step must be > 0.")
    values = np.arange(min_offset_s, max_offset_s + 0.5 * step_s, step_s)
    return [float(v) for v in values]


def _prefer_offset_candidate(curr, cand):
    if curr is None:
        return True
    curr_valid, curr_rmse, curr_matches, curr_abs = curr
    cand_valid, cand_rmse, cand_matches, cand_abs = cand
    if cand_valid and not curr_valid:
        return True
    if curr_valid and not cand_valid:
        return False
    if cand_rmse < curr_rmse - 1e-12:
        return True
    if np.isclose(cand_rmse, curr_rmse, atol=1e-12, rtol=0.0):
        if cand_matches > curr_matches:
            return True
        if cand_matches == curr_matches and cand_abs < curr_abs - 1e-12:
            return True
    return False


def _search_direct_offset_from_matched_pairs(
    *,
    t_ref,
    pos_ref,
    t_est,
    pos_est,
    initial_offset_s: float,
    max_diff_s: float,
    min_match_ratio: float,
    coarse_min_s: float = -20.0,
    coarse_max_s: float = 20.0,
    coarse_step_s: float = 0.2,
    refine_window_s: float = 0.5,
    refine_step_s: float = 0.01,
):
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    t_est = np.asarray(t_est, dtype=float).reshape(-1)
    pref = np.asarray(pos_ref, dtype=float)
    pest = np.asarray(pos_est, dtype=float)
    if t_ref.size != pref.shape[0] or t_est.size != pest.shape[0]:
        raise ValueError("Timestamp and position array lengths are inconsistent.")

    pair_cap = max(1, min(t_ref.size, t_est.size))
    min_required = max(3, int(float(min_match_ratio) * float(pair_cap)))

    best_key = None
    best = None

    def _eval_offset(offset_s: float):
        ids_ref, ids_est = _associate_gt_est(
            t_ref,
            t_est,
            max_diff_s=float(max_diff_s),
            offset_est_s=-float(offset_s),
        )
        n = len(ids_ref)
        if n < 3:
            return None
        pref_sel = pref[np.asarray(ids_ref, dtype=int)]
        pest_sel = pest[np.asarray(ids_est, dtype=int)]
        Rw, tw = solve_world_alignment(pest_sel, pref_sel)
        pred = (Rw @ pest_sel.T).T + tw
        rmse = float(np.sqrt(np.mean(np.sum((pred - pref_sel) ** 2, axis=1))))
        valid = bool(n >= min_required)
        key = (valid, rmse, int(n), abs(float(offset_s)))
        return {
            "offset_s": float(offset_s),
            "matches": int(n),
            "valid": valid,
            "rmse_pairs_m": rmse,
            "ids_ref": np.asarray(ids_ref, dtype=int),
            "ids_est": np.asarray(ids_est, dtype=int),
            "_key": key,
        }

    for off in _offset_grid(coarse_min_s, coarse_max_s, coarse_step_s):
        candidate = _eval_offset(off)
        if candidate is None:
            continue
        if _prefer_offset_candidate(best_key, candidate["_key"]):
            best_key = candidate["_key"]
            best = candidate

    init_candidate = _eval_offset(float(initial_offset_s))
    if init_candidate is not None and _prefer_offset_candidate(best_key, init_candidate["_key"]):
        best_key = init_candidate["_key"]
        best = init_candidate

    if best is None:
        return None

    refine_start = max(float(coarse_min_s), float(best["offset_s"]) - float(refine_window_s))
    refine_end = min(float(coarse_max_s), float(best["offset_s"]) + float(refine_window_s))
    for off in _offset_grid(refine_start, refine_end, refine_step_s):
        candidate = _eval_offset(off)
        if candidate is None:
            continue
        if _prefer_offset_candidate(best_key, candidate["_key"]):
            best_key = candidate["_key"]
            best = candidate

    best.pop("_key", None)
    best["required_matches"] = int(min_required)
    return best


def _run_time_alignment(
    *,
    t_gt,
    quat_gt,
    t_est,
    quat_est,
    dt_resample: float,
    offset_search_window_s: float,
    offset_min_match_ratio: float,
    evo_match_max_diff_s: float,
    artificial_offset_s: float | None = None,
):
    t_gt_mid, om_gt = get_angular_velocity_norm(t_gt, quat_gt)
    t_est_mid, om_est = get_angular_velocity_norm(t_est, quat_est)

    dt_resample = float(dt_resample)
    if dt_resample <= 0.0:
        raise ValueError("--dt-resample must be > 0.")
    offset_search_window_s = float(offset_search_window_s)
    offset_min_match_ratio = float(offset_min_match_ratio)
    if not (0.0 <= offset_min_match_ratio <= 1.0):
        raise ValueError("--offset-min-match-ratio must be in [0, 1].")

    t_min = max(float(np.min(t_gt_mid)), float(np.min(t_est_mid)))
    t_max = min(float(np.max(t_gt_mid)), float(np.max(t_est_mid)))
    if t_max - t_min < 100 * dt_resample:
        raise ValueError("Not enough overlap between GT and estimation for robust time alignment.")

    eps = max(1e-9, abs(dt_resample) * 1e-6)
    safe_min = float(np.nextafter(t_min + eps, np.inf))
    safe_max = float(np.nextafter(t_max - eps, -np.inf))
    if safe_max - safe_min < 100 * dt_resample:
        raise ValueError("Not enough valid overlap after applying interpolation safety margins.")
    sample_count = int(np.floor((safe_max - safe_min) / dt_resample)) + 1
    t_uniform = safe_min + dt_resample * np.arange(sample_count, dtype=float)
    t_uniform = np.clip(t_uniform, safe_min, safe_max)
    if t_uniform.size < 100:
        raise ValueError(
            "Not enough valid resampled points after applying interpolation safety margins."
        )

    sig_gt = interp1d(t_gt_mid, om_gt, kind="linear")(t_uniform)
    sig_est = interp1d(t_est_mid, om_est, kind="linear")(t_uniform)
    sig_gt -= np.mean(sig_gt)
    sig_est -= np.mean(sig_est)
    if (not np.all(np.isfinite(sig_gt))) or (not np.all(np.isfinite(sig_est))):
        raise ValueError(
            "Step-1 signals contain non-finite values. Check timestamps for duplicates/non-monotonic samples."
        )

    corr = correlate(sig_est, sig_gt, mode="full")
    if not np.all(np.isfinite(corr)):
        raise ValueError(
            "Cross-correlation contains non-finite values (possible invalid timestamp deltas or signal values)."
        )
    lags = np.arange(-len(sig_gt) + 1, len(sig_est))
    offsets_s = lags.astype(float) * dt_resample

    search_mask = np.ones_like(offsets_s, dtype=bool)
    if offset_search_window_s > 0.0:
        search_mask = np.abs(offsets_s) <= offset_search_window_s
        if not np.any(search_mask):
            raise ValueError(
                "Offset search window has no valid lag candidates. Increase --offset-search-window-s."
            )

    search_indices = np.flatnonzero(search_mask)
    peak_idx = int(search_indices[int(np.argmax(corr[search_indices]))])
    calculated_offset = float(offsets_s[peak_idx])

    def _count_matches_for_offset(offset_s: float) -> dict[str, object]:
        return _offset_match_diagnostics(
            t_gt,
            t_est,
            max_diff_s=evo_match_max_diff_s,
            offset_est_s=-float(offset_s),
        )

    def _best_near_zero_offset(window_s: float) -> tuple[float, int]:
        zero_mask = np.abs(offsets_s) <= float(window_s)
        if offset_search_window_s > 0.0:
            zero_mask &= search_mask
        if np.any(zero_mask):
            zero_indices = np.flatnonzero(zero_mask)
            zero_peak_idx = int(zero_indices[int(np.argmax(corr[zero_indices]))])
        else:
            zero_peak_idx = int(np.argmin(np.abs(offsets_s)))
        return float(offsets_s[zero_peak_idx]), zero_peak_idx

    def _omega_rmse_after_offset(offset_s: float) -> float:
        sig_est_shifted_for_offset = interp1d(
            t_uniform - float(offset_s),
            sig_est,
            kind="linear",
            fill_value="extrapolate",
        )(t_uniform)
        return rmse(sig_est_shifted_for_offset - sig_gt)

    match_diag = _count_matches_for_offset(calculated_offset)
    match_ratio_global = float(match_diag["ratio_global"])
    match_ratio_overlap = float(match_diag["ratio_overlap"])
    match_ratio_gate = float(match_diag["ratio_gate"])
    overlap_pair_cap = int(match_diag["pair_cap_overlap"])
    overlap_gate_min_pairs = int(match_diag["overlap_gate_min_pairs"])
    fallback_used = 0.0
    fallback_offset = float("nan")
    fallback_ratio_global = float("nan")
    fallback_ratio_overlap = float("nan")
    fallback_ratio_gate = float("nan")
    step1_forced_candidate = False
    step1_force_reason = ""
    near_zero_window_s = 1.0
    near_zero_offset, near_zero_peak_idx = _best_near_zero_offset(near_zero_window_s)
    near_zero_diag = _count_matches_for_offset(near_zero_offset)
    near_zero_ratio_global = float(near_zero_diag["ratio_global"])
    near_zero_ratio_overlap = float(near_zero_diag["ratio_overlap"])
    near_zero_ratio_gate = float(near_zero_diag["ratio_gate"])
    omega_rmse_after_peak = _omega_rmse_after_offset(calculated_offset)
    omega_rmse_after_near_zero = _omega_rmse_after_offset(near_zero_offset)
    zero_offset = 0.0
    zero_peak_idx = int(np.argmin(np.abs(offsets_s)))
    zero_diag = _count_matches_for_offset(zero_offset)
    zero_ratio_gate = float(zero_diag["ratio_gate"])
    omega_rmse_after_zero = _omega_rmse_after_offset(zero_offset)
    small_offset_improve_ratio = (omega_rmse_after_zero - omega_rmse_after_near_zero) / (
        omega_rmse_after_zero + 1e-12
    )
    if (
        abs(near_zero_offset) <= near_zero_window_s
        and abs(near_zero_offset) > 0.0
        and zero_ratio_gate >= offset_min_match_ratio
        and small_offset_improve_ratio < 0.01
    ):
        near_zero_offset = zero_offset
        near_zero_peak_idx = zero_peak_idx
        near_zero_diag = zero_diag
        near_zero_ratio_global = float(near_zero_diag["ratio_global"])
        near_zero_ratio_overlap = float(near_zero_diag["ratio_overlap"])
        near_zero_ratio_gate = float(near_zero_diag["ratio_gate"])
        omega_rmse_after_near_zero = omega_rmse_after_zero

    near_zero_preferred = (
        offset_search_window_s <= 0.0
        and abs(calculated_offset) > near_zero_window_s
        and near_zero_ratio_gate >= offset_min_match_ratio
        and omega_rmse_after_peak >= omega_rmse_after_near_zero * 0.95
    )
    # A large correlation peak is ignored when timestamp matching and angular
    # velocity residuals show that the near-zero candidate is equally plausible.
    if near_zero_preferred:
        calculated_offset = near_zero_offset
        peak_idx = near_zero_peak_idx
        match_ratio_global = near_zero_ratio_global
        match_ratio_overlap = near_zero_ratio_overlap
        match_ratio_gate = near_zero_ratio_gate
        overlap_pair_cap = int(near_zero_diag["pair_cap_overlap"])
        overlap_gate_min_pairs = int(near_zero_diag["overlap_gate_min_pairs"])

    if match_ratio_gate < offset_min_match_ratio:
        fallback_offset = near_zero_offset
        zero_peak_idx = near_zero_peak_idx

        fallback_diag = _count_matches_for_offset(fallback_offset)
        fallback_ratio_global = float(fallback_diag["ratio_global"])
        fallback_ratio_overlap = float(fallback_diag["ratio_overlap"])
        fallback_ratio_gate = float(fallback_diag["ratio_gate"])
        if fallback_ratio_gate >= offset_min_match_ratio:
            calculated_offset = fallback_offset
            peak_idx = zero_peak_idx
            match_ratio_global = fallback_ratio_global
            match_ratio_overlap = fallback_ratio_overlap
            match_ratio_gate = fallback_ratio_gate
            overlap_pair_cap = int(fallback_diag["pair_cap_overlap"])
            overlap_gate_min_pairs = int(fallback_diag["overlap_gate_min_pairs"])
            fallback_used = 1.0
        else:
            step1_force_reason = (
                "Unreliable step-1 offset estimate: "
                f"best_match_ratio_global={match_ratio_global:.3f}, "
                f"best_match_ratio_overlap={match_ratio_overlap:.3f}, "
                f"best_match_ratio_gate={match_ratio_gate:.3f}, "
                f"fallback_match_ratio_global={fallback_ratio_global:.3f}, "
                f"fallback_match_ratio_overlap={fallback_ratio_overlap:.3f}, "
                f"fallback_match_ratio_gate={fallback_ratio_gate:.3f}, "
                f"required>={offset_min_match_ratio:.3f}. "
                "Check timestamp quality (duplicates/non-monotonic) or adjust offset search settings."
            )
            step1_forced_candidate = True
            print("--- STEP 1 FALLBACK ---")
            print(step1_force_reason)
            print(f"Continuing with highest-confidence candidate offset: {calculated_offset:.4f} s")

    sig_est_shifted = interp1d(
        t_uniform - calculated_offset,
        sig_est,
        kind="linear",
        fill_value="extrapolate",
    )(t_uniform)

    corr_norm_peak = corr[peak_idx] / (np.linalg.norm(sig_gt) * np.linalg.norm(sig_est) + 1e-12)
    psr = compute_psr(corr, peak_idx, guard_bins=max(1, int(0.02 / dt_resample)))
    time_metrics = {
        "offset_est_s": calculated_offset,
        "offset_err_ms": np.nan,
        "xcorr_peak_normalized": corr_norm_peak,
        "xcorr_psr": psr,
        "omega_rmse_before": rmse(sig_est - sig_gt),
        "omega_rmse_after": rmse(sig_est_shifted - sig_gt),
        "offset_search_window_s": offset_search_window_s,
        "offset_min_match_ratio": offset_min_match_ratio,
        "offset_fallback_used": fallback_used,
        "offset_match_ratio_global": match_ratio_global,
        "offset_match_ratio_overlap": match_ratio_overlap,
        "offset_match_ratio_gate": match_ratio_gate,
        "offset_overlap_pair_cap": float(overlap_pair_cap),
        "offset_overlap_gate_min_pairs": float(overlap_gate_min_pairs),
        "fallback_offset_s": float(fallback_offset),
        "fallback_match_ratio_global": float(fallback_ratio_global),
        "fallback_match_ratio_overlap": float(fallback_ratio_overlap),
        "fallback_match_ratio_gate": float(fallback_ratio_gate),
        "step1_forced_candidate_code": 1.0 if step1_forced_candidate else 0.0,
        "near_zero_offset_s": float(near_zero_offset),
        "near_zero_match_ratio_global": float(near_zero_ratio_global),
        "near_zero_match_ratio_overlap": float(near_zero_ratio_overlap),
        "near_zero_match_ratio_gate": float(near_zero_ratio_gate),
        "near_zero_omega_rmse_after": float(omega_rmse_after_near_zero),
        "zero_omega_rmse_after": float(omega_rmse_after_zero),
        "near_zero_improve_ratio": float(small_offset_improve_ratio),
        "near_zero_preferred_code": 1.0 if near_zero_preferred else 0.0,
    }

    evo_t_offset_used_s = -calculated_offset
    match_ids_ref, _ = matching_time_indices(
        t_gt, t_est, max_diff=evo_match_max_diff_s, offset_2=evo_t_offset_used_s
    )
    time_metrics["evo_t_offset_used_s"] = evo_t_offset_used_s
    time_metrics["evo_match_max_diff_s"] = evo_match_max_diff_s
    time_metrics["evo_matches_equivalent"] = float(len(match_ids_ref))
    time_metrics["evo_matches_ratio_equivalent"] = float(match_ratio_global)
    time_metrics["evo_matches_ratio_overlap_aware"] = float(match_ratio_overlap)
    time_metrics["evo_matches_ratio_for_gate"] = float(match_ratio_gate)
    if artificial_offset_s is not None:
        time_metrics["offset_err_ms"] = abs(calculated_offset - float(artificial_offset_s)) * 1e3

    time_metrics["omega_rmse_improve_pct"] = (
        (time_metrics["omega_rmse_before"] - time_metrics["omega_rmse_after"])
        / (time_metrics["omega_rmse_before"] + 1e-12)
        * 100.0
    )

    return {
        "calculated_offset": calculated_offset,
        "time_metrics": time_metrics,
        "step1_forced_candidate": step1_forced_candidate,
        "step1_force_reason": step1_force_reason,
        "t_uniform": t_uniform,
        "sig_gt": sig_gt,
        "sig_est": sig_est,
        "corr": corr,
        "lags": lags,
    }
