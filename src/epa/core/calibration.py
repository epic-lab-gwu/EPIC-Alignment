import numpy as np
from scipy.spatial.transform import Rotation as R


def build_translation_system(pv, qv, pr, qr, Rext):
    pv = np.asarray(pv, dtype=float)
    pr = np.asarray(pr, dtype=float)
    Rext = np.asarray(Rext, dtype=float)
    Rv = R.from_quat(qv).as_matrix()
    Rr = R.from_quat(qr).as_matrix()
    if pv.shape[0] < 2:
        raise ValueError("No valid translation constraints for extrinsic translation solve.")

    dpv = pv[1:] - pv[:-1]
    dpr = pr[1:] - pr[:-1]
    Rv_i_t = np.swapaxes(Rv[:-1], 1, 2)
    Rr_i_t = np.swapaxes(Rr[:-1], 1, 2)
    ta = np.einsum("nij,nj->ni", Rv_i_t, dpv)
    tb = np.einsum("nij,nj->ni", Rr_i_t, dpr)
    mask = np.linalg.norm(ta, axis=1) >= 1e-3
    if not np.any(mask):
        raise ValueError("No valid translation constraints for extrinsic translation solve.")

    C = np.einsum("nij,njk->nik", Rv_i_t[mask], Rv[1:][mask]) - np.eye(3)
    d = np.einsum("ij,nj->ni", Rext, tb[mask]) - ta[mask]
    return C.reshape(-1, 3), d.reshape(-1), int(np.count_nonzero(mask))


def solve_extrinsic_rotation(qv, qr):
    Rv = R.from_quat(qv)
    Rr = R.from_quat(qr)
    vA = (Rv[:-1].inv() * Rv[1:]).as_rotvec()
    vB = (Rr[:-1].inv() * Rr[1:]).as_rotvec()
    mask = np.linalg.norm(vA, axis=1) > 1e-3
    if np.sum(mask) < 3:
        raise ValueError("Not enough rotational excitation to estimate extrinsic rotation.")
    H = vB[mask].T @ vA[mask]
    U, _, Vt = np.linalg.svd(H)
    X = Vt.T @ U.T
    if np.linalg.det(X) < 0:
        Vt[2, :] *= -1
        X = Vt.T @ U.T
    return X


def solve_extrinsic_translation(pv, qv, pr, qr, Rext):
    C, d, _ = build_translation_system(pv, qv, pr, qr, Rext)
    res, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
    return res


def solve_world_alignment(P, Q):
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    Rw = Vt.T @ U.T
    if np.linalg.det(Rw) < 0:
        Vt[2, :] *= -1
        Rw = Vt.T @ U.T
    return Rw, cQ - Rw @ cP
