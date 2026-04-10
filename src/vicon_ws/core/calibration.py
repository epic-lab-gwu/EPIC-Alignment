import numpy as np
from scipy.spatial.transform import Rotation as R


def build_translation_system(pv, qv, pr, qr, Rext):
    Rv = R.from_quat(qv).as_matrix()
    Rr = R.from_quat(qr).as_matrix()
    C, d = [], []
    for i in range(len(pv) - 1):
        ta = Rv[i].T @ (pv[i + 1] - pv[i])
        tb = Rr[i].T @ (pr[i + 1] - pr[i])
        if np.linalg.norm(ta) < 1e-3:
            continue
        C.append(Rv[i].T @ Rv[i + 1] - np.eye(3))
        d.append(Rext @ tb - ta)
    if not C:
        raise ValueError("No valid translation constraints for extrinsic translation solve.")
    return np.vstack(C), np.concatenate(d), len(C)


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

