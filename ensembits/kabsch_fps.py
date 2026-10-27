"""Farthest point sampling of MD frames by Kabsch-aligned Cα RMSD.

What it does
------------
Picks K frames out of a trajectory so that they are as different from each
other as possible, measuring the difference between two frames as their Cα
RMSD after optimal superposition (Kabsch). Superposition removes rigid-body
drift, so the selection reflects conformational change only.

Paper
-----
`appendix.tex` \\subsection{Dataset} (\\label{app:dataset}): frames are chosen
"by FPS on Cα RMSD" (BioEmu-CATH, and MISATO from the retrain on, code_fix T1.6).

Origin
------
Copied from `src/utils/kabsch_fps.py` (functions `kabsch_rmsd`,
`kabsch_fps_select`; the self-test block was dropped). Logic unchanged.
Used by `data_prep/build_misato_real_bb.py --fps kabsch`.

Used by the reported model: from the retrain on (code_fix E0, T1.6).
"""
from __future__ import annotations

import numpy as np


def kabsch_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    """Cα RMSD between two (L, 3) structures after optimal superposition."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    H = Yc.T @ Xc                                  # 3x3 cross-covariance
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T        # proper rotation (no reflection)
    Y_aligned = Yc @ R.T
    return float(np.sqrt(np.mean(np.sum((Xc - Y_aligned) ** 2, axis=-1))))


def kabsch_fps_select(ca_traj: np.ndarray, K: int, seed_idx: int = 0) -> list[int]:
    """Greedy farthest point sampling of K frames by Kabsch RMSD.

    ca_traj: (N, L, 3) Cα coordinates of N frames. Starts from frame `seed_idx`
    (default the first frame), then repeatedly adds the frame farthest from
    every frame chosen so far. Returns the K indices in time order.
    """
    N = ca_traj.shape[0]
    if K >= N:
        return list(range(N))
    selected = [int(seed_idx)]
    dists = np.array([kabsch_rmsd(ca_traj[seed_idx], ca_traj[j]) for j in range(N)],
                     dtype=np.float64)
    dists[seed_idx] = -np.inf
    for _ in range(K - 1):
        nxt = int(np.argmax(dists))
        if nxt in selected:
            break
        selected.append(nxt)
        new_d = np.array([kabsch_rmsd(ca_traj[nxt], ca_traj[j]) for j in range(N)],
                         dtype=np.float64)
        dists = np.minimum(dists, new_d)
        dists[nxt] = -np.inf
    return sorted(selected)
