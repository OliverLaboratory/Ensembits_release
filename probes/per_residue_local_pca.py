"""Per-residue local-Kabsch PCA features.

Recipe (matches the manuscript's Token-Dynamics ANOVA exactly):

For each residue r in a single protein with P frames of Cα coords
`ca` of shape (P, L, 3):

  1. Pick the spatial neighbourhood of r in *frame 0*:
         ball_idx = { j : ‖ca[0, j] − ca[0, r]‖ ≤ ball_radius (Å) }.
     The ball composition is fixed by frame 0 — it is **not** recomputed
     per frame. Residues whose ball has fewer than 4 atoms are skipped
     (their feature row stays at zeros so downstream filters can find
     them by `feats[:, 0] == 0`).

  2. For each frame p ≥ 1, compute the Kabsch rotation `R_p` that
     aligns frame p's ball to frame 0's ball (centroids `m_c` and
     `ref_c`, then proper-rotation correction via det(Vt.T @ U.T)).
     The reference (frame 0) is left untouched.

  3. Apply `R_p` to *only residue r*'s coordinate in frame p:
         r_aligned[0] = ca[0, r]
         r_aligned[p] = (ca[p, r] − m_c) @ R_p.T + ref_c    (p ≥ 1)
     This isolates residue r's local fluctuation relative to its
     in-frame-0 neighbourhood, with rigid-body motion of the ball
     itself removed.

  4. Center the (P, 3) matrix and SVD it:
         X = r_aligned − r_aligned.mean(0)
         _, s, Vt = np.linalg.svd(X, full_matrices=False)
     Returns s = (s_1, s_2, s_3) — singular values in Å, **not**
     squared eigenvalues — and v_1 = Vt[0], the principal direction
     (sign-flipped so its largest |component| is positive, for
     reproducibility).

The 6-D feature row per residue is (s_1, s_2, s_3, v_1[0], v_1[1],
v_1[2]). Downstream uses `feats[:, 0]` (= s_1) as the response
variable for the η²-on-tokens ANOVA.
"""
from __future__ import annotations

import numpy as np


def local_kabsch_s1(
    ca: np.ndarray,
    ball_radius: float = 10.0,
    ref_frame: int = 0,
) -> np.ndarray:
    """Return per-residue local-Kabsch PCA features `(L, 6)`.

    Args:
        ca:           `(P, L, 3) float` Cα coordinates for one protein.
        ball_radius:  Å threshold for the spatial neighbourhood (paper
                      default: 10.0).
        ref_frame:    Frame index used as the reference (paper default: 0).

    Returns:
        `(L, 6) float32` of `(s_1, s_2, s_3, v_1[0], v_1[1], v_1[2])`
        per residue. Residues with fewer than 4 ball atoms are returned
        as zero rows.
    """
    if ca.ndim != 3 or ca.shape[-1] != 3:
        raise ValueError(f"expected ca shape (P, L, 3), got {ca.shape}")
    P, L, _ = ca.shape
    feats = np.zeros((L, 6), dtype=np.float32)
    ref = ca[ref_frame]

    for r in range(L):
        d = np.linalg.norm(ref - ref[r], axis=-1)
        ball_idx = np.where(d <= ball_radius)[0]
        if len(ball_idx) < 4:
            continue

        ref_ball = ref[ball_idx]
        ref_c = ref_ball.mean(axis=0)
        ref_centered = ref_ball - ref_c

        r_aligned = np.empty((P, 3), dtype=np.float32)
        r_aligned[ref_frame] = ref[r]

        ok = True
        for p in range(P):
            if p == ref_frame:
                continue
            m_ball = ca[p, ball_idx]
            m_c = m_ball.mean(axis=0)
            m_centered = m_ball - m_c
            H = m_centered.T @ ref_centered
            try:
                U, _, Vt = np.linalg.svd(H)
            except np.linalg.LinAlgError:
                ok = False
                break
            d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T
            r_aligned[p] = (ca[p, r] - m_c) @ R.T + ref_c
        if not ok:
            continue

        X = r_aligned - r_aligned.mean(axis=0, keepdims=True)
        try:
            _, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        v1 = Vt[0]
        if v1[np.argmax(np.abs(v1))] < 0:
            v1 = -v1
        feats[r, :3] = s[:3] if s.shape[0] >= 3 else np.pad(s, (0, 3 - s.shape[0]))
        feats[r, 3:] = v1
    return feats


def s1_per_residue(
    ca: np.ndarray,
    ball_radius: float = 10.0,
    ref_frame: int = 0,
) -> np.ndarray:
    """Convenience wrapper — return only `s_1` per residue, shape `(L,)`."""
    return local_kabsch_s1(ca, ball_radius=ball_radius,
                           ref_frame=ref_frame)[:, 0]
