"""Statistics of the token ANOVA: one-way ANOVA with omega^2, within-protein nulls, held-out R^2.

What it does
------------
Pure numpy functions used by `token_anova.py` (and its tests); no file I/O.

  * `oneway_anova(y, g)`: eta^2 = SS_between / SS_total, omega^2 =
    (SS_between - (M - 1) MS_within) / (SS_total + MS_within), F with df (M - 1, N - M) and
    the parametric p (kept only as `p_parametric_invalid_under_dependence`: residues of one
    protein are not independent, so the F distribution does not hold). M = number of
    distinct groups present; every group counts, singletons included (no pooling).
  * `circular_shift(g, blocks, rng)`: the within-protein null. Each protein's token
    sequence is rotated by a random offset (between `min_shift` and L - `min_shift`), so
    each protein keeps its own token multiset and the autocorrelation along the chain,
    and only the alignment between tokens and the target is broken. Proteins shorter than
    2 * min_shift + 1 residues are left unshifted (conservative: their alignment is kept).
  * `permutation_null(...)`: eta^2 (categorical) or OLS R^2 (continuous) under the
    circular-shift or the plain global-shuffle null; p = (b + 1) / (n + 1).
  * Held-out within-protein R^2: `center_by_protein`, `group_folds` (proteins grouped by a
    label, e.g. CATH topology, into k folds), `heldout_token_means` (token means of the
    protein-centred target fitted on the training folds; an unseen token predicts 0),
    `balanced_group_folds` (labels kept whole, folds balanced in residues),
    `heldout_ridge` (continuous features), `within_r2` (the prediction is centred per
    protein too, which uses no label; R^2 = 1 - sum (y_c - yhat_c)^2 / sum y_c^2 over all
    held-out residues) and `per_protein_ss` + `bootstrap_r2` (cluster bootstrap of the
    fixed cross-fitted predictions; with `clusters` a resampled CATH topology carries all
    its proteins and residues); `perm_summary` (b, n, p and "p < 10^-3 (0/1,000)").
  * Confounds: `enddist_bins`, `burial_deciles`, `ss_proxy` (C-alpha-only secondary
    structure: pseudo-torsion and bond angle), `confound_cells`, `backfit` (additive
    effects by backfitting; ridge components for continuous features) and
    `heldout_gain_over_confounds` (confounds vs confounds + method on the training folds:
    joint backfit, or the method fitted on the confound model's residual).

Paper
-----
`appendix.tex`, \\section{Tokens encode distinguishable local dynamics: an ANOVA test}
(\\label{app:token_anova}), Eq. (eq:eta2), the permutation null and Tables
tab:token_anova / tab:token_anova_baselines / tab:token_anova_controls; the statistics
follow the authors' decisions A1-A4 of `camera_ready/anova_audit.md` (2026-10-18).

Origin
------
New file (anova_audit A1-A4, 2026-10-18). The definitions follow the statistics audit
(`scratchpad/anova_audit/stats/anova_stats.py`, `assemble.py`, `extra.py`): omega^2,
circular shift with offsets in [5, L - 5], CATH-topology folds, centred held-out R^2,
chain-end bins {0, 1, 2, 3, 4-5, 6-9, >= 10}, burial = C-alpha count within 10 A in
frame 0 (deciles), the C-alpha secondary-structure proxy, backfitted confound cells.
Five deliberate changes: (i) the circular shift is vectorised (same distribution);
(ii) `heldout_gain_over_confounds` fits the method jointly with the confounds (the audit's
definition, primary) or, as a sensitivity mode, on the confound model's training residual;
(iii) the pseudo-torsion sign of `ss_proxy` is fixed (FIXED below); (iv) folds are
size-balanced (`balanced_group_folds`, implementation audit fix round 2); (v) the
bootstrap resamples CATH topologies (`bootstrap_weights(clusters=...)`, fix round 2).

Used by the reported model: yes (all statistics of app:token_anova).

Inputs / outputs
----------------
none (library).

How to run
----------
imported by `downstream/analyses/token_anova.py`; tested in `tests/test_token_anova_stats.py`.
"""
from __future__ import annotations

import numpy as np

# Chain-end distance bins (residues from the nearer chain end): 0, 1, 2, 3, 4-5, 6-9, >= 10.
ENDDIST_EDGES = (1, 2, 3, 4, 6, 10)


# ---------------------------------------------------------------- one-way ANOVA

def oneway_anova(y, g) -> dict:
    """One-way ANOVA of y on the groups g (any integer ids; every group counts).

    Paper: appendix.tex app:token_anova Eq. (eq:eta2). omega^2 is the bias-corrected
    effect size (A1, 2026-10-18); the parametric p is not valid under within-protein
    dependence and is labelled so.
    """
    from scipy import stats
    y = np.asarray(y, dtype=np.float64)
    _, inv = np.unique(np.asarray(g), return_inverse=True)
    inv = inv.ravel()
    N, M = len(y), int(inv.max()) + 1
    n = np.bincount(inv, minlength=M).astype(np.float64)
    s = np.bincount(inv, weights=y, minlength=M)
    gm = y.mean()
    ss_t = float(((y - gm) ** 2).sum())
    ss_b = float((n * (s / n - gm) ** 2).sum())
    ss_w = max(ss_t - ss_b, 0.0)
    df_b, df_w = M - 1, N - M
    out = dict(N=N, M=M, eta2=ss_b / ss_t if ss_t > 0 else float('nan'),
               ss_between=ss_b, ss_within=ss_w, ss_total=ss_t, df_between=df_b,
               df_within=df_w, n_singleton_groups=int((n == 1).sum()))
    if df_b > 0 and df_w > 0 and ss_w > 0:
        ms_w = ss_w / df_w
        out['F'] = (ss_b / df_b) / ms_w
        out['omega2'] = (ss_b - df_b * ms_w) / (ss_t + ms_w)
        out['p_parametric_invalid_under_dependence'] = float(stats.f.sf(out['F'], df_b, df_w))
    else:
        out['F'] = out['omega2'] = out['p_parametric_invalid_under_dependence'] = float('nan')
    return out


def eta2_from_sums(y, inv, n, gm, ss_t) -> float:
    """eta^2 for group ids `inv` (0..M-1) when the group sizes `n` are known."""
    s = np.bincount(inv, weights=y, minlength=len(n))
    ok = n > 0
    return float((n[ok] * (s[ok] / n[ok] - gm) ** 2).sum() / ss_t)


def ols_r2_no_intercept(X, y) -> float:
    """R^2 of y ~ X without intercept, against the centred total sum of squares (a
    profile row sums to 1, so the constant is in the span of X)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=1e-10)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------- protein blocks / nulls

def protein_blocks(prot) -> tuple[np.ndarray, np.ndarray]:
    """(start, length) per protein of a residue array whose rows are contiguous per
    protein (protein ids 0..P-1 in order). ValueError otherwise."""
    prot = np.asarray(prot)
    if len(prot) and (np.any(np.diff(prot) < 0) or prot[0] != 0
                      or len(np.unique(prot)) != int(prot.max()) + 1):
        raise ValueError('protein ids must be 0..P-1 with contiguous rows per protein')
    lens = np.bincount(prot)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return starts, lens


def circular_shift(g, blocks, rng, min_shift: int = 5) -> np.ndarray:
    """Rotate each protein's sequence of g by a random offset in [min_shift, L - min_shift]
    (np.roll semantics); proteins with L < 2 * min_shift + 1 keep their order."""
    starts, lens = blocks
    g = np.asarray(g)
    ok = lens >= 2 * min_shift + 1
    k = np.zeros(len(lens), dtype=np.int64)
    if ok.any():
        k[ok] = rng.integers(min_shift, lens[ok] - min_shift + 1)
    prot = np.repeat(np.arange(len(lens)), lens)
    pos = np.arange(len(g)) - starts[prot]
    src = starts[prot] + (pos - k[prot]) % lens[prot]
    return g[src]


def permutation_null(y, g=None, X=None, *, blocks, n: int, seed: int, kind: str,
                     min_shift: int = 5) -> np.ndarray:
    """Null distribution of eta^2 (g given) or of the no-intercept OLS R^2 (X given).

    kind 'circular': circular_shift within proteins; 'shuffle': global permutation.
    The continuous case shifts y instead of the rows of X (the same paired permutation).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    gm = y.mean()
    ss_t = float(((y - gm) ** 2).sum())
    idx = np.arange(len(y))

    def draw():
        return circular_shift(idx, blocks, rng, min_shift) if kind == 'circular' \
            else rng.permutation(idx)

    out = np.empty(n, dtype=np.float64)
    if g is not None:
        _, inv = np.unique(np.asarray(g), return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv).astype(np.float64)
        for i in range(n):
            out[i] = eta2_from_sums(y, inv[draw()], cnt, gm, ss_t)
        return out
    X = np.asarray(X, dtype=np.float64)
    G = X.T @ X
    A = np.linalg.pinv(G, rcond=1e-10)
    yy = float((y ** 2).sum())
    for i in range(n):
        yp = y[draw()]
        b = X.T @ yp
        ss_res = yy - float(b @ A @ b)
        out[i] = 1.0 - ss_res / (ss_t + 1e-12)
    return out


def perm_p(null, observed) -> float:
    """(b + 1) / (n + 1), b = number of null draws >= observed."""
    null = np.asarray(null)
    return float(((null >= observed).sum() + 1) / (len(null) + 1))


def perm_summary(null, observed) -> dict:
    """b, n, p = (b + 1)/(n + 1), the null mean / sd / max and a text form: "p < 10^-3
    (0/1,000)" style when no draw reaches the observed value (b = 0)."""
    null = np.asarray(null)
    b, n = int((null >= observed).sum()), int(len(null))
    p = (b + 1) / (n + 1)
    if b == 0:
        e = int(np.floor(np.log10(n))) if n else 0
        txt = f'p < 10^-{e} ({b}/{n:,})' if n else 'n/a'
    else:
        txt = f'p = {p:.3g} ({b}/{n:,})'
    return dict(b=b, n=n, p=float(p), text=txt, null_mean=float(null.mean()),
                null_sd=float(null.std()), null_max=float(null.max()))


# ---------------------------------------------------------------- held-out R^2

def center_by_protein(y, prot) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    m = np.bincount(prot, weights=y) / np.bincount(prot)
    return y - m[prot]


def group_folds(labels_per_protein, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold (0..k-1) of each protein; proteins sharing a label are in the same fold
    (labels shuffled with `seed`, then assigned round-robin). Kept for reference; the
    analysis uses `balanced_group_folds`."""
    ul, inv = np.unique(np.asarray(labels_per_protein), return_inverse=True)
    r = np.random.default_rng(seed).permutation(len(ul))
    fl = np.zeros(len(ul), dtype=np.int64)
    fl[r] = np.arange(len(ul)) % k
    return fl[inv.ravel()]


def balanced_group_folds(labels_per_protein, sizes_per_protein, k: int = 5) -> np.ndarray:
    """Fold (0..k-1) of each protein, proteins sharing a label in the same fold, folds
    balanced in total size (residues). Deterministic greedy: labels sorted by total size
    (descending, then label), each to the fold with the smallest running total (lowest
    fold id on ties).
    FIXED(anova_audit fix round 2, 2026-10-18): was `group_folds` (random round-robin over
    topologies; one fold held 32% of the residues)."""
    ul, inv = np.unique(np.asarray(labels_per_protein), return_inverse=True)
    inv = inv.ravel()
    tot = np.bincount(inv, weights=np.asarray(sizes_per_protein, dtype=np.float64), minlength=len(ul))
    order = sorted(range(len(ul)), key=lambda i: (-tot[i], str(ul[i])))
    run = np.zeros(k)
    fl = np.zeros(len(ul), dtype=np.int64)
    for i in order:
        f = int(np.argmin(run))
        fl[i] = f
        run[f] += tot[i]
    return fl[inv]


def _means(y, inv, M):
    n = np.bincount(inv, minlength=M)
    s = np.bincount(inv, weights=y, minlength=M)
    return np.where(n > 0, s / np.maximum(n, 1), 0.0)


def heldout_token_means(yc, g, folds) -> np.ndarray:
    """Cross-fitted prediction: the mean of yc over the training folds' residues of the
    same token; a token unseen in the training folds predicts 0."""
    _, inv = np.unique(np.asarray(g), return_inverse=True)
    inv = inv.ravel()
    M = int(inv.max()) + 1
    yhat = np.zeros(len(yc), dtype=np.float64)
    for f in np.unique(folds):
        tr = folds != f
        yhat[~tr] = _means(yc[tr], inv[tr], M)[inv[~tr]]
    return yhat


def ridge_fit(X, y, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def heldout_ridge(X, yc, folds, alpha: float = 1.0) -> np.ndarray:
    """Cross-fitted ridge prediction (no intercept: yc is centred, profiles sum to 1)."""
    X = np.asarray(X, dtype=np.float64)
    yhat = np.zeros(len(yc), dtype=np.float64)
    for f in np.unique(folds):
        tr = folds != f
        yhat[~tr] = X[~tr] @ ridge_fit(X[tr], yc[tr], alpha)
    return yhat


def per_protein_ss(yc, yhat, prot, n_prot: int | None = None):
    """(SSE_p, SST_p) per protein, the prediction centred per protein."""
    n_prot = int(prot.max()) + 1 if n_prot is None else n_prot
    hc = center_by_protein(yhat, prot)
    sse = np.bincount(prot, weights=(yc - hc) ** 2, minlength=n_prot)
    sst = np.bincount(prot, weights=yc ** 2, minlength=n_prot)
    return sse, sst


def within_r2(yc, yhat, prot) -> float:
    sse, sst = per_protein_ss(yc, yhat, prot)
    return float(1.0 - sse.sum() / sst.sum())


def bootstrap_weights(n_prot: int, B: int, seed: int, clusters=None) -> np.ndarray:
    """(B, n_prot) protein multiplicities of a bootstrap. Without `clusters`: proteins drawn
    with replacement. With `clusters` (cluster id per protein, e.g. CATH topology): clusters
    drawn with replacement, a drawn cluster brings all its proteins (cluster bootstrap;
    FIXED anova_audit fix round 2, 2026-10-18: the analysis uses CATH topologies)."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        W = np.zeros((B, n_prot), dtype=np.float64)
        for b in range(B):
            W[b] = np.bincount(rng.integers(0, n_prot, n_prot), minlength=n_prot)
        return W
    _, cid = np.unique(np.asarray(clusters), return_inverse=True)
    cid = cid.ravel()
    nc = int(cid.max()) + 1
    Wc = np.zeros((B, nc), dtype=np.float64)
    for b in range(B):
        Wc[b] = np.bincount(rng.integers(0, nc, nc), minlength=nc)
    return Wc[:, cid]


def bootstrap_r2(W, sse, sst) -> np.ndarray:
    """R^2 of every bootstrap draw (rows of W) from per-protein SSE / SST."""
    return 1.0 - (W @ sse) / (W @ sst)


def ci95(a) -> list[float]:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ---------------------------------------------------------------- confounds

def enddist_bins(res, L) -> np.ndarray:
    """Bin of the distance (in residues) to the nearer chain end."""
    d = np.minimum(np.asarray(res), np.asarray(L) - 1 - np.asarray(res))
    return np.digitize(d, ENDDIST_EDGES)


def burial_deciles(cn) -> np.ndarray:
    """Decile (0..9) of the C-alpha neighbour count over the residue set."""
    cn = np.asarray(cn)
    return np.digitize(cn, np.percentile(cn, np.arange(10, 100, 10)))


def ca_neighbour_count(ca0, radius: float = 10.0) -> np.ndarray:
    """Number of other C-alpha atoms within `radius` A of each C-alpha (one frame)."""
    D = np.linalg.norm(ca0[:, None] - ca0[None], axis=-1)
    return (D <= radius).sum(1) - 1


def ss_proxy(ca) -> np.ndarray:
    """C-alpha-only secondary structure (0 coil, 1 helix, 2 strand) of one frame (L, 3).

    Residue i takes the pseudo-torsion C-alpha(i-1, i, i+1, i+2) and the bond angle at i:
    helix 20-90 deg torsion and 80-105 deg angle; strand torsion < -110 or > 150 deg and
    105-150 deg angle; a call needs the same call at i-1 or i+1 (helix wins ties).
    """
    ca = np.asarray(ca, dtype=np.float64)
    L = len(ca)
    ss = np.zeros(L, dtype=np.int8)
    if L < 4:
        return ss
    b1 = ca[1:-2] - ca[:-3]; b2 = ca[2:-1] - ca[1:-2]; b3 = ca[3:] - ca[2:-1]
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2, axis=1, keepdims=True))
    # FIXED(anova_audit, 2026-10-18): the minus sign gives the IUPAC sign of the pseudo-
    #   torsion (right-handed alpha helix ~ +50 deg). The statistics audit's assemble.py
    #   had atan2(m1.n2, n1.n2) without it, so right-handed helices (~ -50 deg there) were
    #   never called helix and the strand window was mirrored.
    tau = -np.degrees(np.arctan2((m1 * n2).sum(1), (n1 * n2).sum(1)))
    v1 = ca[:-2] - ca[1:-1]; v2 = ca[2:] - ca[1:-1]
    cosang = (v1 * v2).sum(1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    t = np.full(L, np.nan); t[1:-2] = tau
    a = np.full(L, np.nan); a[1:-1] = ang
    with np.errstate(invalid='ignore'):
        hel = (t > 20) & (t < 90) & (a > 80) & (a < 105)
        strd = ((t < -110) | (t > 150)) & (a > 105) & (a < 150)

    def support(x):
        c = x.astype(np.int64)
        s = c.copy(); s[1:] += c[:-1]; s[:-1] += c[1:]
        return x & (s >= 2)
    h, s = support(hel), support(strd)
    ss[h] = 1
    ss[s & ~h] = 2
    return ss


def confound_cells(enddist_bin, burial_decile, ss) -> np.ndarray:
    """Joint cell id of (chain-end bin, burial decile, SS proxy), relabelled 0..C-1."""
    cell = (np.asarray(enddist_bin) * 10 + np.asarray(burial_decile)) * 3 + np.asarray(ss)
    return np.unique(cell, return_inverse=True)[1].ravel()


def _comp_fit(r, comp, ridge_alpha):
    kind, data = comp
    if kind == 'tokens':
        return _means(r, data, int(data.max()) + 1 if len(data) else 0)
    return ridge_fit(data, r, ridge_alpha)


def _comp_pred(eff, comp):
    kind, data = comp
    if kind == 'tokens':
        out = np.zeros(len(data))
        ok = data < len(eff)                 # ids unseen in the training rows -> 0
        out[ok] = eff[data[ok]]
        return out
    return np.asarray(data, dtype=np.float64) @ eff


def backfit(y, comps, n_iter: int = 15, ridge_alpha: float = 1.0):
    """Additive model y ~ mu + sum_j f_j by backfitting; a component is ('tokens', ids)
    (one effect per id) or ('ridge', X) (ridge on X). -> (mu, effects)."""
    mu = float(y.mean())
    eff = [None] * len(comps)
    parts = [np.zeros(len(y)) for _ in comps]
    for _ in range(n_iter):
        for j, c in enumerate(comps):
            r = y - mu - sum(parts[k] for k in range(len(comps)) if k != j)
            eff[j] = _comp_fit(r, c, ridge_alpha)
            parts[j] = _comp_pred(eff[j], c)
    return mu, eff


def backfit_predict(mu, eff, comps):
    return mu + sum(_comp_pred(e, c) for e, c in zip(eff, comps))


def _as_comp(kind, data):
    if kind == 'tokens':
        return ('tokens', np.unique(np.asarray(data), return_inverse=True)[1].ravel())
    return ('ridge', np.asarray(data, dtype=np.float64))


def _sub(comp, mask):
    return (comp[0], comp[1][mask])


def heldout_gain_over_confounds(yc, conf_groups, folds, methods: dict, n_iter: int = 15,
                                ridge_alpha: float = 1.0, mode: str = 'joint'):
    """Cross-fitted predictions of the confound model and of confounds + each method.

    conf_groups: list of integer arrays (e.g. [confound cells, amino acid]), additive.
    methods: {name: ('tokens', g) or ('ridge', X)}.
    mode 'joint' (default): the method is a further additive component, backfitted jointly
      with the confounds on the training folds (the statistics audit's definition: the
      incremental R^2 of the method given the confounds).
    mode 'residual': the confound model is fitted first; the method adds the token mean
      (ridge prediction) of the training residual (one backfitting step; conservative,
      shared variance stays with the confounds).
    Unseen token / cell -> 0. Returns (yhat_confounds, {name: yhat_confounds_plus_method}).
    """
    if mode not in ('joint', 'residual'):
        raise ValueError(mode)
    conf = [('tokens', np.unique(np.asarray(g), return_inverse=True)[1].ravel()) for g in conf_groups]
    mcomp = {name: _as_comp(kind, data) for name, (kind, data) in methods.items()}
    y0 = np.zeros(len(yc), dtype=np.float64)
    out = {name: np.zeros(len(yc), dtype=np.float64) for name in methods}
    for f in np.unique(folds):
        tr = folds != f; te = ~tr
        ctr = [_sub(c, tr) for c in conf]; cte = [_sub(c, te) for c in conf]
        mu, eff = backfit(yc[tr], ctr, n_iter, ridge_alpha)
        p_te = backfit_predict(mu, eff, cte)
        y0[te] = p_te
        resid = yc[tr] - backfit_predict(mu, eff, ctr)
        for name, mc in mcomp.items():
            if mode == 'residual':
                e = _comp_fit(resid, _sub(mc, tr), ridge_alpha)
                out[name][te] = p_te + _comp_pred(e, _sub(mc, te))
            else:
                mu2, eff2 = backfit(yc[tr], ctr + [_sub(mc, tr)], n_iter, ridge_alpha)
                out[name][te] = backfit_predict(mu2, eff2, cte + [_sub(mc, te)])
    return y0, out
