"""Token-conditioned ANOVA and held-out R^2 of per-residue motion on mdCATH-div (Ensembits and baselines).

What it does
------------
  1. Residue table of every mdCATH-div residue (manifest order, 2,442 domains, 308,293
     rows), computed from `ca_10` (P = 10 frames) of mdCATH-div by default:
       s_1 (and s_2): singular values of the (P, 3) C-alpha trajectory of residue r after
         Kabsch-aligning every frame on the 10 A C-alpha ball around r in frame 0
         (`per_residue_pca_features`, copied unchanged); a residue with fewer than 4
         C-alpha in its ball keeps the placeholder s_1 = 0 and is MASKED (53 residues,
         all near chain ends; the count is recorded);
       burial = number of other C-alpha within 10 A in frame 0; a C-alpha secondary
         structure proxy of frame 0 (`anova_stats.ss_proxy`).
     The table is cached in `output/token_anova/residue_table/` with a provenance sidecar
     (manifest sha256, source dir, definition version, content sha256) and verified on
     reuse. `--s1-source cache` instead takes s_1 / s_2 from the research cache
     `experiments/results/shared_states/token_pca_residues_aligned.npz`, which is refused
     without a provenance sidecar unless `--allow-unverified-s1-cache` (recorded). (The
     audit of 2026-10-18 found that cache equal to the recompute; it has no sidecar, so
     the default is the recompute.)
  2. Second target: the mdCATH-div all-frames RMSF label used by the RMSF probe
     (`rmsf_common.RMSF_LABEL_FILES['mdcath_div']['allframes']`, label sidecar and content
     hash checked as in `probe_rmsf_baselines.py`).
  3. Ensembits tokens of every checkpoint in `--variants` (level-1 token, M_1 = 2048) at
     P = 10 (frames 0-9) and P = 1 (frame 0 only, the frame every single-frame baseline
     reads; `p_inference` semantics of `tokenize_misato_and_eval.py`: the first P frames).
     From the mdCATH-div descriptor cache of the checkpoint's convention (the descriptor of
     a frame does not depend on the other frames, so frames [:P] of the P = 10 cache are
     the P-frame descriptor) or, with `--descriptor-source compute`, from the real
     backbones `bb_10[:P]`. Tokens are cached per variant, checkpoint hash, convention, P
     and descriptor source.
  4. Baselines (the caches must exist and carry a current provenance sidecar,
     `rmsf_common.check_cache_provenance`; a missing cache, or a protein of the residue set
     missing from a cache, is an error):
       Vote_3Di (paper rule `mini3di_baselines.vote_from_frame_tokens` on the K = 10
         mini3di tokens: plurality with count >= 2, else the centroid nearest to the frames'
         mean centroid, lowest letter id on exact ties), ProtProfileMD_K10 (the Foldseek /
         MMseqs2 profile of the first 10 frames, `protprofile_K10_mdcath.npz`; linear fit on
         the 20-D profile, authors' decision: OLS R^2 in-sample with adjusted R^2 as the
         omega^2 counterpart, ridge for the held-out R^2), 3Di (mini3di, frame 0; chain-end
         residues take Foldseek's invalid state 2, as Foldseek does; the count is
         recorded), AminoAseed, ESM3struct, ProToken (frame 0), amino acid (mdcath_real_bb
         sequence, X -> H; always read, it is part of the confound model; a letter outside
         the 20 is an error) and, with `random` in `--baselines`, one chance row per
         Ensembits row (its own tokens permuted over the residue set: same M and token
         frequencies, seeded).
  5. One residue set for every row: the requested proteins minus the s_1 placeholders.
     Any other gap (a missing token, feature, label or amino acid) is an error, so no
     residue leaves the set silently. No pooling and no dropping of rare tokens (A1);
     `--min-count` adds a diagnostic block with rare tokens pooled, the main numbers are
     always on raw tokens.
  6. Per method and target (s_1, all-frames RMSF):
       (a) one-way ANOVA: eta^2, omega^2 (M = distinct tokens used), F with df; the
           parametric p only as `p_parametric_invalid_under_dependence`; permutation
           p = (b + 1) / (n + 1) under the within-protein circular-shift null (`--n-perm`,
           default 1000; b, n, null mean / sd / max stored; "p < 10^-3 (0/1,000)" when
           b = 0) and, for reference in the JSON only, the plain-shuffle null.
       (b) within-protein held-out R^2: the target centred per protein; `--n-folds` (5)
           folds of proteins grouped by CATH topology (`representatives[pid]['topology']` of
           `data/external/mdcath_div/_manifest.json`), size-balanced by a deterministic greedy
           assignment (`anova_stats.balanced_group_folds`; the per-protein folds are stored
           in `<stem>.residues.npz`); token means fitted on the other folds, held-out
           residues predicted by their token mean (unseen token -> 0); the prediction is
           centred per protein (no label used); R^2 over all held-out residues. 95% CIs from
           a CATH-topology cluster bootstrap (`--n-boot` 1000; a drawn topology brings all
           its proteins) for every row and for Delta = Ensembits - best baseline of the
           same frame regime (authors' decision: P = 1 vs the single-frame tokenizers 3Di,
           AminoAseed, ESM3struct, ProToken; P = 10 vs the 10-frame baselines Vote_3Di,
           ProtProfileMD_K10; best on the point estimate, per target; paired bootstrap).
       (c) controls on the same residues: eta^2 of the target by CATH class, protein-length
           quintile, residue-position quintile (ported from the research script
           `test_token_dynamics_controls.py`), plus chain-end distance bins and protein
           identity (the share of variance between proteins).
       (d) gain over confounds (same folds): within-protein held-out R^2 of confounds +
           method minus that of the confounds alone (joint cell of chain-end bin x burial
           decile x SS proxy, plus amino acid). `gain_over_confounds` (primary, table.md):
           the method backfitted jointly with the confounds on the training folds (token
           effects; ridge for ProtProfileMD). `gain_over_confounds_residual_sensitivity`
           (JSON only): the method fitted on the confound model's training residual.
           Paired cluster-bootstrap CIs.
  7. Writes the JSON (`--out`; code file hashes, git HEAD and each checkpoint's lambda =
     consistency_mse_weight recorded), its provenance sidecar, `<stem>.residues.npz` (the
     residue set with the raw token ids of every categorical row, targets and folds),
     `<stem>.nulls.npz` (null distributions), `<stem>.table.md` (summary tables) and,
     unless `--no-figures`, the ANOVA figure per variant (`plot_anova_test.py`) in
     `--fig-dir/<variant>/`. `--export-token-cache` writes, per variant, the research cache
     format (tokens, feats, pids, ress) with RAW P = 10 token ids (plus tokens_P1).

Paper
-----
  * `appendix.tex`, \\section{Tokens encode distinguishable local dynamics: an ANOVA test}
    (\\label{app:token_anova}): \\paragraph{Setup.}, Eq. (eq:eta2), the F-statistic and
    permutation-null paragraphs, Table tab:token_anova (`--feature s1_over_s2` for the
    s_1/s_2 row), Table tab:token_anova_controls, \\paragraph{Comparison to existing
    structural tokenizers.} Table tab:token_anova_baselines, Figure fig:anova_test.
  * `2_results.tex`, "We also perform an ANOVA test ..." (main-text claim).
  The statistics follow the authors' decisions A1-A5 of `camera_ready/anova_audit.md`
  (2026-10-18); the paper text is updated separately.

Origin
------
Merged from research scripts, then changed by the FIXED items below:
  * `submission_exp/src/_anova_all_variants.py` (Ensembits variants);
  * `experiments/shared_states_v2/test_token_dynamics_baselines.py` (baselines);
  * `experiments/shared_states_v2/plot_token_pca_correlation.py` (per_residue_pca_features);
  * `experiments/shared_states_v2/test_token_dynamics.py` (s_1/s_2, permutation null; its
    figure is ported to `plot_anova_test.py`);
  * `experiments/shared_states_v2/test_token_dynamics_controls.py` (controls; its
    direction-alignment test is not ported: no paper element uses it).
FIXED(anova_audit A1-A5 + code fixes, 2026-10-18): removed the rare-token pooling, the
count filter and the residue-set modes `intersection` / `per_method` (with no count filter
the per-method sets are the coverage sets; the submitted per-method numbers are
reproduced by the audit scripts); removed the `rare_id` group and the pooled ids from the
export; the default checkpoint is `output/ensembits_bm_lam10_rv` (was the legacy
`ensembits.paths.CKPT_DIR`); a checkpoint / encode error exits non-zero (was an `err:` row
with exit 0); Vote_3Di uses the paper rule (was a plain argmax); ProtProfileMD_K10 reads
the Foldseek / MMseqs2 profile cache (was a 20-bin histogram of the mini3di tokens); a
missing baseline cache is an error (was SKIP, which changed the residue set); baseline,
s_1 and RMSF inputs are provenance-checked; SS_between counts singleton groups (was: groups
with one residue were left out of SS_between); p_perm = (b + 1) / (n + 1) (was b / n);
added omega^2, the circular-shift null, the held-out R^2 with bootstrap CIs, P = 1 rows,
the RMSF target, the controls and the gain over confounds.

Used by the reported model: yes (the numbers of app:token_anova).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/external/mdcath_div/_manifest.json, <pid>.npz (ca_10)        (layout, s_1, burial, SS, CATH)
          output/phase_b/cached_desc/rmsf_div_allframes.npy (+ sidecar)     (RMSF target)
          output/phase_b/cached_desc/esm3desc_dihedral[_iupac]_k16_P10.npy   (--descriptor-source cache)
          data/cached_descriptors/mdcath_real_bb/<pid>.npz (bb_10, seq)       (compute; amino acid)
          data/cached_descriptors/{mini3di,aminoaseed,protoken,esm3struct}_tokens_mdcath.npz,
          mini3di_tokens_K10_mdcath.npz, protprofile_K10_mdcath.npz (+ sidecars, codebooks)
          <variant>/config.json, best.pt, stats.npz
  writes  --out JSON (+ .provenance.json), <stem>.residues.npz, <stem>.nulls.npz, <stem>.table.md,
          output/token_anova/figures/<variant>/ANOVAtest.{pdf,png},
          output/token_anova/residue_table/residue_table__<manifest sha12>__<definition sha8>.npz (+ sidecar),
          output/token_anova/token_cache/tokens_mdcath_flat__<variant>__<sha12>__<conv>__P<P>__<src>.npz

How to run
----------
    # the chosen λ = 10 checkpoint (default variant output/ensembits_bm_lam10_rv)
    python downstream/analyses/token_anova.py --out output/token_anova/token_anova.json
    # all lambda variants (baselines computed once; one row pair per variant)
    python downstream/analyses/token_anova.py --variants lambda_sweep \\
        --out output/token_anova/token_anova_lambda.json
    # smoke: 40 proteins, few draws
    python downstream/analyses/token_anova.py --max-proteins 40 --n-perm 50 --n-boot 50 \\
        --out <scratch>/anova_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _common.py, anova_stats.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # downstream/: rmsf_common
from rmsf_common import (normalize_mdcath_real_bb_seq, check_cache_provenance,  # noqa: E402
                         check_token_ids, check_label_hash, check_flat_label_sidecar,
                         CacheError, RMSF_LABEL_FILES)
from ensembits.paths import REPO_ROOT as ROOT, MDCATH_DIV_SRC  # noqa: E402
from ensembits.descriptor import with_convention  # noqa: E402
from ensembits.featurize import file_sha256  # noqa: E402
from ensembits.data_versions import write_provenance, read_provenance, provenance_path  # noqa: E402
from _common import (resolve_model_dir, ckpt_hash, cfg_get, checkpoint_settings,  # noqa: E402
                     verify_sidecar, find_sidecar)
import anova_stats as S  # noqa: E402

S1_CACHE = ROOT / 'experiments' / 'results' / 'shared_states' / 'token_pca_residues_aligned.npz'
PHASE_B = ROOT / 'output' / 'phase_b' / 'cached_desc'
TOKENS_DIR = ROOT / 'data' / 'cached_descriptors'
MDCATH_REAL_BB = TOKENS_DIR / 'mdcath_real_bb'
OUT_ROOT = ROOT / 'output' / 'token_anova'
TOKEN_CACHE_DIR = OUT_ROOT / 'token_cache'
RESIDUE_TABLE_DIR = OUT_ROOT / 'residue_table'
FIG_DIR = OUT_ROOT / 'figures'
RMSF_ALLFRAMES = RMSF_LABEL_FILES['mdcath_div']['allframes']
DATASET = 'mdcath_div'
SCHEMA = 'token_anova-2026-10-18'
# Definition of the residue table (bump when per_residue_pca_features / burial / SS change).
RESIDUE_TABLE_VERSION = 'ca_10 P=10; s1,s2 aligned 10A ball frame 0; cn10 frame 0; ss_proxy v2 (IUPAC sign)'

# FIXED(recon2 A.7, 2026-10-25): defaults switched to the chosen λ = 10 _rv run
#   (decisions_20261023 R3). lambda_sweep alias uses _rv runs.
DEFAULT_VARIANTS = ['output/ensembits_bm_lam10_rv']
VARIANT_ALIASES = {'lambda_sweep': ['output/ensembits_bm_e0_rv', 'output/ensembits_bm_lam0_rv',
                                    'output/ensembits_bm_lam0p05_rv', 'output/ensembits_bm_lam0p25_rv',
                                    'output/ensembits_bm_lam0p5_rv', 'output/ensembits_bm_lam1_rv']}
P_FRAMES = (10, 1)
# Frame regimes of the Delta comparison (authors' decision, fix round 2, 2026-10-18):
# Ensembits P = 1 vs the single-frame tokenizers, P = 10 vs the 10-frame baselines.
# The amino acid row is shown but is not a tokenizer baseline.
REGIMES = {1: ('single-frame', ['3di', 'aminoaseed', 'esm3struct', 'protoken']),
           10: ('10-frame', ['vote_3di', 'protprofile_K10'])}

# Baseline caches (pid -> per-residue array, manifest pids), their provenance frames, and
# whether the feature uses the sequence (rmsf_common.check_cache_provenance).
BASELINE_FILES = {
    '3di':             ('mini3di_tokens_mdcath.npz',      'frame 0',           False, None),
    'aminoaseed':      ('aminoaseed_tokens_mdcath.npz',   'frame 0',           False, 'aminoaseed_codebook.npy'),
    'protoken':        ('protoken_tokens_mdcath.npz',     'frame 0',           False, 'protoken_codebook.npy'),
    'esm3struct':      ('esm3struct_tokens_mdcath.npz',   'frame 0',           False, 'esm3struct_codebook.npy'),
    'vote_3di':        ('mini3di_tokens_K10_mdcath.npz',  'all 10',            False, None),
    'protprofile_K10': ('protprofile_K10_mdcath.npz',     'first 10 of bb_10', True,  None),
}
BASELINE_KEYS = ('vote_3di', 'protprofile_K10', '3di', 'aminoaseed', 'esm3struct', 'protoken',
                 'aa', 'random')
BASELINE_NAMES = {'vote_3di': 'Vote_3Di', 'protprofile_K10': 'ProtProfileMD_K10',
                  '3di': '3Di (frame 0)', 'aminoaseed': 'AminoAseed', 'esm3struct': 'ESM3struct',
                  'protoken': 'ProToken', 'aa': 'amino acid', 'random': 'random'}
MINI3DI_INVALID_STATE = 2       # Foldseek's state for residues without a 3Di descriptor


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_savez(path: Path, compressed=True, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- ANOVA core (kept API)

def eta_squared(values, groups):
    """One-way ANOVA. Returns (eta^2, F, df_b, df_w, p, M).

    Paper: appendix.tex app:token_anova Eq. (eq:eta2). p is the parametric F p-value
    (invalid under within-protein dependence; not used by the paper outputs).
    FIXED(anova_audit, 2026-10-18): singleton groups count in SS_between (was skipped).
    """
    a = S.oneway_anova(values, groups)
    return (a['eta2'], a['F'], a['df_between'], a['df_within'],
            a['p_parametric_invalid_under_dependence'], a['M'])


def eta_squared_fast(values, groups) -> float:
    return S.oneway_anova(values, groups)['eta2']


ols_r2_no_intercept = S.ols_r2_no_intercept


def pool_rare_tokens(groups: np.ndarray, min_count: int):
    """DIAGNOSTIC only (--min-count): tokens with fewer than `min_count` residues are
    merged into one group with id -2 (outside every token range, so no collision).
    Returns (pooled groups, n residues pooled, n tokens pooled)."""
    g = np.asarray(groups).copy()
    uniq, cnt = np.unique(g, return_counts=True)
    rare = uniq[cnt < min_count]
    sel = np.isin(g, rare)
    g[sel] = -2
    return g, int(sel.sum()), int(len(rare))


# ---------------------------------------------------------------- s_1 feature

def per_residue_pca_features(ca: np.ndarray, aligned: bool = False,
                             ball_radius: float = 10.0) -> np.ndarray:
    """ca: (P, L, 3). Returns features: (L, 6) = (s1, s2, s3, v1).

    Per residue r:
      - if aligned=False: take raw ca[:, r, :] (P, 3) -> SVD.
      - if aligned=True : Kabsch-align all P frames using the 10 Å Cα
        ball around r in frame 0, then take the aligned r's (P, 3) -> SVD.
        This removes rigid-body motion and isolates local fluctuations.
    Sign-align v1 so the largest |component| is positive.

    Paper: appendix.tex app:token_anova \\paragraph{Setup.} (s_1(r)).
    Copied unchanged from experiments/shared_states_v2/plot_token_pca_correlation.py.
    (A residue with < 4 Cα in its ball keeps the placeholder 0; masked by the caller.)
    """
    P, L, _ = ca.shape
    feats = np.zeros((L, 6), dtype=np.float32)
    ref = ca[0]

    for r in range(L):
        if aligned:
            d = np.linalg.norm(ref - ref[r], axis=-1)
            ball_idx = np.where(d <= ball_radius)[0]
            if len(ball_idx) < 4:
                continue
            ref_ball = ref[ball_idx]
            ref_c = ref_ball.mean(axis=0)
            ref_centered = ref_ball - ref_c
            r_aligned = np.empty((P, 3), dtype=np.float32)
            r_aligned[0] = ref[r]
            ok = True
            for p in range(1, P):
                m_ball = ca[p, ball_idx]
                m_c = m_ball.mean(axis=0)
                m_centered = m_ball - m_c
                H = m_centered.T @ ref_centered
                try:
                    U, _, Vt = np.linalg.svd(H)
                except np.linalg.LinAlgError:
                    ok = False; break
                d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
                R = Vt.T @ np.diag([1.0, 1.0, d_sign]) @ U.T
                r_aligned[p] = (ca[p, r] - m_c) @ R.T + ref_c
            if not ok:
                continue
            X = r_aligned - r_aligned.mean(axis=0, keepdims=True)
        else:
            X = ca[:, r, :] - ca[:, r, :].mean(axis=0, keepdims=True)

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


# ---------------------------------------------------------------- mdCATH-div layout

def manifest() -> dict:
    return json.loads((MDCATH_DIV_SRC / '_manifest.json').read_text())['representatives']


def manifest_pids() -> list[str]:
    return list(manifest().keys())


def manifest_layout() -> dict[str, tuple[int, int]]:
    """pid -> (start row, L) in the flat mdCATH-div layout (manifest order)."""
    L_by_pid = {}
    cur = 0
    for pid in manifest_pids():
        L = np.load(MDCATH_DIV_SRC / f'{pid}.npz', allow_pickle=False)['ca_10'].shape[1]
        L_by_pid[pid] = (cur, L)
        cur += L
    return L_by_pid


def map_flat_to_pid_res(pids: np.ndarray, ress: np.ndarray, L_by_pid) -> np.ndarray:
    """For each (pid, res) row, its index in the flat mdCATH-div layout, or -1."""
    out = np.full(len(pids), -1, dtype=np.int64)
    for i in range(len(pids)):
        pid = str(pids[i])
        if pid not in L_by_pid: continue
        s, L = L_by_pid[pid]
        r = int(ress[i])
        if 0 <= r < L:
            out[i] = s + r
    return out


# ---------------------------------------------------------------- residue table

def _residue_table_one(pid: str):
    ca = np.load(MDCATH_DIV_SRC / f'{pid}.npz', allow_pickle=False)['ca_10'].astype(np.float32)
    feats = per_residue_pca_features(ca, aligned=True)
    cn = S.ca_neighbour_count(ca[0].astype(np.float64))
    return pid, feats, cn.astype(np.int32), S.ss_proxy(ca[0])


def ca10_sha256(pids) -> str:
    """sha256 over (pid, dtype, shape, bytes) of `ca_10` of every pid, in the given order."""
    import hashlib
    h = hashlib.sha256()
    for pid in pids:
        a = np.ascontiguousarray(np.load(MDCATH_DIV_SRC / f'{pid}.npz', allow_pickle=False)['ca_10'])
        h.update(pid.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def residue_table(L_by_pid, workers: int, cache_dir: Path, rebuild: bool = False) -> tuple[dict, dict]:
    """Flat arrays (manifest order) feats (N, 6), cn10, ss for every manifest pid, cached
    with a provenance sidecar (verified on reuse). -> (arrays, provenance record).
    FIXED(anova_audit fix round 2, 2026-10-18): the cache is keyed on the content of every
    ca_10 array (was the manifest sha256 only)."""
    import hashlib
    man_sha = file_sha256(MDCATH_DIV_SRC / '_manifest.json')
    ca_sha = ca10_sha256(list(L_by_pid))
    total = sum(L for _, L in L_by_pid.values())
    def_sha = hashlib.sha256(RESIDUE_TABLE_VERSION.encode()).hexdigest()[:8]
    path = Path(cache_dir) / f'residue_table__{ca_sha[:12]}__{def_sha}.npz'
    want = {'manifest_sha256': man_sha, 'ca10_sha256': ca_sha, 'definition': RESIDUE_TABLE_VERSION,
            'n_rows': total, 'n_pids': len(L_by_pid)}
    if path.exists() and not rebuild:
        side = read_provenance(path)
        if side is None or any(side.get(k) != v for k, v in want.items()) \
                or side.get('sha256') != file_sha256(path):
            raise SystemExit(f'{path} does not match its sidecar / the manifest / the definition; '
                             f'delete it or pass --rebuild-residue-table')
        z = np.load(path)
        return {k: z[k] for k in z.files}, {'file': str(path), 'sha256': side['sha256'], **want}
    from multiprocessing import Pool
    t0 = time.time()
    pids = list(L_by_pid)
    with Pool(max(1, workers)) as pool:
        res = pool.map(_residue_table_one, pids, chunksize=4)
    feats = np.zeros((total, 6), np.float32); cn = np.zeros(total, np.int32); ss = np.zeros(total, np.int8)
    for pid, f, c, s_ in res:
        s0, L = L_by_pid[pid]
        if f.shape[0] != L:
            raise SystemExit(f'{pid}: ca_10 has {f.shape[0]} residues, the layout {L}')
        feats[s0:s0 + L] = f; cn[s0:s0 + L] = c; ss[s0:s0 + L] = s_
    arr = dict(feats=feats, cn10=cn, ss=ss)
    _atomic_savez(path, **arr)
    sha = file_sha256(path)
    write_provenance(path, {**want, 'sha256': sha, 'source': str(MDCATH_DIV_SRC),
                            'builder': 'downstream/analyses/token_anova.py',
                            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'seconds': round(time.time() - t0, 1)})
    print(f'  [table] {total:,} residues in {time.time() - t0:.0f}s -> {path}', flush=True)
    return arr, {'file': str(path), 'sha256': sha, **want}


def load_s1_research_cache(cache: Path, L_by_pid, total: int, allow_unverified: bool):
    """(feats (total, 6), mapped mask, record) from the research s_1 cache."""
    if not cache.exists():
        raise SystemExit(f's_1 cache not found: {cache}')
    side = read_provenance(cache) or (json.loads(find_sidecar(cache).read_text())
                                      if find_sidecar(cache) else None)
    if side is None and not allow_unverified:
        raise SystemExit(f'{cache} has no provenance sidecar; use the default --s1-source compute '
                         f'or pass --allow-unverified-s1-cache')
    d = np.load(cache, allow_pickle=True)
    # Only feats / pids / ress are used; its "tokens" column is from an old checkpoint.
    pids = np.array([str(p) for p in d['pids']]); ress = d['ress'].astype(np.int64)
    fi = map_flat_to_pid_res(pids, ress, L_by_pid)
    feats = np.zeros((total, 6), np.float32); ok = np.zeros(total, bool)
    feats[fi[fi >= 0]] = d['feats'][fi >= 0]; ok[fi[fi >= 0]] = True
    return feats, ok, {'file': str(cache), 'sha256': file_sha256(cache), 'sidecar': side is not None,
                       'unverified': side is None}


# ---------------------------------------------------------------- Ensembits tokens

def _coords_mdcath(pid, P, allow_ideal=False):
    """((P, L, 3, 3) N/CA/C of the first P frames, is_ideal) for one mdCATH-div pid, or
    (None, False). The real backbone bb_10 of mdcath_real_bb; the ideal-geometry fallback
    (ca_10 + ideal N/C) only with --allow-ideal and only when no real file exists."""
    rb = MDCATH_REAL_BB / f'{pid}.npz'
    if rb.exists():
        d = np.load(rb, allow_pickle=False)
        if 'bb_10' in d.files:
            bb = np.asarray(d['bb_10'], dtype=np.float32)
            if bb.shape[0] >= P:
                bb = bb[:P]
                return np.stack([bb[:, :, 0, :], bb[:, :, 1, :], bb[:, :, 2, :]], axis=2), False
        return None, False
    if not allow_ideal:
        return None, False
    src = MDCATH_DIV_SRC / f'{pid}.npz'
    if not src.exists():
        return None, False
    ca = np.asarray(np.load(src, allow_pickle=False)['ca_10'], dtype=np.float32)
    if ca.shape[0] < P:
        return None, False
    ca = ca[:P]
    from ensembits.ideal_backbone import _reconstruct_n_c_ideal
    n_list, c_list = [], []
    for fp in range(P):
        n_, c_ = _reconstruct_n_c_ideal(ca[fp])
        n_list.append(n_); c_list.append(c_)
    return np.stack([np.stack(n_list, 0), ca, np.stack(c_list, 0)], axis=2), True


def _encode_rows(ens, desc_rows, device, p_frames=None, bs=16384):
    """Level-1 tokens of descriptor rows (N, P_cache, 256), using frames [:p_frames]."""
    import torch
    mu = ens.mean.astype(np.float32); std = ens.std.astype(np.float32)
    out = np.zeros(desc_rows.shape[0], dtype=np.int64)
    with torch.no_grad():
        for s in range(0, desc_rows.shape[0], bs):
            x = np.asarray(desc_rows[s:s + bs], dtype=np.float32)
            if p_frames is not None:
                x = x[:, :p_frames]
            x = (x - mu) / std
            x = torch.tensor(x, device=device)
            out[s:s + bs] = ens.model.encode(x).cpu().numpy().astype(np.int64)
    return out


def load_variant_tokens(variant_dir: Path, L_by_pid, pids_needed, args,
                        p_frames: int | None = None) -> tuple[np.ndarray, dict]:
    """Flat (N_rows,) level-1 tokens of the mdCATH-div residues (manifest order) from the
    first `p_frames` frames (default: the checkpoint's P), -1 for rows not tokenized, plus
    provenance. Any error propagates (exit non-zero)."""
    import torch
    from ensembits.featurize import load_model
    ens = load_model(str(variant_dir), device=args.device)
    cfg = ens.config
    conv, cutoff = checkpoint_settings(ens)
    mode = cfg_get(cfg, 'descriptor_mode', 'esm3desc_dihedral')
    k = cfg_get(cfg, 'k', 16)
    P_cfg = int(cfg_get(cfg, 'num_prototypes', 10))
    P = P_cfg if p_frames is None else int(p_frames)
    if not 1 <= P <= 10:
        raise ValueError(f'p_frames must be in 1..10, got {P}')
    if mode != 'esm3desc_dihedral' or int(cfg_get(cfg, 'input_dim', 256)) != 256:
        raise ValueError(f'only the 256-D esm3desc_dihedral descriptor is supported, got {mode!r}')
    sha = ckpt_hash(variant_dir)
    total = sum(L for _, L in L_by_pid.values())
    full = set(pids_needed) == set(L_by_pid)
    allow_ideal = bool(getattr(args, 'allow_ideal', False))
    prov = {'ckpt_sha256_12': sha, 'dihedral_convention': conv, 'chain_break_cutoff': cutoff,
            'descriptor_source': args.descriptor_source, 'allow_ideal': allow_ideal,
            'p_frames': P, 'frames_read': f'frames 0-{P - 1}' if P > 1 else 'frame 0'}

    meta, desc_path, desc_sc_sha, ideal_pids = None, None, '', []
    if args.descriptor_source == 'cache':
        # The cache file name follows the checkpoint's convention; its sidecar must record the
        # same convention / cutoff / row count (code_fix P20, mdCATH re-extraction N3).
        desc_path = PHASE_B / with_convention(f'{mode}_k{k}_P10.npy', conv)
        if not desc_path.exists():
            raise SystemExit(f'descriptor cache {desc_path} missing; build it with '
                             f'data_prep/build_esm3desc_dihedral_perdataset.py --dataset mdcath '
                             f'--dihedral-convention {conv}, or use --descriptor-source compute')
        meta = verify_sidecar(desc_path, convention=conv, chain_break_cutoff=cutoff, n_rows=total)
        if meta is None:
            if not getattr(args, 'allow_legacy_cache', False):
                raise SystemExit(f'no sidecar next to {desc_path}; its convention and ideal-'
                                 f'geometry domains (IDEAL_PIDS) are unknown. Rebuild it with '
                                 f'data_prep/build_esm3desc_dihedral_perdataset.py --dataset mdcath, '
                                 f'or pass --allow-legacy-cache')
            print(f'  [warn] --allow-legacy-cache: no sidecar next to {desc_path.name}')
            prov['allow_legacy_cache'] = True
            prov['descriptor_sidecar_sha256'] = None
        else:
            desc_sc_sha = file_sha256(find_sidecar(desc_path))
            ideal_pids = sorted(meta.get('IDEAL_PIDS') or [])
            prov['descriptor_sidecar_sha256'] = desc_sc_sha
        prov['descriptor_ideal_pids'] = ideal_pids

    # Token cache named by variant, checkpoint hash, convention, P and descriptor source
    # (FIXED anova_audit A3, 2026-10-18: P added; P = 10 and P = 1 never share a file).
    if args.descriptor_source == 'cache':
        src_tag = f'desc{desc_sc_sha[:12]}' if desc_sc_sha else 'desclegacy'
    else:
        src_tag = 'compute'
    cache = Path(args.token_cache_dir) / (f'tokens_mdcath_flat__{variant_dir.name}__{sha}__{conv}'
                                          f'__P{P}__{src_tag}{"__ideal" if allow_ideal else ""}.npz')
    want = {'ckpt_sha256_12': sha, 'dihedral_convention': conv,
            'descriptor_source': args.descriptor_source,
            'descriptor_sidecar_sha256': desc_sc_sha,
            'ideal_pids': json.dumps(ideal_pids), 'allow_ideal': str(allow_ideal),
            'p_frames': str(P)}
    if full and cache.exists() and not args.no_token_cache:
        z = np.load(cache, allow_pickle=False)
        bad = [f for f, v in want.items() if f not in z.files or str(z[f]) != v]
        if bad or z['tokens'].shape[0] != total:
            raise SystemExit(f'{cache} does not match this checkpoint / descriptor cache / layout '
                             f'(fields {bad or ["tokens"]}); delete it')
        prov['token_cache'] = str(cache)
        return z['tokens'], prov

    tokens = np.full(total, -1, dtype=np.int64)
    if args.descriptor_source == 'cache':
        desc = np.load(desc_path, mmap_mode='r')
        if desc.shape[0] != total:
            raise SystemExit(f'{desc_path} has {desc.shape[0]} rows, the manifest layout {total}')
        ideal_rows = set(ideal_pids) & set(pids_needed)
        # FIXED(anova_audit fix round 2, 2026-10-18): ideal-geometry domains are an error
        #   unless --allow-ideal (was: a warning, and their rows left the residue set).
        if ideal_rows and not allow_ideal:
            raise SystemExit(f'{len(ideal_rows)} domains of the descriptor cache use ideal geometry '
                             f'({sorted(ideal_rows)[:5]}); rebuild the cache, or pass --allow-ideal')
        prov['descriptor_cache'] = str(desc_path.relative_to(ROOT)) if desc_path.is_relative_to(ROOT) \
            else desc_path.name
        if full:
            tokens[:] = _encode_rows(ens, desc, args.device, P)
        else:
            for pid in pids_needed:
                s, L = L_by_pid[pid]
                tokens[s:s + L] = _encode_rows(ens, desc[s:s + L], args.device, P)
        prov['ideal_pids_used'] = sorted(ideal_rows)
    else:
        from ensembits.descriptor import _descriptor
        mu = ens.mean.astype(np.float32); std = ens.std.astype(np.float32)
        n_ideal = 0
        for i, pid in enumerate(pids_needed):
            s, L = L_by_pid[pid]
            c, ideal = _coords_mdcath(pid, P, allow_ideal)
            if c is None or c.shape[1] != L:
                # FIXED(anova_audit fix round 2, 2026-10-18): error (was: skipped silently)
                raise SystemExit(f'{pid}: no real backbone bb_10 (or L != {L}); rebuild '
                                 f'mdcath_real_bb or pass --allow-ideal')
            n_ideal += int(ideal)
            with torch.no_grad():
                # Paper: appendix.tex app:descriptor-esm3; ideal-geometry backbones are
                # treated as connected (cutoff None), as in the cache builder.
                d = _descriptor(torch.from_numpy(c), args.device, convention=conv,
                                chain_break_cutoff=None if ideal else cutoff).numpy()
                x = (d.transpose(1, 0, 2) - mu) / std                         # (L, P, 256)
                tokens[s:s + L] = ens.model.encode(
                    torch.as_tensor(x, dtype=torch.float32, device=args.device)).cpu().numpy()
            if (i + 1) % 200 == 0:
                print(f'  [tok] {i + 1}/{len(pids_needed)}', flush=True)
        prov['n_ideal_geometry_pids'] = n_ideal
    if full and not args.no_token_cache:
        _atomic_savez(cache, tokens=tokens, **want)
        prov['token_cache'] = str(cache)
    return tokens, prov


# ---------------------------------------------------------------- baselines

def _flat_from_npz(z, what, L_by_pid, pids_needed, total, shape_tail=()):
    """Per-pid arrays of an npz placed in the flat layout. A pid of the layout absent from
    the cache or a length mismatch is an error (FIXED anova_audit fix round 2, 2026-10-18:
    a missing pid used to leave the residue set silently)."""
    out = np.full((total,) + shape_tail, -1, dtype=np.float32 if shape_tail else np.int64)
    valid = np.zeros(total, dtype=bool)
    missing = []
    files = set(z.files)
    for pid in pids_needed:
        s, L = L_by_pid[pid]
        if pid not in files:
            missing.append(pid); continue
        a = z[pid]
        if a.shape[0] != L:
            raise SystemExit(f'{what}: {pid} has {a.shape[0]} residues, the layout {L}')
        out[s:s + L] = a
        valid[s:s + L] = True
    if missing:
        raise SystemExit(f'{what}: {len(missing)} proteins of the residue set missing from the '
                         f'cache ({missing[:5]}); rebuild it')
    return out, valid, missing


def amino_acids(L_by_pid, pids_needed) -> np.ndarray:
    """Amino-acid index (0..19) of every residue: mdcath_real_bb/<pid>.npz['seq'], X (CHARMM
    His) -> H (Q29). A missing file or a letter outside the 20 is an error (FIXED anova_audit
    fix round 2, 2026-10-18: was -1, which dropped the residue silently)."""
    AA20 = "ACDEFGHIKLMNPQRSTVWY"
    aa_index = {c: i for i, c in enumerate(AA20)}
    total = sum(L for _, L in L_by_pid.values())
    g = np.full(total, -1, dtype=np.int64)
    for pid in pids_needed:
        s, L = L_by_pid[pid]
        rb = MDCATH_REAL_BB / f'{pid}.npz'
        if not rb.exists():
            raise SystemExit(f'amino acid: {rb} missing (data_prep/build_mdcath_real_bb.py)')
        seq = normalize_mdcath_real_bb_seq(str(np.load(rb, allow_pickle=False)['seq']), pid)
        if len(seq) != L:
            raise SystemExit(f'amino acid: {pid} sequence has {len(seq)} residues, layout {L}')
        bad = sorted({c for c in seq if c not in aa_index})
        if bad:
            raise SystemExit(f'amino acid: {pid} has letters outside the 20 amino acids: {bad}')
        g[s:s + L] = [aa_index[c] for c in seq]
    return g


def baseline_methods(L_by_pid, pids_needed, keys, tokens_dir: Path, *, verify: bool = True) -> list[dict]:
    """Method records of the requested baselines (all but `random`, built later) in the
    flat layout. A missing cache is an error; with `verify` the cache's provenance sidecar
    must be current (rmsf_common.check_cache_provenance)."""
    from ensembits.baselines.mini3di_baselines import vote_from_frame_tokens, mini3di_centroids
    total = sum(L for _, L in L_by_pid.values())
    methods = []
    for key in [k for k in BASELINE_KEYS if k in keys and k in BASELINE_FILES]:
        fname, frames, uses_seq, cb = BASELINE_FILES[key]
        path = Path(tokens_dir) / fname
        # FIXED(anova_audit code fix, 2026-10-18): a missing cache is an error (was SKIP,
        #   which silently changed the residue set); its sidecar is checked.
        if not path.exists():
            raise SystemExit(f'baseline cache {path} missing (builders: README.md)')
        cb_path = Path(tokens_dir) / cb if cb else None
        if verify:
            try:
                rec = check_cache_provenance(path, DATASET, frames=frames, uses_sequence=uses_seq,
                                             codebook=cb_path, required=list(pids_needed))
            except CacheError as e:
                raise SystemExit(f'baseline {key}: {e}')
        else:
            rec = {'file': path.name, 'sha256': file_sha256(path), 'verified': False}
        z = np.load(path, allow_pickle=False)
        m = dict(key=key, name=BASELINE_NAMES[key], cache=rec, frames_read=frames)
        if key == 'protprofile_K10':
            # FIXED(anova_audit code fix, 2026-10-18): the Foldseek / MMseqs2 profile of the
            #   first 10 frames (was a 20-bin histogram of the mini3di tokens).
            X, valid, missing = _flat_from_npz(z, key, L_by_pid, pids_needed, total, (20,))
            m.update(kind='continuous', X=X, valid=valid)
        elif key == 'vote_3di':
            # FIXED(anova_audit code fix, 2026-10-18): the paper rule (plurality with count
            #   >= 2, else the centroid nearest to the frames' mean centroid) via
            #   mini3di_baselines.vote_from_frame_tokens (was a plain argmax of the counts).
            K10, valid, missing = _flat_from_npz(z, key, L_by_pid, pids_needed, total, (10,))
            cent = mini3di_centroids()
            g = np.full(total, -1, dtype=np.int64)
            for pid in pids_needed:
                s, L = L_by_pid[pid]
                if valid[s]:
                    tk = check_token_ids(K10[s:s + L].astype(np.int64), 20, f'vote_3di {pid}')
                    g[s:s + L] = vote_from_frame_tokens(tk, cent)
            m.update(kind='categorical', groups=g, valid=valid)
        else:
            g, valid, missing = _flat_from_npz(z, key, L_by_pid, pids_needed, total)
            n_codes = 20 if key == '3di' else int(np.load(cb_path, mmap_mode='r').shape[0])
            check_token_ids(g[valid], n_codes, key)
            m.update(kind='categorical', groups=g, valid=valid, n_codes=n_codes)
        m['missing_pids'] = missing
        methods.append(m)

    # amino acid: always read (it is part of the confound model); a row only if requested.
    g = amino_acids(L_by_pid, pids_needed)
    methods.append(dict(key='aa', name=BASELINE_NAMES['aa'], kind='categorical', groups=g,
                        valid=g >= 0, missing_pids=[], frames_read='sequence', row='aa' in keys,
                        cache={'dir': str(MDCATH_REAL_BB)}))
    return methods


# ---------------------------------------------------------------- analysis

def analyse(methods, targets: dict, prot, folds, topo_p, args, set_info) -> dict:
    """All statistics on the residue set (arrays already restricted to it).

    folds: fold of every residue (CATH-topology groups, size-balanced); topo_p: CATH topology
    of every protein (bootstrap clusters).
    """
    NP = int(prot.max()) + 1
    blocks = S.protein_blocks(prot)
    if len(np.unique(folds)) < 2:
        raise SystemExit(f'the residue set has {len(np.unique(topo_p))} CATH topology(ies): '
                         f'cannot make held-out folds (choose proteins of several topologies)')
    # FIXED(anova_audit fix round 2, 2026-10-18): topology-cluster bootstrap (was proteins).
    W = S.bootstrap_weights(NP, args.n_boot, args.boot_seed, clusters=topo_p)
    res = {'folds': {'k': args.n_folds, 'grouping': 'CATH topology',
                     'assignment': 'size-balanced greedy (anova_stats.balanced_group_folds)',
                     'n_residues_per_fold': np.bincount(folds).tolist()},
           'bootstrap': {'B': args.n_boot, 'seed': args.boot_seed, 'unit': 'CATH topology',
                         'n_clusters': int(len(np.unique(topo_p)))},
           'nulls': {'n': args.n_perm, 'circular_min_shift': args.circ_min_shift,
                     'n_proteins_unshifted': int((blocks[1] < 2 * args.circ_min_shift + 1).sum())},
           'regimes': {name: dict(ensembits_p_frames=P, baselines=keys)
                       for P, (name, keys) in REGIMES.items()},
           'targets': {}}
    nulls_out = {}
    present = {m['key'] for m in methods}
    conf = set_info['confounds']
    for tn, y in targets.items():
        t0 = time.time()
        yc = S.center_by_protein(y, prot)
        T = {'anova': {}, 'heldout_r2': {}, 'delta_vs_best_baseline_by_regime': {}}
        ss = {}
        for m in methods:
            row = {}
            if m['kind'] == 'categorical':
                g = m['groups']
                a = S.oneway_anova(y, g)
                row.update({k: a[k] for k in ('N', 'M', 'eta2', 'omega2', 'F', 'df_between',
                                              'df_within', 'n_singleton_groups',
                                              'p_parametric_invalid_under_dependence')})
                if args.min_count:
                    gp, npool, ntok = pool_rare_tokens(g, args.min_count)
                    ap_ = S.oneway_anova(y, gp)
                    row['diagnostic_pooled'] = dict(min_count=args.min_count, M=ap_['M'],
                                                    eta2=ap_['eta2'], omega2=ap_['omega2'],
                                                    n_pooled_residues=npool, n_pooled_tokens=ntok)
                yhat = S.heldout_token_means(yc, g, folds)
                kw = dict(g=g)
            else:
                # Authors' decision (fix round 2): ProtProfileMD keeps a linear fit on its 20-D
                # profile; adjusted R^2 is the counterpart of omega^2.
                X = m['X']
                r2 = S.ols_r2_no_intercept(X, y)
                n, p = X.shape
                row.update(N=n, n_features=p, statistic='ols_r2_no_intercept (profile sums to 1)',
                           r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / (n - p))
                yhat = S.heldout_ridge(X, yc, folds, alpha=args.ridge_alpha)
                kw = dict(X=X)
            obs = row['eta2'] if m['kind'] == 'categorical' else row['r2']
            if args.n_perm > 0:
                for kind in ('circular', 'shuffle'):
                    null = S.permutation_null(y, blocks=blocks, n=args.n_perm, seed=args.seed,
                                              kind=kind, min_shift=args.circ_min_shift, **kw)
                    row[f'perm_{kind}'] = S.perm_summary(null, obs)
                    nulls_out[f'{tn}|{m["key"]}|{kind}'] = null.astype(np.float32)
                row['perm_shuffle']['role'] = 'reference only (ignores within-protein dependence)'
            T['anova'][m['key']] = row
            ss[m['key']] = S.per_protein_ss(yc, yhat, prot, NP)
            r2_b = S.bootstrap_r2(W, *ss[m['key']])
            T['heldout_r2'][m['key']] = dict(r2=S.within_r2(yc, yhat, prot), ci95=S.ci95(r2_b),
                                             r2_uncentred_prediction=float(1 - ((yc - yhat) ** 2).sum()
                                                                           / (yc ** 2).sum()))
        # Delta = Ensembits row - best baseline OF ITS FRAME REGIME (authors' decision, fix
        # round 2): P = 1 vs the single-frame tokenizers, P = 10 vs the 10-frame baselines;
        # best on the point estimate, per target; paired topology-cluster bootstrap.
        for m in methods:
            if not m.get('is_ensembits'):
                continue
            regime, keys = REGIMES[m['p_frames']]
            keys = [k for k in keys if k in present]
            if not keys:
                continue
            best = max(keys, key=lambda k: T['heldout_r2'][k]['r2'])
            d_b = S.bootstrap_r2(W, *ss[m['key']]) - S.bootstrap_r2(W, *ss[best])
            T['delta_vs_best_baseline_by_regime'][m['key']] = dict(
                regime=regime, best_baseline=best,
                delta=T['heldout_r2'][m['key']]['r2'] - T['heldout_r2'][best]['r2'], ci95=S.ci95(d_b))
        # Controls (same residues): eta^2 of the target by each grouping variable
        ctl = {}
        for name, g in set_info['controls'].items():
            a = S.oneway_anova(y, g)
            ctl[name] = dict(M=a['M'], eta2=a['eta2'], omega2=a['omega2'])
        T['controls'] = ctl
        # Gain over confounds (chain-end bin x burial decile x SS proxy cell, + amino acid):
        # joint backfit = primary; residual fit = sensitivity field (JSON only).
        mets = {}
        for m in methods:
            if m['key'] == 'aa':
                continue                       # amino acid is part of the confound model
            mets[m['key']] = ('tokens', m['groups']) if m['kind'] == 'categorical' \
                else ('ridge', m['X'])
        for mode, key in (('joint', 'gain_over_confounds'),
                          ('residual', 'gain_over_confounds_residual_sensitivity')):
            y0, yh = S.heldout_gain_over_confounds(yc, [conf['cell'], conf['aa']], folds, mets,
                                                   ridge_alpha=args.ridge_alpha, mode=mode)
            b0 = S.bootstrap_r2(W, *S.per_protein_ss(yc, y0, prot, NP))
            r0 = S.within_r2(yc, y0, prot)
            G = {'mode': mode, 'confounds_only': dict(r2=r0, ci95=S.ci95(b0)), 'methods': {}}
            for k, yhk in yh.items():
                bk = S.bootstrap_r2(W, *S.per_protein_ss(yc, yhk, prot, NP))
                rk = S.within_r2(yc, yhk, prot)
                G['methods'][k] = dict(r2_confounds_plus_method=rk, ci95=S.ci95(bk),
                                       gain=rk - r0, gain_ci95=S.ci95(bk - b0))
            T[key] = G
        T['seconds'] = round(time.time() - t0, 1)
        res['targets'][tn] = T
        print(f'  [{tn}] done in {T["seconds"]:.0f}s', flush=True)
    return res, nulls_out


# ---------------------------------------------------------------- report

def _fmt_ci(d, key='r2', ci='ci95'):
    return f"{d[key]:.3f} [{d[ci][0]:.3f}, {d[ci][1]:.3f}]"


def summary_markdown(out: dict) -> str:
    names = {m['key']: m['name'] for m in out['methods'] if m.get('row', True)}
    regime_of = {}
    for m in out['methods']:
        if m.get('is_ensembits'):
            regime_of[m['key']] = REGIMES[m['p_frames']][0]
    for P, (rn, keys) in REGIMES.items():
        for k in keys:
            regime_of[k] = rn
    A = out['analysis']
    tg = A['targets']
    tl = list(tg)
    t1 = tl[0]
    rs = out['residue_set']
    k = A['folds']['k']
    L = [f"# Token ANOVA ({out['created']})", '',
         f"Residue set: {rs['n_residues']:,} residues, {rs['n_proteins']:,} proteins (s_1 "
         f"placeholders masked: {rs['n_s1_placeholder_masked']}). ANOVA columns are on {t1}.", '',
         '| method | regime | M | eta2 | omega2 | F (df) | R2 (profile, OLS) | adj. R2 (profile) | '
         f'circular-shift p (null mean) | ' + ' | '.join(f'held-out R2 {t} [95% CI]' for t in tl)
         + ' | ' + ' | '.join(f'gain over confounds {t} [95% CI]' for t in tl) + ' |',
         '|' + '---|' * (9 + 2 * len(tl))]
    for key, nm in names.items():
        a = tg[t1]['anova'][key]
        if 'eta2' in a:
            stat = (f"{a['M']:,} | {a['eta2']:.3f} | {a['omega2']:.3f} | "
                    f"{a['F']:.1f} ({a['df_between']:,}, {a['df_within']:,}) | - | -")
        else:
            stat = f"{a['n_features']}-D | - | - | - | {a['r2']:.3f} | {a['adj_r2']:.3f}"
        pc = a.get('perm_circular')
        ptxt = f"{pc['text']} ({pc['null_mean']:.4f})" if pc else '-'
        h = ' | '.join(_fmt_ci(tg[t]['heldout_r2'][key]) for t in tl)
        gn = []
        for t in tl:
            g = tg[t]['gain_over_confounds']['methods'].get(key)
            gn.append(_fmt_ci(g, 'gain', 'gain_ci95') if g else '- (in the confounds)')
        L.append(f"| {nm} | {regime_of.get(key, '-')} | {stat} | {ptxt} | {h} | {' | '.join(gn)} |")
    L += ['', '| Ensembits row | regime | target | best baseline of the regime | Delta held-out R2 [95% CI] |',
          '|---|---|---|---|---|']
    for t in tl:
        for key, d in tg[t]['delta_vs_best_baseline_by_regime'].items():
            L.append(f"| {names[key]} | {d['regime']} | {t} | {names[d['best_baseline']]} | "
                     f"{_fmt_ci(d, 'delta')} |")
    L += ['', '| control (same residues) | M | ' + ' | '.join(f'eta2 {t}' for t in tl) + ' |',
          '|---|---|' + '---|' * len(tl)]
    for c in tg[t1]['controls']:
        L.append(f"| {c} | {tg[t1]['controls'][c]['M']:,} | "
                 + ' | '.join(f"{tg[t]['controls'][c]['eta2']:.3f}" for t in tl) + ' |')
    three = rs.get('three_di_chain_end_state2') or {}
    L += ['', 'Confounds only (held-out within-protein R2): '
          + '; '.join(f"{t} {_fmt_ci(tg[t]['gain_over_confounds']['confounds_only'])}" for t in tl), '',
          'Notes:',
          f"* Held-out R2: target centred per protein; {k} folds of proteins grouped by CATH topology "
          f"(size-balanced; residues per fold {A['folds']['n_residues_per_fold']}); token means of the "
          "training folds (unseen token -> 0); the prediction is also centred per protein (no label "
          "used; `r2_uncentred_prediction` in the JSON without it). ProtProfileMD: ridge on the 20-D "
          "profile (alpha " + f"{out['settings']['ridge_alpha']})" + ".",
          f"* 95% CIs: bootstrap over {A['bootstrap']['n_clusters']:,} CATH topologies "
          f"(B = {A['bootstrap']['B']:,}; a drawn topology brings all its proteins); Delta paired.",
          f"* Circular-shift null: each protein's token sequence rotated by {A['nulls']['circular_min_shift']} "
          f"to L - {A['nulls']['circular_min_shift']} residues (n = {A['nulls']['n']:,}); "
          "p = (b + 1)/(n + 1), shown as p < 10^-k (0/n) when no draw reaches the observed value. "
          "The plain-shuffle null and the parametric F p are in the JSON only.",
          "* Delta: Ensembits P = 1 vs the best single-frame tokenizer (3Di, AminoAseed, ESM3struct, "
          "ProToken), P = 10 vs the best 10-frame baseline (Vote_3Di, ProtProfileMD_K10); amino acid "
          "is shown but is not a tokenizer baseline.",
          "* chance rows: the Ensembits row's own tokens permuted over the residue set (same M and "
          "token frequencies, seeded).",
          "* Gain over confounds: held-out R2 of confounds (chain-end distance bin x burial decile x "
          "C-alpha SS proxy, + amino acid) + the method, backfitted jointly, minus confounds alone. "
          "A residual-fit sensitivity version is in the JSON.",
          f"* 3Di (frame 0): the first and last residue of every chain carry Foldseek's invalid state 2 "
          f"({three.get('n_state2_at_chain_ends', '?'):,} of {three.get('n_chain_end_residues', '?'):,} "
          f"chain-end residues of the set).",
          "* Vote_3Di: plurality (count >= 2), else the centroid nearest to the frames' mean centroid, "
          "lowest letter id on exact ties."]
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS,
                    help='Ensembits model dirs, or "lambda_sweep" (e0, lam0, lam0p05, lam0p25, '
                         'lam0p5, lam1); default output/ensembits_bm_lam10_rv')
    ap.add_argument('--out', required=True, help='output JSON')
    ap.add_argument('--min-count', type=int, default=None,
                    help='DIAGNOSTIC: add a pooled-rare-token block (main numbers stay raw)')
    ap.add_argument('--device', default=None, help='torch device (default cuda:0 if available)')
    ap.add_argument('--allow-ideal', action='store_true')
    ap.add_argument('--baselines', nargs='*', default=list(BASELINE_KEYS), choices=BASELINE_KEYS,
                    help='baseline rows (the amino acid is always read for the confound model; '
                         '"random" adds a chance row per Ensembits row)')
    ap.add_argument('--feature', choices=('s1', 's1_over_s2'), default='s1',
                    help='first target: s1, or s1/(s2+1e-3) (motion-shape row)')
    ap.add_argument('--no-rmsf', action='store_true', help='skip the all-frames RMSF target')
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--circ-min-shift', type=int, default=5)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--boot-seed', type=int, default=1)
    ap.add_argument('--ridge-alpha', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0, help='nulls and the chance rows')
    ap.add_argument('--s1-source', choices=('compute', 'cache'), default='compute')
    ap.add_argument('--s1-cache', default=str(S1_CACHE))
    ap.add_argument('--allow-unverified-s1-cache', action='store_true')
    ap.add_argument('--rebuild-residue-table', action='store_true')
    ap.add_argument('--residue-table-dir', default=str(RESIDUE_TABLE_DIR))
    ap.add_argument('--workers', type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument('--descriptor-source', choices=('cache', 'compute'), default='cache')
    ap.add_argument('--tokens-dir', default=str(TOKENS_DIR), help='baseline caches')
    ap.add_argument('--allow-unverified-baselines', action='store_true',
                    help='baseline caches without a current sidecar (recorded; not for the paper)')
    ap.add_argument('--token-cache-dir', default=str(TOKEN_CACHE_DIR))
    ap.add_argument('--no-token-cache', action='store_true')
    ap.add_argument('--allow-legacy-cache', action='store_true')
    ap.add_argument('--pids', nargs='*', default=None)
    ap.add_argument('--max-proteins', type=int, default=0)
    ap.add_argument('--fig-dir', default=str(FIG_DIR))
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--export-token-cache', default=None,
                    help='npz path in the research cache format; must contain {variant} when '
                         'several variants are run')
    args = ap.parse_args(argv)
    v = []
    for x in args.variants:
        v += VARIANT_ALIASES.get(x, [x])
    args.variants = v
    if args.device is None:
        import torch
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


CODE_FILES = ('downstream/analyses/token_anova.py', 'downstream/analyses/anova_stats.py',
              'downstream/analyses/plot_anova_test.py', 'downstream/analyses/_common.py',
              'downstream/rmsf_common.py', 'ensembits/baselines/mini3di_baselines.py')


def code_record() -> dict:
    """sha256 of the code files used and the git HEAD / dirty state of the repository."""
    import subprocess
    rec = {'files': {f: file_sha256(CODE_ROOT / f) for f in CODE_FILES}}
    try:
        rec['git_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, capture_output=True,
                                         text=True, timeout=30).stdout.strip() or None
        st = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=CODE_ROOT,
                            capture_output=True, text=True, timeout=60).stdout
        rec['git_code_dir_untracked_or_dirty'] = bool(st.strip())
    except Exception as e:                                   # no git: hashes are enough
        rec['git_error'] = type(e).__name__
    return rec


def checkpoint_lambda(cfg: dict):
    """lambda of the SFTD consistency loss = `consistency_mse_weight` (config top level or the
    training `args` block, where train.py stores it); None when the config has neither."""
    if cfg.get('consistency_mse_weight') is not None:
        return cfg['consistency_mse_weight']
    return (cfg.get('args') or {}).get('consistency_mse_weight')


def main(argv=None):
    args = parse_args(argv)
    T0 = time.time()
    out_path = Path(args.out)
    vdirs = [resolve_model_dir(v) for v in args.variants]
    for vdir in vdirs:
        if not (vdir / 'config.json').exists():
            raise SystemExit(f'no checkpoint config in {vdir}')
    L_by_pid = manifest_layout()
    man = manifest()
    total = sum(L for _, L in L_by_pid.values())
    if args.pids:
        # FIXED(anova_audit fix round 2, 2026-10-18): unknown pids are an error (was: ignored)
        unknown = [p for p in args.pids if p not in L_by_pid]
        if unknown:
            raise SystemExit(f'--pids: {len(unknown)} pids are not mdCATH-div representatives: {unknown[:5]}')
        pids_needed = list(dict.fromkeys(args.pids))
    elif args.max_proteins:
        pids_needed = list(L_by_pid)[:args.max_proteins]
    else:
        pids_needed = list(L_by_pid)
    rows_keep = np.zeros(total, dtype=bool)
    for p in pids_needed:
        s, L = L_by_pid[p]; rows_keep[s:s + L] = True
    prot_all = np.concatenate([np.full(L, i) for i, (_, L) in enumerate(L_by_pid.values())])
    res_all = np.concatenate([np.arange(L) for _, L in L_by_pid.values()])
    len_all = np.concatenate([np.full(L, L) for _, L in L_by_pid.values()])
    pid_names = np.array(list(L_by_pid))

    # 1. residue table (s_1, s_2, burial, SS proxy) -- always the full table (cached)
    table, table_rec = residue_table(L_by_pid, args.workers, Path(args.residue_table_dir),
                                     args.rebuild_residue_table)
    feats = table['feats']
    s1_rec = {'source': 'compute (ca_10)', **table_rec}
    if args.s1_source == 'cache':
        feats, s1_ok, s1_rec = load_s1_research_cache(Path(args.s1_cache), L_by_pid, total,
                                                      args.allow_unverified_s1_cache)
        s1_rec['source'] = 'research cache'
        if not s1_ok[rows_keep].all():
            raise SystemExit(f'{args.s1_cache}: {int((rows_keep & ~s1_ok).sum())} residues of the '
                             f'requested proteins have no s_1')
    ball = table['cn10'].astype(np.int64) + 1
    placeholder = ball < 4                       # s_1 = 0 placeholder of per_residue_pca_features
    y_s1 = feats[:, 0].astype(np.float64) if args.feature == 's1' else \
        (feats[:, 0] / (feats[:, 1] + 1e-3)).astype(np.float64)
    tname = 's1' if args.feature == 's1' else 's1_over_s2'

    # 2. RMSF target
    targets_full = {tname: y_s1}
    rmsf_rec = None
    if not args.no_rmsf:
        rmsf_raw = np.load(RMSF_ALLFRAMES)
        try:
            lab_sha = check_label_hash(RMSF_ALLFRAMES, rmsf_raw)      # hash of the stored dtype
            check_flat_label_sidecar(RMSF_ALLFRAMES, DATASET, 'allframes', len(rmsf_raw))
        except (CacheError, ValueError) as e:
            raise SystemExit(f'RMSF label: {e}')
        # FIXED(anova_audit fix round 2, 2026-10-18): the label rows must be in manifest order
        side = read_provenance(RMSF_ALLFRAMES) or {}
        if side.get('row_order') != '_manifest.json representatives' \
                or int(side.get('n_pids', -1)) != len(L_by_pid):
            raise SystemExit(f'RMSF label: sidecar row_order {side.get("row_order")!r} / n_pids '
                             f'{side.get("n_pids")!r}, expected "_manifest.json representatives" / '
                             f'{len(L_by_pid)}')
        rmsf = rmsf_raw.astype(np.float64)
        if len(rmsf) != total:
            raise SystemExit(f'{RMSF_ALLFRAMES.name} has {len(rmsf)} rows, the layout {total}')
        targets_full['rmsf_allframes'] = rmsf
        rmsf_rec = {'file': str(RMSF_ALLFRAMES), 'label': 'allframes', 'labels_sha256': lab_sha,
                    'row_order': side['row_order']}

    # 3. Ensembits tokens (P = 10 and P = 1 per variant); errors exit non-zero
    methods = []
    variant_names, lambdas = [], {}
    for vdir in vdirs:
        variant_names.append(vdir.name)
        lam = checkpoint_lambda(json.loads((vdir / 'config.json').read_text()))
        lambdas[vdir.name] = lam
        for P in P_FRAMES:
            t0 = time.time()
            print(f'[anova] tokenizing {vdir.name} at P = {P} ...', flush=True)
            tok, prov = load_variant_tokens(vdir, L_by_pid, pids_needed, args, p_frames=P)
            prov['consistency_mse_weight'] = lam
            print(f'  {int((tok[rows_keep] >= 0).sum()):,} residues, '
                  f'{len(np.unique(tok[tok >= 0]))} codes ({time.time() - t0:.0f}s)', flush=True)
            methods.append(dict(key=f'ens:{vdir.name}:P{P}', variant=vdir.name, p_frames=P,
                                name=f'Ensembits P={P}{" (frame 0)" if P == 1 else ""} [{vdir.name}]',
                                kind='categorical', groups=tok, valid=tok >= 0, is_ensembits=True,
                                prov=prov, frames_read=prov['frames_read'], lam=lam))

    # 4. baselines (computed once, shared by all variants); amino acid always (confounds)
    methods += baseline_methods(L_by_pid, pids_needed, set(args.baselines), Path(args.tokens_dir),
                                verify=not args.allow_unverified_baselines)

    # 5. the residue set: requested proteins minus the s_1 placeholders. Every other gap is an
    #    error (FIXED anova_audit fix round 2, 2026-10-18: was a silent intersection).
    n_placeholder = int((rows_keep & placeholder).sum())
    in_set = rows_keep & ~placeholder
    gaps = {}
    for name, y in targets_full.items():
        gaps[f'target {name}'] = int((in_set & ~np.isfinite(y)).sum())
    for m in methods:
        gaps[m['key']] = int((in_set & ~m['valid']).sum())
    gaps = {k: v for k, v in gaps.items() if v}
    if gaps:
        raise SystemExit(f'residues of the requested proteins without a value: {gaps}')
    idx = np.where(in_set)[0]
    # B.15: assert no s_1 = 0 placeholder survives into the Delta candidates
    s1_in_set = feats[idx, 0]
    n_zero = int((s1_in_set == 0).sum())
    assert n_zero == 0, (f'{n_zero} residues in the set have s_1 = 0 '
                         f'(placeholders should have been masked)')
    _, prot = np.unique(prot_all[idx], return_inverse=True)
    prot = prot.ravel()
    print(f'[anova] residue set: {len(idx):,} residues, {prot.max() + 1} proteins '
          f'(masked s_1 placeholders: {n_placeholder})', flush=True)
    for m in methods:
        if m['kind'] == 'categorical':
            m['groups'] = m['groups'][idx]
        else:
            m['X'] = m['X'][idx]
    aa_g = next(m for m in methods if m['key'] == 'aa')['groups']
    rows = [m for m in methods if m.get('row', True)]
    if 'random' in args.baselines:
        # FIXED(anova_audit fix round 2, 2026-10-18): one chance row per Ensembits row (its own
        #   tokens permuted: same M and frequencies), was one shuffle of the first variant.
        for i, e in enumerate([m for m in rows if m.get('is_ensembits')]):
            g = np.random.default_rng(args.seed + 1000 + i).permutation(e['groups'])
            rows.append(dict(key=f'chance:{e["variant"]}:P{e["p_frames"]}',
                             name=f'chance (permuted {e["name"]})', kind='categorical', groups=g,
                             frames_read='-', matched_to=e['key'], is_chance=True))
    three = next((m for m in rows if m['key'] == '3di'), None)
    r_i, L_i = res_all[idx], len_all[idx]
    chain_end = (r_i == 0) | (r_i == L_i - 1)
    upid = np.unique(prot_all[idx])
    topo_p = np.array([man[pid_names[i]]['topology'] for i in upid])
    class_p = np.array([t.split('.')[0] for t in topo_p])
    len_p = np.array([L_by_pid[pid_names[i]][1] for i in upid])
    nres_p = np.bincount(prot)
    fold_p = S.balanced_group_folds(topo_p, nres_p, k=args.n_folds)
    folds = fold_p[prot]
    qcuts = np.quantile(len_p, [0.2, 0.4, 0.6, 0.8])
    ed = S.enddist_bins(r_i, L_i)
    set_info = {
        'controls': {
            'CATH class': np.unique(class_p, return_inverse=True)[1].ravel()[prot],
            'protein-length quintile': np.searchsorted(qcuts, len_p)[prot],
            'residue-position quintile': np.clip((r_i / np.maximum(L_i - 1, 1) * 5).astype(int), 0, 4),
            'chain-end distance bin': ed,
            'protein identity': prot,
        },
        'confounds': {'cell': S.confound_cells(ed, S.burial_deciles(table['cn10'][idx]),
                                               table['ss'][idx]),
                      'aa': aa_g},
    }
    residue_set = {
        'n_residues': int(len(idx)), 'n_proteins': int(prot.max() + 1),
        'n_rows_layout': total, 'n_rows_requested': int(rows_keep.sum()),
        'n_s1_placeholder_masked': n_placeholder,
        'placeholder_rule': 'fewer than 4 C-alpha within 10 A in frame 0 (s_1 = 0)',
        'three_di_chain_end_state2': (None if three is None else dict(
            n_chain_end_residues=int(chain_end.sum()),
            n_state2_at_chain_ends=int((three['groups'][chain_end] == MINI3DI_INVALID_STATE).sum()),
            note='mini3di masks the first and last residue; they take Foldseek\'s invalid state 2')),
        'cath_topology_source': 'data/external/mdcath_div/_manifest.json representatives[pid].topology',
        'cath_class_source': 'first field of the CATH topology',
    }

    targets = {k: v[idx] for k, v in targets_full.items()}
    analysis, nulls = analyse(rows, targets, prot, folds, topo_p, args, set_info)

    meth_rec = []
    for m in methods + [r for r in rows if r.get('is_chance')]:
        r = {k: m[k] for k in ('key', 'name', 'kind', 'frames_read') if k in m}
        for k in ('variant', 'p_frames', 'prov', 'cache', 'matched_to', 'is_ensembits', 'is_chance',
                  'row'):
            if k in m:
                r[k] = m[k]
        if m.get('is_ensembits'):
            r['consistency_mse_weight'] = m['lam']
        meth_rec.append(r)
    code = code_record()
    out = {'schema': SCHEMA, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'variants': variant_names, 'consistency_mse_weight': lambdas, 'feature': tname,
           'settings': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           'residue_set': residue_set, 'methods': meth_rec, 'analysis': analysis,
           'inputs': {'s1': s1_rec, 'rmsf': rmsf_rec,
                      'manifest_sha256': file_sha256(MDCATH_DIV_SRC / '_manifest.json')},
           'code': code,
           'notes': {
               'p_parametric': 'F-test p kept only as p_parametric_invalid_under_dependence '
                               '(residues of a protein are not independent)',
               'perm_shuffle': 'plain global shuffle: reference only (ignores within-protein dependence)',
               'heldout_r2': 'target centred per protein; prediction centred per protein (label-free); '
                             'r2_uncentred_prediction without that step',
               'bootstrap': 'CATH-topology cluster bootstrap of the fixed cross-fitted predictions',
               'protprofile': 'continuous 20-D Foldseek/MMseqs2 profile: OLS R2 (no intercept, rows sum '
                              'to 1) in-sample, adjusted R2 as the omega2 counterpart; ridge '
                              '(alpha = --ridge-alpha) for held-out R2 (authors: linear fit kept)',
               'delta': 'per frame regime: P=1 vs single-frame tokenizers, P=10 vs 10-frame baselines',
               'chance': 'each Ensembits row permuted over the residue set (same M and frequencies)',
               'gain': 'gain_over_confounds = joint backfit (primary); '
                       'gain_over_confounds_residual_sensitivity = residual fit (sensitivity)'},
           'runtime_s': None}
    stem = out_path.with_suffix('')
    arrays = dict(pids=pid_names[prot_all[idx]].astype(str), ress=r_i.astype(np.int32),
                  fold=folds.astype(np.int8), protein_pid=pid_names[upid].astype(str),
                  protein_topology=topo_p.astype(str), protein_fold=fold_p.astype(np.int8),
                  **{f'target__{k}': v.astype(np.float32) for k, v in targets.items()})
    for m in rows:
        if m['kind'] == 'categorical':
            arrays[f'tokens__{m["key"]}'] = m['groups'].astype(np.int64)     # raw ids
    _atomic_savez(Path(str(stem) + '.residues.npz'), **arrays)
    _atomic_savez(Path(str(stem) + '.nulls.npz'), **nulls)
    if args.export_token_cache:
        if len(variant_names) > 1 and '{variant}' not in args.export_token_cache:
            raise SystemExit('--export-token-cache needs {variant} with several variants')
        for vn in variant_names:
            exp = Path(args.export_token_cache.format(variant=vn))
            g10 = next(m for m in methods if m.get('variant') == vn and m['p_frames'] == 10)
            g1 = next(m for m in methods if m.get('variant') == vn and m['p_frames'] == 1)
            _atomic_savez(exp, compressed=False, tokens=g10['groups'], tokens_P1=g1['groups'],
                          feats=feats[idx], pids=arrays['pids'].astype(object), ress=r_i,
                          variant=vn, ckpt_sha256_12=g10['prov']['ckpt_sha256_12'],
                          dihedral_convention=g10['prov']['dihedral_convention'])
            print(f'  exported raw tokens of {vn} -> {exp}')
    out['runtime_s'] = round(time.time() - T0, 1)
    _atomic_write_text(out_path, json.dumps(out, indent=1, default=float))
    write_provenance(out_path, {'schema': SCHEMA, 'builder': 'downstream/analyses/token_anova.py',
                                'sha256': file_sha256(out_path), 'variants': variant_names,
                                'consistency_mse_weight': lambdas, 'code': code,
                                'checkpoints': {m['key']: m['prov']['ckpt_sha256_12']
                                                for m in methods if m.get('is_ensembits')},
                                'inputs': out['inputs'],
                                'baseline_caches': {m['key']: m.get('cache') for m in methods
                                                    if not m.get('is_ensembits')}})
    _atomic_write_text(Path(str(stem) + '.table.md'), summary_markdown(out))
    if not args.no_figures:
        from plot_anova_test import figures_from_outputs
        for f in figures_from_outputs(out_path, Path(args.fig_dir)):
            print(f'  figure -> {f}')
    print(f'\nwrote {out_path} ({out["runtime_s"]:.0f}s)')
    print(Path(str(stem) + '.table.md').read_text())


if __name__ == '__main__':
    main()
