"""Is a ProToken token about its own residue? RMSF locality diagnostic on MISATO (validation only).

What it does
------------
ProToken scores near zero in the per-residue RMSF probe although its tokens equal the
official ProToken-1.0 output and decode back to the structure (fidelity audit,
2026-10-18). This script measures where the RMSF information of a ProToken token lives:

  1. Within-protein eta^2 of the RMSF rank on the token id, for ProToken, AminoAseed,
     ESM3struct, mini3di (3Di, frame 0) and the amino acid: per complex the all-frames
     RMSF is turned into its centred rank in [0, 1]; eta^2 = SS_between / SS_total of
     that rank grouped by token id, pooled over the train complexes. A permuted-token
     control is reported next to each value.
  2. The same eta^2 for ProToken with token j paired with the RMSF of residue j + k,
     k in --offsets (the residue order of every array is the misato_real_bb order).
  3. A ridge regression (sklearn Ridge, alpha 10) from the frozen codebook vectors of
     the tokens j - w ... j + w (zero padded at the complex ends) to the RMSF rank of
     residue j, fitted on the train complexes and scored on the validation complexes
     as the mean per-protein Spearman (complexes with >= 10 residues and a
     non-constant label), for ProToken and AminoAseed and every w in --windows.

No test complex is read. Train complexes: the first --n-train complexes of the split's
train list (file order) that have an RMSF label (default 1,500; the full train set does
not fit the AminoAseed windowed design matrix in memory); validation: all of them.

Paper
-----
`appendix.tex` \\label{app:baselines}, ProToken item ("its tokens are not residue
local ..."), and the ProToken remark in \\label{app:token_anova}.

Origin
------
New (2026-10-18), from the ad hoc checks run when the authors asked whether ProToken
was really that weak. Same computation, made a script with a JSON output.

Used by the reported model: no (diagnostic of a baseline).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/misato_rmsf_allframes.npz
          data/cached_descriptors/{protoken,aminoaseed,esm3struct,mini3di}_tokens_misato.npz
          data/cached_descriptors/{protoken,aminoaseed}_codebook.npy
          data/cached_descriptors/misato_real_bb/<PID>.npz   (sequence)
          data/misato_splits_v2.json (ensembits.paths.MISATO_SPLITS, D3 fix)
                                                              (--split, default structure)
  writes  output/analyses/protoken_locality_misato_<split>.json

How to run
----------
    python downstream/analyses/protoken_locality.py [--split structure] [--n-train 1500] \\
        [--offsets -3,-2,-1,0,1,2,3,4,5,6,8,10] [--windows 0,3,6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# NOTE: REPO_ROOT below uses parents[4] which is STALE — it was correct in the
# submission_exp layout but is wrong for the camera-ready tree. The split file
# is loaded via ensembits.paths.MISATO_SPLITS (data/misato_splits_v2.json, D3 fix)
# at runtime (line ~117); this fallback is only used for the default output path
# and the cached-descriptors directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
C = REPO_ROOT / 'data' / 'cached_descriptors'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def centred_rank(y):
    r = np.argsort(np.argsort(y)) / max(len(y) - 1, 1)
    return r - r.mean()


def eta2(t, y):
    tot = ((y - y.mean()) ** 2).sum()
    b = 0.0
    for k in np.unique(t):
        m = t == k
        b += m.sum() * (y[m].mean() - y.mean()) ** 2
    return float(b / tot)


def paired(tokens, labels, pids, off):
    """Token j with the label of residue j + off (off may be negative)."""
    ts, ys = [], []
    for p in pids:
        x, y = tokens[p], labels[p]
        L = len(x)
        if L != len(y):
            raise SystemExit(f'[error] {p}: {len(x)} tokens vs {len(y)} labels')
        if off >= 0:
            ts.append(x[:L - off]); ys.append(y[off:])
        else:
            ts.append(x[-off:]); ys.append(y[:L + off])
    return np.concatenate(ts), np.concatenate(ys)


def window_feats(X, w):
    L = len(X)
    pad = np.zeros((w, X.shape[1]), np.float32)
    Xp = np.vstack([pad, X, pad])
    return np.hstack([Xp[w + o:w + o + L] for o in range(-w, w + 1)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='structure')
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--offsets', default='-3,-2,-1,0,1,2,3,4,5,6,8,10')
    ap.add_argument('--windows', default='0,3,6')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    R = np.load(C / 'misato_rmsf_allframes.npz')
    # FIXED(benchmark_audit D3, 2026-10-18): the v2 split file (ensembits.paths.MISATO_SPLITS).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # codebase root
    from ensembits.paths import MISATO_SPLITS
    from ensembits.data_versions import load_misato_splits
    sp = load_misato_splits(MISATO_SPLITS)[a.split]
    train = [p.upper() for p in sp['train'] if p.upper() in R.files][:a.n_train]
    val = [p.upper() for p in sp['val'] if p.upper() in R.files]
    raw = {p: R[p].astype(float) for p in train + val}
    rank = {p: centred_rank(raw[p]) for p in train}

    tok = {k: np.load(C / f'{k}_tokens_misato.npz')
           for k in ('protoken', 'aminoaseed', 'esm3struct', 'mini3di')}
    toks = {k: {p: d[p].astype(np.int64).ravel() for p in train + val} for k, d in tok.items()}
    toks['aa'] = {p: np.array([AA.find(c) for c in str(np.load(C / 'misato_real_bb' / f'{p}.npz')['seq'])])
                  for p in train}
    rng = np.random.default_rng(0)

    # FIXED(benchmark_audit D3, 2026-10-19): the output records the split file (path + sha256).
    from ensembits.data_versions import misato_splits_record
    out = {**misato_splits_record(MISATO_SPLITS), 'split': a.split, 'n_train_complexes': len(train), 'n_val_complexes': len(val),
           'eta2_within_protein': {}, 'protoken_offset_eta2': {}, 'ridge_val_per_protein_spearman': {}}
    for k in ('protoken', 'aminoaseed', 'esm3struct', 'mini3di', 'aa'):
        t, y = paired(toks[k], rank, train, 0)
        out['eta2_within_protein'][k] = {'eta2': round(eta2(t, y), 4),
                                         'permuted': round(eta2(rng.permutation(t), y), 4),
                                         'n_residues': int(len(t))}
        print(k, out['eta2_within_protein'][k], flush=True)
    for off in [int(s) for s in a.offsets.split(',')]:
        t, y = paired(toks['protoken'], rank, train, off)
        out['protoken_offset_eta2'][str(off)] = round(eta2(t, y), 4)
    print('protoken offsets', out['protoken_offset_eta2'], flush=True)

    for k in ('protoken', 'aminoaseed'):
        cb = np.load(C / f'{k}_codebook.npy').astype(np.float32)
        out['ridge_val_per_protein_spearman'][k] = {}
        for w in [int(s) for s in a.windows.split(',')]:
            Xtr = np.vstack([window_feats(cb[toks[k][p]], w) for p in train])
            ytr = np.concatenate([rank[p] for p in train])
            m = Ridge(alpha=10.0).fit(Xtr, ytr)
            del Xtr
            sc = []
            for p in val:
                y = raw[p]
                if len(y) < 10 or np.ptp(y) == 0:
                    continue
                sc.append(spearmanr(m.predict(window_feats(cb[toks[k][p]], w)), y).correlation)
            out['ridge_val_per_protein_spearman'][k][str(w)] = {'mean': round(float(np.nanmean(sc)), 4),
                                                                 'n_proteins': len(sc)}
            print(k, 'window', w, out['ridge_val_per_protein_spearman'][k][str(w)], flush=True)

    dst = Path(a.out) if a.out else REPO_ROOT / 'output' / 'analyses' / f'protoken_locality_misato_{a.split}.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.json.part')
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(dst)
    print('wrote', dst)
    return 0


if __name__ == '__main__':
    sys.exit(main())
