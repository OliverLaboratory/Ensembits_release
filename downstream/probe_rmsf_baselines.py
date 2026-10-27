"""RMSF probe for the baselines on mdCATH-div or BioEmu-CATH (per-residue MLP on fixed features).

What it does
------------
Builds one feature vector per residue for a baseline (amino-acid one-hot, ESM2
embeddings, mini3di, 3Di tokens, ProtProfile, Vote_3Di, AminoAseed, ProToken,
ESM3-struct codewords, random, ...), aligned with the RMSF label array of the
2,442 mdCATH-div representatives (308,293 residues, manifest order). Trains the
same MLP regressor as the Ensembits probe (`probe_rmsf_eval.py`) with the same
protocol: early stopping on the validation residues, then one evaluation on the
test residues.

Since 2026-10-16 (preretrain_audit_r2 Q2 / Q3 / Q29, user decisions): every method is
trained, validated and tested on the common residue set
(`data/cached_descriptors/rmsf_residue_set_<dataset>.json`, see
`data_prep/build_rmsf_residue_sets.py`); a domain of the set without a feature, or any
feature whose length differs from the label's, is an error (the loaders used to cut to
min(...) and skip absent domains); domains outside the set may be absent (on mdCATH-div
none since the re-extraction of 2026-10-17; before, the 8 domains without a real backbone). `--residue-set all` keeps the old per-feature coverage (legacy).
The JSON gains the per-protein Spearman (val / test), the residue-set record, `dataset`
and the RMSF label name (`--rmsf-label`, default allframes). The `aa` one-hot reads the
mdCATH real-backbone sequence with 'X' -> 'H' (the cache writes CHARMM HSD/HSE/HSP as X;
rmsf_common.normalize_mdcath_real_bb_seq). `--dataset bioemu` (phase 2) reads the BioEmu
caches `<name>_bioemu.npz` (keyed by system; built by data_prep/baselines with
--dataset bioemu), `aa` from bioemu_cath_extras, labels / splits from
output/phase_b/cached_desc/bioemu/, results in output/probe_rmsf_bioemu/.

Paper
-----
`2_results.tex` \label{result:rmsf}, Table tab:rmsf (baseline rows, mdCATH-div
columns); baselines in `appendix.tex` \label{app:baselines}.

Origin
------
Copied from `submission_exp/src/probe_rmsf_baselines.py`. Changed 2026-10-15
(code_fix C4 / §7): the fixed 40-epoch loop that reported the best validation
epoch was replaced by `probe_rmsf_eval.fit_probe_early_stopping` (early stop on
val, report test; output tag `__test`); `MDCATH_DIV` from `ensembits.paths`
(was another user's absolute path); `ROOT` from `ensembits.paths.REPO_ROOT`;
`--epochs` became `--max-epochs` / `--patience`. Feature construction unchanged.
2026-10-19: Vote_3Di = mini3di_baselines.vote_from_frame_tokens (the ANOVA's rule; result
records rmsf_common.VOTE_RULE).
2026-10-16: see above (FIXED markers: common residue set, strict lengths, His,
per-protein Spearman, labels, --dataset); `get_div_layout` moved to
rmsf_common.mdcath_div_layout.

Used by the reported model: no (baseline rows of the same table).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/full_train/cached_desc/splits.json,
          output/phase_b/cached_desc/rmsf_div_allframes.npy (--rmsf-label),
          data/cached_descriptors/rmsf_residue_set_mdcath_div.json,
          data/cached_descriptors/<feature caches>, data/cached_descriptors/mdcath_real_bb (aa),
          data/external/mdcath_div/{_manifest.json,<pid>.npz}
          (bioemu: output/phase_b/cached_desc/bioemu/{rmsf_bioemu_allframes.npy,splits.json},
          rmsf_residue_set_bioemu.json, <name>_bioemu.npz, bioemu_cath_extras, bioemu_cath_real_bb)
  writes  output/probe_rmsf_div/baseline_<feature>[_seed<k>]__test.json
          (bioemu: output/probe_rmsf_bioemu/baseline_<feature>[_seed<k>]__test.json)
          Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): next to each JSON,
          <json stem>.preds.npz (downstream/protein_preds.py: val / test predictions of the
          selected epoch, labels, protein ids, residue index and label row), recorded with its sha256 in
          the JSON's `per_protein_preds`; read only by downstream/protein_level_stats.py, which
          refuses results without it. --redo-missing-preds recomputes an existing result that
          lacks it (otherwise it is skipped with a note).

How to run
----------
    python downstream/probe_rmsf_baselines.py --feature aminoaseed --seed 0 --device cuda:0 \\
        [--dataset bioemu] [--rmsf-label allframes] [--residue-set common]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in downstream/
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.paths import MDCATH_DIV_SRC      # noqa: E402  data/external/mdcath_div
from probe_rmsf_eval import fit_probe_early_stopping, MAX_EPOCHS, PATIENCE   # noqa: E402
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from probe_rmsf_eval import write_flat_rmsf_preds, rmsf_meta   # noqa: E402
from protein_preds import PREDS_KEY, needs_rerun               # noqa: E402
from function_common import atomic_write_json                  # noqa: E402
# Common residue set, strict feature placement, His normalisation (preretrain_audit_r2
# Q3 / Q29, 2026-10-16).
from rmsf_common import (ResidueCoverageError, check_layout_matches_set,  # noqa: E402
                         load_mdcath_real_bb_seq, mdcath_div_layout, place_per_pid_features,
                         resolve_residue_set, residue_set_row_mask, restrict_splits_to_rows,
                         row_groups, dataset_layout, FLAT_DATASETS, RESULT_DIRS,
                         BIOEMU_EXTRAS_DIR, BIOEMU_REAL_BB_DIR, CacheError,
                         check_cache_provenance, check_token_ids, file_record,
                         VOTE_RULE, vote_rule_problem,
                         check_label_hash, records_changed, check_one_split, splits_record,
                         check_bioemu_split_layout, check_flat_label_sidecar)
MDCATH_DIV = str(MDCATH_DIV_SRC)
SPLITS = ROOT / 'output' / 'full_train' / 'cached_desc' / 'splits.json'
# FIXED(user decision 2026-10-16, all-frames labels): old = {'converged', 'legacy'}; new =
#   rmsf_common.RMSF_LABEL_FILES['mdcath_div'] = {'allframes' (default), 'converged5' (the
#   former 'converged'), 'legacy'}.
from rmsf_common import RMSF_LABEL_FILES, RMSF_LABEL_DEFAULT, normalize_label_name  # noqa: E402
RMSF_LABELS = RMSF_LABEL_FILES['mdcath_div']  # code_fix P3
CACHE = ROOT / 'data' / 'cached_descriptors'
OUT_DIR = ROOT / 'output' / 'probe_rmsf_div'

AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWYX')
AA_IDX = {a: i for i, a in enumerate(AA_VOCAB)}


def get_div_layout():
    """(pids, {pid: (start, end)}, total) of the mdCATH-div label rows (manifest order,
    ca_5 lengths); moved to rmsf_common.mdcath_div_layout on 2026-10-16 (same logic)."""
    return mdcath_div_layout(MDCATH_DIV)


def _seq_one_hot(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), len(AA_VOCAB)), dtype=np.float32)
    for i, a in enumerate(seq):
        x[i, AA_IDX.get(a, AA_IDX['X'])] = 1.0
    return x


def _vote(tk: np.ndarray, cent: np.ndarray) -> np.ndarray:
    """K-frame mini3di tokens (L, K) -> chosen state per residue (plurality, else the
    centroid nearest to the frames' mean centroid).

    FIXED(function_reaudit pre-run fix 7, 2026-10-19): now
    ensembits.baselines.mini3di_baselines.vote_from_frame_tokens (float64 distances, lowest
    letter id on ties), the ANOVA's rule. Old (`_vote_float32`, kept for reference): the same
    vote with float32 distances and argmin, which picked either of two exactly tied centroids.
    """
    from ensembits.baselines.mini3di_baselines import vote_from_frame_tokens
    return vote_from_frame_tokens(tk, cent)


def _vote_float32(tk: np.ndarray, cent: np.ndarray) -> np.ndarray:
    """The vote before 2026-10-19 (not used)."""
    M = cent.shape[0]
    L, K = tk.shape
    chosen = np.zeros(L, dtype=np.int64)
    for i in range(L):
        ti = tk[i]
        counts = np.bincount(ti, minlength=M)
        mx = counts.max()
        n_mx = int((counts == mx).sum())
        if mx >= 2 and n_mx == 1:
            chosen[i] = int(np.argmax(counts))
        else:
            avg = cent[ti].mean(axis=0)
            chosen[i] = int(np.argmin(np.linalg.norm(cent - avg, axis=1)))
    return chosen


def per_pid_features(spec: str, pids, offsets, dataset: str = 'mdcath_div',
                     caches: list | None = None, verify: bool = True, required=None):
    """{pid: (L, D) float32} for one baseline, and D. Feature construction as before
    2026-10-16 (aa: sequence source and His normalisation changed, see below).

    dataset "bioemu" (phase 2): the same features from the BioEmu caches
    (`<name>_bioemu.npz`, keyed by system, built by the data_prep/baselines builders with
    --dataset bioemu); aa from the extras' sequence; pca_eig from bb_10.
    """
    sfx = FLAT_DATASETS[dataset]['feature_suffix']
    caches = [] if caches is None else caches

    def use(path, codebook=None, frames='frame 0', uses_sequence=False):
        # FIXED(preretrain_audit_r2 F1, 2026-10-16): old = any file of the right name was
        #   read. New = the cache must carry a current provenance sidecar
        #   (rmsf_common.check_cache_provenance; CacheError otherwise); its sha256 goes into
        #   the result JSON. --allow-unverified-cache (side experiments only) records the
        #   file hash with verified = False instead.
        if verify:
            caches.append(check_cache_provenance(path, dataset, codebook=codebook, frames=frames,
                                                 uses_sequence=uses_sequence, required=required))
        else:
            caches.append({**file_record(path), 'verified': False})
        return path
    if spec == 'aa' and dataset == 'bioemu':
        # extras must come from build_bioemu_extras.py with no mismatch (folder sidecar)
        side = Path(str(BIOEMU_EXTRAS_DIR) + '.provenance.json')
        prov = json.loads(side.read_text()) if side.exists() else None
        if verify and (prov is None or prov.get('mismatch')):
            raise CacheError(f'{BIOEMU_EXTRAS_DIR.name}: no provenance sidecar or re-derivation '
                             f'mismatches; rebuild with data_prep/build_bioemu_extras.py')
        if side.exists():
            caches.append(file_record(side))
        out = {}
        for pid in pids:
            f = BIOEMU_EXTRAS_DIR / f'{pid}.npz'
            if f.exists():
                with np.load(f, allow_pickle=False) as z:
                    out[pid] = _seq_one_hot(str(z['seq']))
        return out, len(AA_VOCAB)
    if spec == 'pca_eig' and dataset == 'bioemu':
        out = {}
        for pid in pids:
            f = BIOEMU_REAL_BB_DIR / f'{pid}.npz'
            if not f.exists(): continue
            with np.load(f, allow_pickle=False) as z:
                ca10 = z['bb_10'][:, :, 1].astype(np.float32)
            cen = ca10 - ca10.mean(axis=0, keepdims=True)
            cov = np.einsum('pli,plj->lij', cen, cen) / ca10.shape[0]
            out[pid] = np.linalg.eigvalsh(cov)[:, ::-1].astype(np.float32)
        return out, 3
    if spec == 'aa':
        # FIXED(code_fix P6, 2026-10-16): sequences from the mdCATH real-backbone cache
        #   first (every domain with a backbone file), the pairs pickle only as fallback.
        #   The pickle alone covered 1,797 of 2,442 domains (75.9% of residues), so AA was
        #   scored on a different residue set than every other method.
        # FIXED(preretrain_audit_r2 Q29, 2026-10-16): old = the cache's 'X' (= CHARMM
        #   HSD/HSE/HSP) went into the X slot, class H was never used. New = X -> H
        #   (rmsf_common.normalize_mdcath_real_bb_seq; refuses a sequence that already has
        #   H). The pickle is only used for domains without a backbone file (outside the
        #   common residue set) and only when its length equals the label's.
        out = {}
        pkl = CACHE / 'aa2ensembits_pairs_mdcath_v2.pkl'
        seq_pkl = {}
        if pkl.exists():
            for k, v in pickle.load(open(pkl, 'rb')).items():
                pid = k.split(':')[-1] if ':' in k else k
                if 'seq' in v: seq_pkl[pid] = v['seq']
        for pid in pids:
            s, e = offsets[pid]
            seq = load_mdcath_real_bb_seq(pid)
            if seq is None:
                seq = seq_pkl.get(pid)
                if seq is None or len(seq) != e - s:
                    continue
            out[pid] = _seq_one_hot(seq)
        return out, len(AA_VOCAB)

    # FIXED(preretrain_audit_r2 G14, 2026-10-17): the options esm2, protprofile_K5,
    #   aug_rot6d, ensembits_ifold*, big_* and *_ourdata were removed: no builder of this
    #   codebase writes their caches (with a provenance sidecar), so they could never pass
    #   the cache check (F1).
    if spec == 'mini3di':
        name, D = f'mini3di_features_{sfx}.npz', 10
        d = np.load(use(CACHE / name))
        return {pid: d[pid].astype(np.float32) for pid in pids if pid in d.files}, D

    if spec == 'protprofile_K10':
        # K=10 mdcath ProtProfileMD with mini3di centroid annotation.
        # Per-residue feature: [p_0, c_0_x, c_0_y, ..., p_19, c_19_x, c_19_y] (60-D).
        d = np.load(use(CACHE / f'protprofile_K10_{sfx}.npz', frames='first 10 of bb_10',
                        uses_sequence=True))
        cent = np.load(CACHE / 'mini3di_centroids.npy').astype(np.float32)  # (20, 2)
        out = {}
        for pid in pids:
            if pid not in d.files: continue
            hist = d[pid].astype(np.float32)
            L = hist.shape[0]
            cb = np.broadcast_to(cent, (L, 20, 2))
            out[pid] = np.concatenate([hist[..., None], cb], axis=-1).reshape(L, 60)
        return out, 60

    if spec == 'vote_3di':
        # K=10 mdcath: plurality vote over 10 frames; tie → average then NN.
        d = np.load(use(CACHE / f'mini3di_tokens_K10_{sfx}.npz', frames='all 10'))
        cent = np.load(CACHE / 'mini3di_centroids.npy').astype(np.float32)
        M = cent.shape[0]
        # FIXED(preretrain_audit_r2 F4, 2026-10-16): old = np.clip of the ids; new = error.
        return ({pid: cent[_vote(check_token_ids(d[pid], M, f'vote_3di {pid}'), cent)]
                 for pid in pids if pid in d.files}, cent.shape[1])

    if spec == 'pca_eig':
        # K=10 mdcath: per-residue cov of 10 Cα positions → 3 eigenvalues desc.
        out = {}
        for pid in pids:
            try:
                with np.load(os.path.join(MDCATH_DIV, f'{pid}.npz'), allow_pickle=False) as dn:
                    ca10 = dn['ca_10'].astype(np.float32)   # (10, L, 3)
            except Exception:
                continue
            P = ca10.shape[0]
            cen = ca10 - ca10.mean(axis=0, keepdims=True)
            cov = np.einsum('pli,plj->lij', cen, cen) / P
            out[pid] = np.linalg.eigvalsh(cov)[:, ::-1].astype(np.float32)
        return out, 3

    if spec == 'random':
        # Deterministic per-(pid, residue) random (2,) vector.
        D = 2
        out = {}
        for pid in pids:
            s, e = offsets[pid]
            h = 2166136261
            for ch in pid:
                h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            out[pid] = np.random.default_rng(h).standard_normal((e - s, D)).astype(np.float32)
        return out, D

    if spec == '3di_tokens':
        d = np.load(use(CACHE / f'mini3di_tokens_{sfx}.npz'))
        cen = np.load(CACHE / 'mini3di_centroids.npy').astype(np.float32)  # (20, 2)
        out = {}
        for pid in pids:
            if pid not in d.files: continue
            # FIXED(preretrain_audit_r2 H4, 2026-10-16): old = ids outside [0, 20) became a
            #   zero vector; new = error (mini3di's masked terminals are state 2, in range).
            out[pid] = cen[check_token_ids(d[pid], 20, f'3di_tokens {pid}')]
        return out, cen.shape[1]

    if spec in ('aminoaseed', 'protoken', 'esm3struct',
                'aminoaseed_retrained', 'esm3struct_retrained'):
        cb_path = CACHE / f'{spec}_codebook.npy'
        cb = np.load(cb_path).astype(np.float32)
        # FIXED(preretrain_audit_r2 G6, 2026-10-17): old = ProToken counted as using the
        #   sequence; its encoder input has no amino-acid type (poly-Ala gives the same
        #   tokens), so no feature here reads a sequence except aa.
        d = np.load(use(CACHE / f'{spec}_tokens_{sfx}.npz', codebook=cb_path))
        # FIXED(preretrain_audit_r2 F4, 2026-10-16): old = np.clip of the ids; new = error.
        return ({pid: cb[check_token_ids(d[pid], cb.shape[0], f'{spec} {pid}')]
                 for pid in pids if pid in d.files}, cb.shape[1])

    raise ValueError(f'Unknown feature spec {spec!r}')


def load_per_residue_features(spec: str, pids, offsets, total, required=None,
                              dataset: str = 'mdcath_div', caches: list | None = None,
                              verify: bool = True):
    """Returns (feat (total, D) float32, covered mask (total,) bool, in_dim); the records
    of the caches read are appended to `caches` (see per_pid_features).

    FIXED(preretrain_audit_r2 Q3/E4, 2026-10-16): old = each branch cut the feature to
    min(len(feature), domain length) and skipped absent domains silently. New = a length
    mismatch is an error (ResidueCoverageError) for every domain; a domain of `required`
    (the common residue set) without a feature is an error; domains outside it may be
    absent (before 2026-10-17: the 8 mdCATH-div domains without a real backbone).
    """
    per_pid, D = per_pid_features(spec, pids, offsets, dataset, caches, verify, required)
    feat, mask = place_per_pid_features(per_pid, offsets, total, D, required=required,
                                        what=spec)
    return feat, mask, D


class Probe(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.mlp(x).squeeze(-1)


# (2026-10-16, F11: the unused ProtProfileMLP head was removed; the ProtProfile baseline
#  uses the 60-D feature with Probe, as before.)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', required=True,
                    # G14 (2026-10-17): only features whose caches this codebase builds
                    choices=['aa', 'mini3di', '3di_tokens', 'protprofile_K10',
                             'vote_3di', 'pca_eig', 'random',
                             'aminoaseed', 'protoken', 'esm3struct',
                             'aminoaseed_retrained', 'esm3struct_retrained'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--batch-size', type=int, default=2048)
    # FIXED(code_fix P3, 2026-10-16): which mdCATH-div RMSF label to use. "converged"
    #   (default) = data_prep/build_mdcath_div_rmsf_labels.py (superposition iterated to
    #   convergence, pose-independent); "legacy" = the old rmsf_div.npy (3 rounds from the
    #   mean of the raw frames), only to reproduce earlier numbers.
    #   FIXED(user decision 2026-10-16): default 'allframes' = every 320 K frame of the 5
    #   replicas, converged superposition; 'converged5' = the 5-frame label above (was
    #   'converged'); 'legacy' as before.
    ap.add_argument('--rmsf-label', default=RMSF_LABEL_DEFAULT,
                    choices=sorted({n for d in FLAT_DATASETS for n in RMSF_LABEL_FILES[d]}))
    # Phase 2 (2026-10-16): the same probe on BioEmu-CATH.
    ap.add_argument('--dataset', default='mdcath_div', choices=tuple(FLAT_DATASETS),
                    help='mdcath_div (default) or bioemu')
    ap.add_argument('--seed', type=int, default=0)
    # FIXED(preretrain_audit_r2 Q3, 2026-10-16): old = each feature scored on the residues
    #   it happened to cover (ProtProfileMD without the 8 Cα-only domains, AA without
    #   1a6tB01, ...). New = the common residue set (default; refused if absent; every
    #   domain of the set must have a feature of the right length); "all" = legacy.
    ap.add_argument('--allow-unverified-cache', action='store_true',
                    help='accept feature caches without a current provenance sidecar (side '
                         'experiments only; recorded, and the compilers refuse such rows)')
    ap.add_argument('--residue-set', default='common',
                    help='"common" (default): the common residue set of every method; "all": '
                         'legacy, the residues this feature covers; or a residue-set JSON path')
    # FIXED(final_fix_list A4 (a), 2026-10-19): new; as in probe_rmsf_eval.py.
    ap.add_argument('--redo-missing-preds', action='store_true',
                    help='recompute an existing result without a per-protein prediction sidecar')
    args = ap.parse_args(argv)
    ds = args.dataset
    if args.rmsf_label not in RMSF_LABEL_FILES[ds]:
        print(f'[error] --rmsf-label {args.rmsf_label} does not exist for {ds}'); return 2
    OUT_DIR = ROOT / 'output' / RESULT_DIRS[ds]
    SPLITS = FLAT_DATASETS[ds]['splits']
    try:
        rs, rs_record = resolve_residue_set(args.residue_set, ds)
    except (FileNotFoundError, ValueError) as e:
        print(f'[error] {e}'); return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_tag = '' if args.seed == 0 else f'_seed{args.seed}'
    # "__test": early stopping on val, Spearman on test (code_fix §7 / C4).
    out_path = OUT_DIR / f'baseline_{args.feature}{seed_tag}__test.json'
    _label_path = RMSF_LABEL_FILES[ds][args.rmsf_label]
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        # FIXED(preretrain_audit_r2 F1 / F5, 2026-10-16): nor one whose label contents or
        #   feature caches have changed since.
        _why = records_changed(prev.get('feature_caches'))
        if SPLITS.exists() and prev.get('splits_sha256') != splits_record(SPLITS)['splits_sha256']:
            _why.append('split file')                       # G4 (2026-10-17)
        if prev.get('cache_verified') is not (not args.allow_unverified_cache):
            _why.append('cache verification')
        if vote_rule_problem(prev):                         # 2026-10-19
            _why.append('vote rule')
        if _label_path.exists():
            try:
                if prev.get('rmsf_label_sha256') != check_label_hash(_label_path, np.load(_label_path)):
                    _why.append('label contents')
            except CacheError as e:
                _why.append(str(e))
        if _why:
            print(f'[error] {out_path.name} exists and differs ({", ".join(_why)}); move it aside')
            return 1
        # FIXED(code_fix P3, 2026-10-16): a result made with another RMSF label is not reused.
        _prev_label = normalize_label_name(ds, prev.get('rmsf_label', 'legacy'))
        if _prev_label != args.rmsf_label:
            print(f'[error] {out_path.name} was made with the {_prev_label} RMSF label, not '
                  f'{args.rmsf_label}; move it aside'); return 1
        # FIXED(preretrain_audit_r2 Q3, 2026-10-16): nor one made on another residue set.
        if prev.get('residue_set_sha256') != rs_record['residue_set_sha256']:
            print(f'[error] {out_path.name} was made on residue set '
                  f'{prev.get("residue_set_sha256")!r}, not {rs_record["residue_set_sha256"]!r}; '
                  f'move it aside'); return 1
        _np = needs_rerun(prev.get(PREDS_KEY), out_path)      # A4 (a), 2026-10-19
        if _np is None or not args.redo_missing_preds:
            print(f'[skip] {out_path.name}' + (f' (note: {_np}; --redo-missing-preds recomputes it)'
                                               if _np else '')); return 0
        print(f'[redo] {out_path.name}: {_np}', flush=True)

    print(f'[setup] feature={args.feature}', flush=True)
    pids, offsets, total = get_div_layout() if ds == 'mdcath_div' else dataset_layout(ds)
    print(f'[layout] {len(pids)} pids, {total} residues', flush=True)

    RMSF = RMSF_LABEL_FILES[ds][args.rmsf_label]
    if not RMSF.exists():
        print(f'[error] RMSF label {RMSF} missing (build it with data_prep/'
              f'build_{"bioemu" if ds == "bioemu" else "mdcath_div"}_rmsf_labels.py)'); return 1
    rmsf = np.load(RMSF)
    try:
        label_sha = check_label_hash(RMSF, rmsf)          # F5: content hash of the label
        check_flat_label_sidecar(RMSF, ds, args.rmsf_label, len(rmsf))   # G1 (2026-10-17)
    except (CacheError, ValueError) as e:
        print(f'[error] {e}'); return 1
    if len(rmsf) != total:
        print(f'[error] {RMSF.name} has {len(rmsf)} rows, the layout {total}'); return 1
    splits = json.loads(SPLITS.read_text())
    train_idx = np.array(splits['train']); val_idx = np.array(splits['val'])
    test_idx = np.array(splits['test'])

    caches = []          # records of the feature caches read (F1)
    try:
        if rs is not None:
            check_layout_matches_set(rs, offsets)
        feat, mask, in_dim = load_per_residue_features(
            args.feature, pids, offsets, total,
            required=set(rs['proteins']) if rs is not None else None, dataset=ds,
            caches=caches, verify=not args.allow_unverified_cache)
        print(f'[features] in_dim={in_dim} covered_residues={mask.sum()}/{total} '
              f'({mask.sum()/total*100:.1f}%)', flush=True)
        if rs is not None:
            # Common residue set: every set row is covered (checked above); train / val /
            # test = the set rows of each split.
            set_mask = residue_set_row_mask(rs, total)
            if not mask[set_mask].all():
                raise ResidueCoverageError(f'{args.feature}: {(~mask[set_mask]).sum()} residues '
                                           f'of the common set have no feature')
            sp_r = restrict_splits_to_rows({'train': train_idx, 'val': val_idx,
                                            'test': test_idx}, set_mask)
            train_idx, val_idx, test_idx = sp_r['train'], sp_r['val'], sp_r['test']
        else:
            # Legacy (--residue-set all): the residues this feature covers.
            train_idx = train_idx[mask[train_idx]]
            val_idx = val_idx[mask[val_idx]]
            test_idx = test_idx[mask[test_idx]]
    except (ResidueCoverageError, CacheError) as e:
        print(f'[error] {e}'); return 1
    print(f'[splits] set={rs_record["residue_set"]} train={len(train_idx)} val={len(val_idx)} '
          f'test={len(test_idx)}', flush=True)
    groups = row_groups(offsets, total)   # protein index per row (per-protein Spearman, Q2)
    # FIXED(preretrain_audit_r2 G4, 2026-10-17): new: every protein lies in one split only;
    #   the split file's hash goes into the JSON (BioEmu: the split was built from this layout).
    try:
        check_one_split(groups, {'train': train_idx, 'val': val_idx, 'test': test_idx}, list(offsets))
        if ds == 'bioemu':
            check_bioemu_split_layout(SPLITS)
    except (ResidueCoverageError, CacheError) as e:
        print(f'[error] {e}'); return 1

    dev = args.device
    torch.manual_seed(args.seed)
    model = Probe(in_dim).to(dev)
    t0 = time.time()
    t = lambda a: torch.tensor(a, device=dev)
    # Same optimiser, batch order and stopping rule as the Ensembits probe.
    res = fit_probe_early_stopping(model, t(feat[train_idx]), t(rmsf[train_idx]),
                                   t(feat[val_idx]), t(rmsf[val_idx]),
                                   t(feat[test_idx]), t(rmsf[test_idx]),
                                   device=dev, seed=args.seed, batch_size=args.batch_size,
                                   max_epochs=args.max_epochs, patience=args.patience,
                                   groups_va=groups[val_idx], groups_te=groups[test_idx],
                                   return_preds=True)            # A4 (a): sidecar below
    preds = res.pop('_preds', None)
    out = {'feature': args.feature, 'in_dim': int(in_dim), **res,
           'coverage': float(mask.sum() / total), 'seed': args.seed,
           'rmsf_label': args.rmsf_label, 'rmsf_label_file': RMSF.name,
           'rmsf_label_sha256': label_sha,               # FIXED(preretrain_audit_r2 F5, 2026-10-16)
           'dataset': ds, **rs_record, **splits_record(SPLITS),
           # FIXED(preretrain_audit_r2 F1, 2026-10-16): new keys: the caches read (sha256)
           'feature_caches': caches, 'cache_verified': not args.allow_unverified_cache,
           'batch_size': args.batch_size,
           **({'vote_rule': VOTE_RULE} if args.feature == 'vote_3di' else {})}   # 2026-10-19
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar <out>.preds.npz recorded
    #   in `per_protein_preds` (probe_rmsf_eval.write_flat_rmsf_preds); atomic JSON write.
    out[PREDS_KEY] = write_flat_rmsf_preds(
        out_path, preds, {'val': val_idx, 'test': test_idx}, rmsf, groups, list(offsets),
        [int(s0) for s0, _ in offsets.values()], rmsf_meta(out))
    atomic_write_json(out_path, out)
    print(f'[done] {out_path.name} test Spearman={res["test_spearman"]:.4f} '
          f'(per protein {res["test_spearman_per_protein_mean"]:.4f}; {time.time()-t0:.0f}s)',
          flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
