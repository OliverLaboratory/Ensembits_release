"""Shared pieces of the RMSF probes: common residue set, per-protein Spearman, sequences.

What it does
------------
Helpers used by `probe_rmsf_eval.py` (Ensembits, mdCATH-div), `probe_rmsf_baselines.py`
(baselines, mdCATH-div), `probe_rmsf_misato.py` (Ensembits + baselines, MISATO), the
residue-set builder `data_prep/build_rmsf_residue_sets.py`, the table compiler
`_compile_rebuttal_results.py` and `rmsf_selection_table.py`:

  * Common residue set (user decision Q3, 2026-10-16; same principle as the ANOVA
    fix P6): one residue set per dataset, chosen from the data before any method is
    run, on which every method is trained, validated and tested.
      - mdCATH-div: every residue of the domains that have real backbone atoms
        (a file in data/cached_descriptors/mdcath_real_bb): all 2,442 since the
        re-extraction of 2026-10-17 (data_prep/build_mdcath_real_bb.py); a domain
        without a file is an error of build_rmsf_residue_sets.py unless
        --allow-missing, which excludes it for every method (recorded).
      - MISATO: every residue of the complexes that are in data/misato_splits_v2.json and
        have an RMSF label and a misato_real_bb (K = 8) file.
    The set is a JSON file (`RESIDUE_SET_FILES`) written by the builder; its sha256 is
    computed over (dataset, sorted proteins with row start and length) and recorded in
    every result JSON. `load_residue_set` recomputes it, so an edited file is refused.
    A probe must cover 100% of the set: a missing protein or a feature / label length
    mismatch inside the set raises `ResidueCoverageError`.
  * `per_protein_spearman` (user decision Q2): Spearman between prediction and label
    within each protein, mean / median over the proteins with at least
    `PER_PROTEIN_MIN_RESIDUES` residues and non-constant labels. The eligible proteins
    depend on the labels only, so every method is averaged over the same proteins; a
    method whose predictions are constant within an eligible protein scores 0 there
    (no ranking information), counted in `n_<split>_proteins_constant_pred`. The unit
    "protein" is a CATH domain on mdCATH-div, a simulated system on BioEmu-CATH (a domain
    simulated in both deposits counts twice) and a complex on MISATO (G15, 2026-10-17).
  * Amino-acid sequences without force-field residue names (preretrain audit Q29):
    `three_to_one` maps CHARMM / AMBER names (HSD/HSE/HSP, HID/HIE/HIP, CYX/CYM, ASH,
    GLH, LYN) to the standard one-letter code; `normalize_mdcath_real_bb_seq` maps the
    'X' of the mdCATH real-backbone cache to 'H' (see the evidence in its docstring).
  * Result-JSON checks shared by the compilers, the selection table and the battery:
    RMSF label, residue-set hash and the checkpoint block (`rmsf_result_problems`,
    `check_rmsf_group`, `strict_same_checkpoint`, `stale_rmsf_outputs`).
  * RMSF label registry (`RMSF_LABEL_FILES`; user decision 2026-10-16: every label uses
    all MD frames with the converged superposition, name "allframes"; the older labels
    stay selectable for reproduction) and the dataset configuration of the flat probes
    (`FLAT_DATASETS`: mdcath_div, bioemu) with the BioEmu-CATH paths (phase 2).

Paper
-----
`2_results.tex` \\label{result:rmsf} (Table tab:rmsf: "Spearman ... between predicted
and ground-truth RMSF on held-out residues"; per-protein Spearman as a second number);
`appendix.tex` \\label{app:baselines} ("for every method": same residues).

Origin
------
New (2026-10-16, preretrain_audit_r2 Q2, Q3, Q11, Q17, Q29, Q29b, Q29c; user decisions
Q2, Q3). `mdcath_div_layout` is `probe_rmsf_baselines.get_div_layout` moved here
(unchanged logic).

Used by the reported model: yes (every RMSF probe and the RMSF tables).

Inputs / outputs
----------------
Reads (relative to REPO_ROOT):
  data/cached_descriptors/rmsf_residue_set_{mdcath_div,misato}.json   (residue sets)
  data/external/mdcath_div/{_manifest.json,<pid>.npz}  (ENSEMBITS_MDCATH_DIV_SRC; layout)
  data/cached_descriptors/bioemu_cath_real_bb/layout_rows.json  (BioEmu layout)
Writes nothing (the builder writes the residue sets).

How to run
----------
Library module; `python -m pytest tests/test_rmsf_probes.py tests/test_rmsf_bioemu.py` tests it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.paths import MDCATH_DIV_SRC      # noqa: E402  data/external/mdcath_div
from ensembits.paths import MDCATH_DIV_RMSF, MDCATH_DIV_RMSF_LEGACY  # noqa: E402

CACHE = ROOT / 'data' / 'cached_descriptors'
MDCATH_REAL_BB_DIR = CACHE / 'mdcath_real_bb'
MISATO_REAL_BB_DIR = CACHE / 'misato_real_bb'          # K = 8 cache (bb_8 / ca_8 / seq)
# Kinds of the MISATO backbone breaks (data_prep/build_misato_chain_breaks.py, 2026-10-18).
MISATO_CHAIN_BREAKS = CACHE / 'misato_chain_breaks.json'
# Deposited PDB entries (gzipped mmCIF from RCSB, + _manifest.json) read by that builder.
RCSB_MMCIF_DIR = ROOT / 'data' / 'external' / 'rcsb_mmcif'
DATASETS = ('mdcath_div', 'misato', 'bioemu')
RESIDUE_SET_FILES = {'mdcath_div': CACHE / 'rmsf_residue_set_mdcath_div.json',
                     'misato': CACHE / 'rmsf_residue_set_misato.json',
                     'bioemu': CACHE / 'rmsf_residue_set_bioemu.json'}
# BioEmu-CATH RMSF probe (phase 2, 2026-10-16). Inputs: the system-keyed real-backbone
# cache (bb_10) and its extras (FPS picks, sequence, Cβ); everything the probe reads that
# is BioEmu-specific (label, row map, splits, descriptor cache) is in BIOEMU_DIR.
BIOEMU_REAL_BB_DIR = CACHE / 'bioemu_cath_real_bb'
BIOEMU_LAYOUT = BIOEMU_REAL_BB_DIR / 'layout_rows.json'
BIOEMU_EXTRAS_DIR = CACHE / 'bioemu_cath_extras'
BIOEMU_RAW_ROOT = ROOT / 'data' / 'reviewer_3_special_corpus' / 'bioemu_cath' / 'extracted'
BIOEMU_DIR = ROOT / 'output' / 'phase_b' / 'cached_desc' / 'bioemu'
BIOEMU_ROWS = BIOEMU_DIR / 'rmsf_bioemu_rows.json'       # row -> system map of the label
BIOEMU_SPLITS = BIOEMU_DIR / 'splits.json'               # residue rows {train, val, test}
# MISATO splits whose union defines the MISATO set (the probe's --split choices).
MISATO_SPLIT_NAMES = ('sequence', 'structure', 'random')
# RMSF labels (user decision 2026-10-16, supersedes Q1 / item 8: every label uses ALL MD
# frames with the converged superposition). Name -> file; the name is recorded in every
# result JSON as "rmsf_label" and the compilers refuse tables that mix names.
#   mdCATH-div  allframes  : every 320 K frame of the 5 replicas (build_mdcath_div_rmsf_labels.py)
#               converged5 : the 5 ca_5 frames, converged superposition (2026-10-16 morning;
#                            result JSONs of that day record it as "converged")
#               legacy     : the old rmsf_div.npy (3 rounds from the raw mean)
#   MISATO      allframes  : all 100 frames of MD.hdf5 (build_misato_rmsf_labels.py)
#               frames8    : the 8 frames of coords_ca8_misato.pkl superposed on frame 0 (old)
RMSF_LABEL_DEFAULT = 'allframes'
MDCATH_DIV_RMSF_ALLFRAMES = ROOT / 'output' / 'phase_b' / 'cached_desc' / 'rmsf_div_allframes.npy'
RMSF_LABEL_FILES = {
    'mdcath_div': {'allframes': MDCATH_DIV_RMSF_ALLFRAMES,
                   'converged5': MDCATH_DIV_RMSF,          # rmsf_div_converged.npy
                   'legacy': MDCATH_DIV_RMSF_LEGACY},      # rmsf_div.npy
    'misato': {'allframes': CACHE / 'misato_rmsf_allframes.npz',
               'frames8': CACHE / 'misato_rmsf.npz'},
    #   BioEmu      allframes  : every frame of every run*_protein trajectory
    'bioemu': {'allframes': BIOEMU_DIR / 'rmsf_bioemu_allframes.npy'},
}
# Old names written into result JSONs before the label rework.
_LABEL_ALIASES = {'mdcath_div': {'converged': 'converged5'}, 'misato': {None: 'frames8'}}


def normalize_label_name(dataset: str, name):
    """Current name of an RMSF label recorded in a result JSON (old aliases mapped)."""
    return _LABEL_ALIASES.get(dataset, {}).get(name, name)
PER_PROTEIN_MIN_RESIDUES = 10
# Result folders of the RMSF probes (under REPO_ROOT/output).
RESULT_DIRS = {'mdcath_div': 'probe_rmsf_div', 'misato': 'probe_rmsf_misato',
               'bioemu': 'probe_rmsf_bioemu'}
# Datasets whose label is one flat per-residue array with fixed rows (mdCATH-style probes:
# probe_rmsf_eval.py / probe_rmsf_baselines.py --dataset): residue splits, descriptor cache
# folder and the cache-name suffix of the baseline features.
FLAT_DATASETS = {
    'mdcath_div': {'splits': ROOT / 'output' / 'full_train' / 'cached_desc' / 'splits.json',
                   'desc_dir': ROOT / 'output' / 'phase_b' / 'cached_desc',
                   'feature_suffix': 'mdcath'},
    'bioemu': {'splits': BIOEMU_SPLITS, 'desc_dir': BIOEMU_DIR, 'feature_suffix': 'bioemu'},
}


class ResidueCoverageError(ValueError):
    """A method does not cover the common residue set exactly (missing protein or length)."""


class CacheError(ValueError):
    """A feature / token cache is stale, unverified or holds invalid values."""


def check_token_ids(tok, n_codes: int, what: str = 'tokens') -> np.ndarray:
    """int64 token ids, all in [0, n_codes); otherwise CacheError.

    FIXED(preretrain_audit_r2 F4, 2026-10-16): old = np.clip(tok, 0, n_codes - 1) in the
    probes, which silently mapped out-of-range ids (e.g. a pad value) to a real code. New =
    any id outside the codebook is an error.
    """
    t = np.asarray(tok).astype(np.int64)
    bad = (t < 0) | (t >= n_codes)
    if bad.any():
        raise CacheError(f'{what}: {int(bad.sum())} token ids outside [0, {n_codes}) '
                         f'(e.g. {int(t[bad][0])})')
    return t


# ---------------------------------------------------------------- residue sets
def residue_set_sha256(dataset: str, proteins: dict) -> str:
    """sha256 of the set content: dataset + sorted [pid, row_start, length].

    proteins: {pid: {"length": L, "row_start": s or None}}. Independent of dict order
    and of everything else in the file (rule text, provenance).
    """
    items = sorted([str(p), (None if v.get('row_start') is None else int(v['row_start'])),
                    int(v['length'])] for p, v in proteins.items())
    blob = json.dumps({'dataset': dataset, 'proteins': items}, sort_keys=True,
                      separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


def make_residue_set(dataset: str, proteins: dict, rule: str, *, n_rows_total=None,
                     excluded=None) -> dict:
    """The JSON content of a residue set (with its sha256)."""
    if dataset not in DATASETS:
        raise ValueError(f'unknown dataset {dataset!r}')
    return {'dataset': dataset, 'rule': rule,
            'n_proteins': len(proteins),
            'n_residues': int(sum(int(v['length']) for v in proteins.values())),
            'n_rows_total': None if n_rows_total is None else int(n_rows_total),
            'proteins': proteins,
            'excluded': dict(excluded or {}),
            'sha256': residue_set_sha256(dataset, proteins)}


def load_residue_set(path) -> dict:
    """Read a residue set and check its recorded sha256 against its content."""
    path = Path(path)
    rs = json.loads(path.read_text())
    for k in ('dataset', 'proteins', 'sha256'):
        if k not in rs:
            raise ValueError(f'{path.name}: not a residue-set file (no {k!r})')
    h = residue_set_sha256(rs['dataset'], rs['proteins'])
    if h != rs['sha256']:
        raise ValueError(f'{path.name}: recorded sha256 {rs["sha256"][:12]} != content '
                         f'{h[:12]} (file edited after the builder wrote it); rebuild it')
    rs['path'] = str(path)
    return rs


def resolve_residue_set(choice: str, dataset: str):
    """--residue-set value -> (residue set or None, record for the result JSON).

    "common" (default) = RESIDUE_SET_FILES[dataset], must exist; "all" = legacy (no
    common set: each method on the residues it has features for, as before 2026-10-16;
    only to reproduce earlier numbers); anything else = path of a residue-set file.
    Raises FileNotFoundError / ValueError with a message for the CLI.
    """
    if choice == 'all':
        return None, {'residue_set': 'all', 'residue_set_sha256': None,
                      'residue_set_file': None, 'residue_set_n_proteins': None,
                      'residue_set_n_residues': None}
    path = RESIDUE_SET_FILES[dataset] if choice == 'common' else Path(choice)
    if not path.exists():
        raise FileNotFoundError(
            f'residue set {path} missing: build it with '
            f'`python data_prep/build_rmsf_residue_sets.py --dataset {dataset}`, or pass '
            f'--residue-set all (legacy, per-method residue sets; not comparable across methods)')
    rs = load_residue_set(path)
    if rs['dataset'] != dataset:
        raise ValueError(f'{path.name} is a {rs["dataset"]} residue set, not {dataset}')
    return rs, {'residue_set': 'common' if choice == 'common' else str(path.name),
                'residue_set_sha256': rs['sha256'], 'residue_set_file': path.name,
                'residue_set_n_proteins': int(rs['n_proteins']),
                'residue_set_n_residues': int(rs['n_residues'])}


def residue_set_row_mask(rs: dict, n_rows_total: int) -> np.ndarray:
    """Boolean row mask of an mdCATH-div residue set over the flat label array."""
    if rs.get('n_rows_total') is not None and int(rs['n_rows_total']) != int(n_rows_total):
        raise ResidueCoverageError(f'residue set was built for {rs["n_rows_total"]} rows, '
                                   f'the label has {n_rows_total}')
    m = np.zeros(int(n_rows_total), dtype=bool)
    for pid, v in rs['proteins'].items():
        s = v.get('row_start')
        if s is None:
            raise ValueError(f'{pid}: residue set has no row_start (not an mdCATH-div set)')
        s, L = int(s), int(v['length'])
        if s < 0 or s + L > n_rows_total:
            raise ResidueCoverageError(f'{pid}: rows {s}..{s + L} outside 0..{n_rows_total}')
        m[s:s + L] = True
    return m


def restrict_splits_to_rows(splits: dict, mask: np.ndarray) -> dict:
    """{train, val, test} row indices restricted to `mask`; every masked row must be in
    exactly one split (else ResidueCoverageError)."""
    out = {k: np.asarray(splits[k], dtype=np.int64) for k in ('train', 'val', 'test')}
    out = {k: v[mask[v]] for k, v in out.items()}
    allr = np.concatenate(list(out.values()))
    if len(np.unique(allr)) != len(allr):
        raise ResidueCoverageError('the splits share residues')
    if len(allr) != int(mask.sum()):
        raise ResidueCoverageError(f'the splits cover {len(allr)} of the {int(mask.sum())} '
                                   f'residues of the residue set')
    return out


def mdcath_div_layout(src=None):
    """Row layout of the mdCATH-div label arrays: (pids, {pid: (start, end)}, total).

    Manifest "representatives" order, each domain's residue count = ca_5 length (the
    row order of rmsf_div*.npy and of every mdCATH-div cache).
    """
    src = str(src or MDCATH_DIV_SRC)
    m = json.load(open(os.path.join(src, '_manifest.json')))
    pids = list(m['representatives'].keys())
    offsets, total = {}, 0
    for pid in pids:
        with np.load(os.path.join(src, f'{pid}.npz'), allow_pickle=True) as d:
            L = d['ca_5'].shape[1]
        offsets[pid] = (total, total + L)
        total += L
    return pids, offsets, total


def bioemu_layout(layout=None):
    """Row layout of the BioEmu label arrays: (systems, {system: (start, end)}, total),
    layout_rows.json order, each system's L (= its bb_10 length)."""
    rows = json.loads(Path(layout or BIOEMU_LAYOUT).read_text())
    names, offsets, total = [], {}, 0
    for r in rows:
        names.append(r['system'])
        offsets[r['system']] = (total, total + int(r['L']))
        total += int(r['L'])
    return names, offsets, total


def dataset_layout(dataset: str):
    """(proteins, {protein: (start, end)}, total) of a flat dataset (mdcath_div, bioemu)."""
    if dataset == 'mdcath_div':
        return mdcath_div_layout()
    if dataset == 'bioemu':
        return bioemu_layout()
    raise ValueError(f'{dataset} has no flat row layout')


def check_layout_matches_set(rs: dict, offsets: dict) -> None:
    """Every protein of an mdCATH-div residue set must sit at the same rows in `offsets`."""
    for pid, v in rs['proteins'].items():
        if pid not in offsets:
            raise ResidueCoverageError(f'{pid} of the residue set is not in the layout')
        s, e = offsets[pid]
        if (int(v['row_start']), int(v['length'])) != (s, e - s):
            raise ResidueCoverageError(f'{pid}: residue set rows {v["row_start"]}+{v["length"]}, '
                                       f'layout {s}+{e - s}')


def residue_set_groups(rs: dict, n_rows_total: int) -> np.ndarray:
    """(n_rows_total,) int protein index of every row of an mdCATH-div set, -1 outside."""
    g = np.full(int(n_rows_total), -1, dtype=np.int64)
    for i, v in enumerate(rs['proteins'].values()):
        s, L = int(v['row_start']), int(v['length'])
        g[s:s + L] = i
    return g


def row_groups(offsets: dict, total: int) -> np.ndarray:
    """(total,) int protein index of every row (order of `offsets`), -1 if none."""
    g = np.full(int(total), -1, dtype=np.int64)
    for i, (s, e) in enumerate(offsets.values()):
        g[s:e] = i
    return g


def place_per_pid_features(per_pid: dict, offsets: dict, total: int, dim: int,
                           required=None, what: str = 'feature'):
    """Scatter {pid: (L, dim)} into a (total, dim) array; returns (feat, covered mask).

    FIXED(preretrain_audit_r2 Q3/E4, 2026-10-16): old = each loader cut feature and label to
    min(len(feature), len(label)) and skipped absent proteins silently. New = a length
    mismatch is an error for every protein; a protein of `required` (the common residue
    set) without a feature is an error; others are left uncovered (legacy --residue-set all).
    """
    feat = np.zeros((int(total), int(dim)), dtype=np.float32)
    mask = np.zeros(int(total), dtype=bool)
    for pid, (s, e) in offsets.items():
        x = per_pid.get(pid)
        if x is None:
            if required is not None and pid in required:
                raise ResidueCoverageError(f'{what}: no feature for {pid}, which is in the '
                                           f'common residue set')
            continue
        x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
        if x.shape[0] != e - s:
            raise ResidueCoverageError(f'{what}: {pid} has {x.shape[0]} residues, the label '
                                       f'{e - s}')
        feat[s:e] = x; mask[s:e] = True
    return feat, mask


# ---------------------------------------------------------------- per-protein Spearman
def per_protein_spearman(pred, y, groups, min_residues: int = PER_PROTEIN_MIN_RESIDUES) -> dict:
    """Spearman within each protein, mean / median over the eligible proteins.

    Eligible = at least `min_residues` residues and non-constant labels (depends on the
    labels only). Constant predictions within an eligible protein score 0.
    Paper: 2_results.tex result:rmsf (per-protein Spearman, user decision Q2).
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    if not (len(pred) == len(y) == len(groups)):
        raise ValueError(f'lengths differ: pred {len(pred)}, y {len(y)}, groups {len(groups)}')
    # FIXED(preretrain_audit_r2 F11, 2026-10-16): old = a NaN prediction made the protein's
    #   Spearman NaN, which was scored 0. New = non-finite predictions or labels are refused.
    n_bad = int((~np.isfinite(pred)).sum()) + int((~np.isfinite(y)).sum())
    if n_bad:
        raise ValueError(f'{n_bad} non-finite predictions / labels; per-protein Spearman refused')
    order = np.argsort(groups, kind='stable')
    uniq, starts = np.unique(groups[order], return_index=True)
    bounds = list(starts) + [len(order)]
    vals, n_short, n_const_y, n_const_p = [], 0, 0, 0
    for i in range(len(uniq)):
        sl = order[bounds[i]:bounds[i + 1]]
        if len(sl) < min_residues:
            n_short += 1; continue
        yy, pp = y[sl], pred[sl]
        if np.all(yy == yy[0]):
            n_const_y += 1; continue
        if np.all(pp == pp[0]):
            vals.append(0.0); n_const_p += 1; continue
        r = spearmanr(pp, yy).correlation
        vals.append(float(r) if r is not None and np.isfinite(r) else 0.0)
    return {'mean': float(np.mean(vals)) if vals else float('nan'),
            'median': float(np.median(vals)) if vals else float('nan'),
            'n_scored': len(vals), 'n_proteins': int(len(uniq)), 'n_too_short': n_short,
            'n_constant_label': n_const_y, 'n_constant_pred': n_const_p}


# ---------------------------------------------------------------- constant predictions
# Authors' decision (2026-10-18): a probe whose predictions are constant because every
#   residue maps to ONE input (e.g. a checkpoint whose P=1 tokens collapse to one token)
#   has an undefined Spearman. The probes record `constant_prediction` (val predictions
#   constant at every epoch, so no epoch was ever selected) and `n_unique_inputs` (distinct
#   input rows over train / val / test, computed in that case); the compilers show a cell
#   whose seeds are all such cases as "—" with a footnote (CONSTANT_NOTE) instead of
#   counting them as missing seeds. Any other NaN still refuses the cell.
CONSTANT_MARK = '—†'
CONSTANT_NOTE = '† every residue maps to one token; Spearman undefined'


def n_unique_rows(*arrays) -> int:
    """Distinct rows over the given (N, ...) arrays / tensors (inputs of a probe)."""
    rows = []
    for a in arrays:
        a = a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
        rows.append(a.reshape(len(a), -1))
    return int(len(np.unique(np.concatenate(rows), axis=0)))


def constant_prediction_fields(never_selected: bool, all_constant: bool, *inputs) -> dict:
    """Result-JSON fields of the constant-prediction rule (see above)."""
    const = bool(never_selected and all_constant)
    return {'constant_prediction': const,
            'n_unique_inputs': n_unique_rows(*inputs) if const else None}


_UNIQUE_TOKENS: dict = {}


def npz_token_count(path, stop_at: int = 2) -> int:
    """Distinct values (first column if 2-D) over all pids of a token npz, counting up to
    `stop_at` (cached per path and mtime)."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime_ns, stop_at)
    if key not in _UNIQUE_TOKENS:
        seen = set()
        with np.load(path, allow_pickle=False) as z:
            for k in z.files:
                a = np.asarray(z[k])
                seen |= set(np.unique(a if a.ndim == 1 else a.reshape(len(a), -1)[:, 0]).tolist())
                if len(seen) >= stop_at:
                    break
        _UNIQUE_TOKENS[key] = len(seen)
    return _UNIQUE_TOKENS[key]


def _legacy_token_caches(js) -> list:
    """Token caches behind a result written before the constant_prediction field: the
    MISATO token caches it recorded, else the checkpoint's MISATO token cache at the same P
    (for the mdCATH-div / BioEmu-CATH results, which tokenize on the fly)."""
    out = [ROOT / c['path'] for c in (js.get('feature_caches') or [])
           if str(c.get('path', '')).endswith('_tokens_misato.npz')]
    v = (js.get('checkpoint') or {}).get('variant')
    if not out and v:
        tag = v.replace('final_model_', '').replace('final_model', '') or v
        p = js.get('p_inference')
        sufs = ['_P1inf'] if p == 1 else (['', f'_P{p}inf'] if p is not None else [''])
        out = [CACHE / f'ensembits_{tag}{s}_tokens_misato.npz' for s in sufs]
    return [q for q in out if q.exists()]


def constant_prediction_seed(js: dict, key: str) -> bool:
    """True if this seed's `key` is NaN because the prediction was constant with every
    residue mapped to one token (the "—" rule). New results: constant_prediction and
    n_unique_inputs == 1. Results written before 2026-10-18: best_epoch 0 (never selected),
    NaN, and the token cache has one distinct token."""
    v = js.get(key)
    if v is not None and v == v:
        return False
    if 'constant_prediction' in js:
        return bool(js.get('constant_prediction')) and js.get('n_unique_inputs') == 1
    if js.get('best_epoch') != 0:
        return False
    caches = _legacy_token_caches(js)
    return bool(caches) and all(npz_token_count(c) == 1 for c in caches)


def per_protein_fields(split: str, stats) -> dict:
    """Result-JSON fields of one split ("val" / "test"); NaN fields when stats is None.

    The split argument corresponds to the v2 split (benchmark_audit D3): 'val' or 'test'
    of ``data/misato_splits_v2.json`` (or the dataset's own split file)."""
    stats = stats or {'mean': float('nan'), 'median': float('nan'), 'n_scored': 0,
                      'n_proteins': 0, 'n_constant_pred': 0}
    return {f'{split}_spearman_per_protein_mean': stats['mean'],
            f'{split}_spearman_per_protein_median': stats['median'],
            f'n_{split}_proteins_scored': int(stats['n_scored']),
            f'n_{split}_proteins': int(stats['n_proteins']),
            f'n_{split}_proteins_constant_pred': int(stats['n_constant_pred'])}


# ---------------------------------------------------------------- sequences
AA20 = set('ACDEFGHIKLMNPQRSTVWY')
# Standard and force-field residue names -> one-letter code (preretrain_audit_r2 Q29).
FORCE_FIELD_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # CHARMM / AMBER protonation and disulfide states
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'HID': 'H', 'HIE': 'H', 'HIP': 'H',
    'CYX': 'C', 'CYM': 'C', 'ASH': 'D', 'GLH': 'E', 'LYN': 'K',
}


def three_to_one(name: str) -> str:
    """One-letter code of a (force-field) residue name; 'X' if unknown."""
    return FORCE_FIELD_3TO1.get(str(name).strip().upper(), 'X')


def normalize_mdcath_real_bb_seq(seq: str, pid: str = '?') -> str:
    """The `seq` of data/cached_descriptors/mdcath_real_bb/<pid>.npz with 'X' -> 'H'.

    FIXED(preretrain_audit_r2 Q29, 2026-10-16): old = used as stored, so every histidine
    was the unknown class X (the cache writes CHARMM HSD/HSE/HSP as 'X'). Evidence
    (2026-10-16, read-only): over all 2,434 files 0 'H' and 6,864 'X' (2.24% of 306,967
    residues); in the raw mdCATH h5 (pdbProteinAtoms) of 12asA00, 153lA00 and 16pkA02
    every 'X' position is HSD / HSE / HSP (13 / 5 / 6 positions) and no other residue
    name is non-standard. data_prep/build_mdcath_real_bb.py (2026-10-17) keeps this
    convention for the 8 re-extracted domains (tested). The mapping is only safe while the
    source has no 'H', so a
    sequence that contains 'H' (a rebuilt cache that maps histidine itself) is refused.
    """
    seq = str(seq)
    if 'H' in seq:
        raise ValueError(f'{pid}: mdcath_real_bb seq already contains H; the X -> H mapping '
                         f'assumes the cache writes every histidine as X (check the cache)')
    return seq.replace('X', 'H')


def load_mdcath_real_bb_seq(pid: str, rb_dir=None):
    """Normalised mdCATH-div sequence from the real-backbone cache, or None (no file)."""
    p = Path(rb_dir or MDCATH_REAL_BB_DIR) / f'{pid}.npz'
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return normalize_mdcath_real_bb_seq(str(z['seq']), pid)


# ---------------------------------------------------------------- result checks
def strict_same_checkpoint(a, b) -> bool:
    """Two checkpoint blocks name the same weights: both best.pt hashes present and equal,
    and the code hashes equal when both are recorded. Two missing hashes never match
    (unlike ensembits.featurize.same_checkpoint; preretrain_audit_r2 Q20)."""
    if not a or not b:
        return False
    ha, hb = a.get('best_pt_sha256'), b.get('best_pt_sha256')
    if ha is None or hb is None or ha != hb:
        return False
    ca, cb = a.get('code_sha256'), b.get('code_sha256')
    return ca is None or cb is None or ca == cb


# FIXED(function_reaudit pre-run fix 7, 2026-10-19): the Vote_3Di feature of the RMSF probes
#   is ensembits.baselines.mini3di_baselines.vote_from_frame_tokens (float64 distances, lowest
#   letter id on ties), the rule of the ANOVA (old: the same vote inline in float32 with
#   argmin, which picked either of two exactly tied centroids). vote_3di results record
#   VOTE_RULE; results without it are refused by the compilers and not reused by the runners.
VOTE_RULE = 'mini3di_baselines.vote_from_frame_tokens (float64, lowest letter id on ties)'


def vote_rule_problem(js: dict):
    """None unless `js` is a vote_3di result without the current VOTE_RULE."""
    if (js or {}).get('feature') == 'vote_3di' and js.get('vote_rule') != VOTE_RULE:
        return 'vote_3di built with the old vote (float32 argmin tie-break); rerun'
    return None


def rmsf_result_problems(js: dict, *, dataset: str, residue_set_sha256=None,
                         expect_label=RMSF_LABEL_DEFAULT, checkpoint=None,
                         is_model=False, label_sha256=None) -> list:
    """Reasons why one RMSF result JSON may not enter a table (empty list = accepted).

    rmsf_label (old names mapped by normalize_label_name) must be `expect_label`;
    residue_set_sha256: the hash every row of the table must carry (None = only require
    that one is recorded); is_model: the JSON must carry a checkpoint block with a best.pt
    hash, equal to `checkpoint` (strict_same_checkpoint) when given.
    """
    out = []
    if vote_rule_problem(js):                               # 2026-10-19
        out.append(vote_rule_problem(js))
    lab = normalize_label_name(dataset, js.get('rmsf_label'))
    if lab != expect_label:
        out.append(f'rmsf_label {lab!r} (expected {expect_label!r})')
    if js.get('splits_sha256') is None:                    # G4 (2026-10-17)
        out.append('no split-file hash (splits_sha256)')
    lh = js.get('rmsf_label_sha256')
    if lh is None:
        out.append('no label content hash (rmsf_label_sha256)')
    elif label_sha256 is not None and lh != label_sha256:
        out.append(f'label content {lh[:12]} (table: {label_sha256[:12]})')
    if js.get('cache_verified') is False:
        out.append('feature caches not verified (--allow-unverified-cache)')
    h = js.get('residue_set_sha256')
    if h is None:
        out.append(f'no common residue set (residue_set={js.get("residue_set")!r})')
    elif residue_set_sha256 is not None and h != residue_set_sha256:
        out.append(f'residue set {h[:12]} (table: {residue_set_sha256[:12]})')
    if is_model:
        ck = js.get('checkpoint')
        if not ck or ck.get('best_pt_sha256') is None:
            out.append('no checkpoint block with a best.pt hash')
        elif checkpoint is not None and not strict_same_checkpoint(ck, checkpoint):
            out.append(f'checkpoint best.pt {str(ck.get("best_pt_sha256"))[:12]} != '
                       f'{str(checkpoint.get("best_pt_sha256"))[:12]}')
    return out


def check_rmsf_group(records, *, dataset: str, expect_label=RMSF_LABEL_DEFAULT,
                     residue_set_sha256=None, checkpoint=None, is_model=False,
                     label_sha256=None) -> list:
    """Problems of one table cell / row: records = [(name, json)] of all its seeds.

    Every JSON must pass rmsf_result_problems; within the group the residue-set hash, the
    label content hash, the feature caches (F1) and (for a model) the checkpoint block must
    be identical across seeds.
    """
    probs = []
    for name, js in records:
        probs += [f'{name}: {p}' for p in rmsf_result_problems(
            js, dataset=dataset, residue_set_sha256=residue_set_sha256,
            expect_label=expect_label, checkpoint=checkpoint, is_model=is_model,
            label_sha256=label_sha256)]
    hs = {js.get('residue_set_sha256') for _, js in records}
    if len(hs) > 1:
        probs.append(f'seeds carry {len(hs)} different residue sets')
    for key, what in (('rmsf_label_sha256', 'label contents'),
                      ('feature_caches', 'feature caches'),
                      ('splits_sha256', 'split files')):          # G4 (2026-10-17)
        vals = {json.dumps(js.get(key), sort_keys=True) for _, js in records}
        if len(vals) > 1:
            probs.append(f'seeds carry {len(vals)} different {what}')
    # G5 (2026-10-17): identical probe settings across seeds
    sets = {tuple(js.get(k) for k in PROBE_SETTINGS) for _, js in records}
    if len(sets) > 1:
        probs.append(f'seeds use different probe settings {PROBE_SETTINGS}: {sorted(sets, key=str)}')
    if is_model:
        cks = {(js.get('checkpoint') or {}).get('best_pt_sha256') for _, js in records}
        if len(cks) > 1:
            probs.append(f'seeds carry {len(cks)} different checkpoints')
    return probs


PROBE_SETTINGS = ('max_epochs', 'patience', 'min_delta', 'batch_size', 'selection')
# FIXED(preretrain_audit_r2 K2, 2026-10-17): the default probe settings (probe_rmsf_eval /
#   _baselines / _misato: MAX_EPOCHS, PATIENCE, MIN_DELTA, --batch-size, selection). The
#   compiler refuses a row whose recorded settings deviate (tests check these equal the
#   probes' constants).
DEFAULT_PROBE_SETTINGS = {'max_epochs': 500, 'patience': 20, 'min_delta': 1e-4,
                          'batch_size': 2048, 'selection': 'val_early_stopping'}
# FIXED(benchmark_audit D3, 2026-10-18): old = data/misato_splits.json; new = the v2 split file
#   (ensembits.paths.MISATO_SPLITS, data/misato_splits_v2.json). The MISATO RMSF probe records
#   its sha256 (splits_record) and the compilers / stale_rmsf_outputs compare it with this
#   file (K1), so results on the old file are refused.
from ensembits.paths import MISATO_SPLITS  # noqa: E402


def current_splits_file(dataset: str) -> Path:
    """The split file a dataset's probes read now (K1, 2026-10-17)."""
    dataset = canonical_dataset(dataset)
    return MISATO_SPLITS if dataset == 'misato' else Path(FLAT_DATASETS[dataset]['splits'])


def current_splits_sha256(dataset: str):
    """sha256 of the current split file of `dataset`, or None if it does not exist (K1)."""
    p = current_splits_file(dataset)
    return splits_record(p)['splits_sha256'] if p.exists() else None


def splits_record(path) -> dict:
    """{splits_file, splits_sha256} of the split file a probe used (G4, 2026-10-17)."""
    from ensembits.featurize import file_sha256
    return {'splits_file': _rel(path), 'splits_sha256': file_sha256(path)}


def check_one_split(groups, splits: dict, names=None) -> None:
    """Every protein (group index) lies in exactly one of train / val / test (G4).

    groups: protein index per row; splits: {split: row indices}. ResidueCoverageError
    when a protein has rows in two splits.
    """
    seen = {}
    for k in ('train', 'val', 'test'):
        for g in np.unique(np.asarray(groups)[np.asarray(splits[k], dtype=np.int64)]):
            if g in seen and seen[g] != k:
                nm = names[g] if names is not None else g
                raise ResidueCoverageError(f'protein {nm} has residues in {seen[g]} and {k}')
            seen[g] = k


def label_content_sha256(labels) -> str:
    """Content hash of an RMSF label: hash_arrays_by_pid of {pid: array} (MISATO) or of
    {"rmsf": flat array} (mdCATH-div, BioEmu). The label builders record the same value as
    `labels_sha256` in their sidecars."""
    from ensembits.data_versions import hash_arrays_by_pid
    if isinstance(labels, dict):
        return hash_arrays_by_pid(labels)
    return hash_arrays_by_pid({'rmsf': np.asarray(labels)})


def check_label_hash(path, labels) -> str:
    """Content hash of a loaded label; CacheError if its sidecar records another one.

    FIXED(preretrain_audit_r2 F5, 2026-10-16): new. The result JSONs record this hash
    (`rmsf_label_sha256`) next to the label name, and the compilers require one hash per
    table.
    """
    from ensembits.data_versions import read_provenance
    h = label_content_sha256(labels)
    side = read_provenance(path) or {}
    rec = side.get('labels_sha256')
    if rec is not None and rec != h:
        raise CacheError(f'{Path(path).name}: content hash {h[:12]} != sidecar {rec[:12]}')
    return h


def current_residue_set_sha256(dataset: str):
    """sha256 of the residue-set file of `dataset`, or None if it has not been built."""
    p = RESIDUE_SET_FILES[dataset]
    return load_residue_set(p)['sha256'] if p.exists() else None


def stale_rmsf_outputs(paths, dataset: str, expect_label=RMSF_LABEL_DEFAULT) -> list:
    """Existing RMSF result JSONs that would be reused but do not carry the current common
    residue set (or none) or the expected label: [(path, reason)]. For the battery's skip
    logic (it skips a job whose output exists, so the probe's own check never runs)."""
    cur = current_residue_set_sha256(dataset)
    cur_splits = current_splits_sha256(dataset)          # K1 (2026-10-17)
    lab_path = RMSF_LABEL_FILES[dataset].get(expect_label)
    cur_label = None
    if any(Path(o).exists() for o in paths) and lab_path is not None and lab_path.exists():
        arr = np.load(lab_path)
        cur_label = label_content_sha256({k: arr[k] for k in arr.files}
                                         if lab_path.suffix == '.npz' else arr)
    bad = []
    for o in paths:
        o = Path(o)
        if not o.exists():
            continue
        try:
            js = json.loads(o.read_text())
        except Exception as e:   # unreadable JSON
            bad.append((o, f'unreadable ({type(e).__name__})')); continue
        h = js.get('residue_set_sha256')
        lab = normalize_label_name(dataset, js.get('rmsf_label'))
        if h is None:
            bad.append((o, 'no common residue set recorded'))
        elif cur is not None and h != cur:
            bad.append((o, f'residue set {h[:12]} != current {cur[:12]}'))
        elif lab != expect_label:
            bad.append((o, f'rmsf_label {lab!r} != {expect_label!r}'))
        # FIXED(preretrain_audit_r2 F5 / F1, 2026-10-16): also the label contents and the
        #   input caches recorded in the result.
        elif cur_label is not None and js.get('rmsf_label_sha256') != cur_label:
            bad.append((o, 'label contents differ from the current label file'))
        elif records_changed(js.get('feature_caches')):
            bad.append((o, '; '.join(records_changed(js.get('feature_caches')))))
        elif cur_splits is not None and js.get('splits_sha256') != cur_splits:
            # FIXED(preretrain_audit_r2 K1, 2026-10-17): compared with the CURRENT split file
            bad.append((o, 'made with another split file than the current one'))
        elif js.get('cache_verified') is False:          # G11 (2026-10-17)
            bad.append((o, 'feature caches were not verified (--allow-unverified-cache)'))
        elif vote_rule_problem(js):                       # 2026-10-19
            bad.append((o, vote_rule_problem(js)))
    return bad


# ---------------------------------------------------------------- baseline-cache provenance
# FIXED(preretrain_audit_r2 F1, 2026-10-16): old = the probes accepted a baseline cache by
#   file name and lengths only (a pre-fix poly-Ala ProToken cache, His-as-UNK ProtProfile,
#   MISATO caches from the old parser all passed). New = every builder used by the RMSF
#   probes writes `<cache>.provenance.json` (write_cache_provenance) and the probes refuse
#   a cache without a current sidecar (check_cache_provenance).
BASELINE_CACHE_SCHEMA = 'baseline-cache-2026-10-16'
CHAIN_BREAK_A = 2.5           # legacy chain split of the baseline builders (code_fix P5)
# FIXED(baseline fidelity audit, 2026-10-18): authors' decision: the baselines are cut only
#   at true chain boundaries, as Ensembits (whose descriptor keeps the protein whole and
#   only zeroes dihedrals across C-N > 2.5 A), never at missing-residue gaps. Old = every
#   builder split at C-N > 2.5 A (code_fix P5), so a MISATO chain with a gap became two
#   pieces. New (option A, 2026-10-18) = MISATO pieces are the true chains of
#   misato_chain_breaks.json (data_prep/build_misato_chain_breaks.py: a MISATO chain_index
#   is a MOLECULE that can hold several chains; cut at boundary / unresolved breaks, kept
#   whole with gap-aware residue numbering across missing_gap / consecutive breaks;
#   data_prep/baselines/_chain_pieces.py), recorded as "true_chain_v2" (v2 table, same day:
#   chain_index changes placed too, SEQRES step, physical check; v1 caches refused) plus the table's
#   sha256 (chain_breaks_record). mdCATH-div domains and BioEmu-CATH systems are single
#   chains and are encoded whole. Every baseline cache sidecar records `chain_split`
#   (replacing chain_break_A); check_cache_provenance requires the value of its dataset
#   (MISATO: and the sha256 of the current chain-break table). Results already written are
#   judged by the caches they recorded (records_changed), so nothing is refused retroactively.
CHAIN_SPLITS = {'misato': 'true_chain_v2',
                'mdcath_div': 'whole protein (single chain)',
                'bioemu': 'whole protein (single chain)'}
# Single-chain datasets whose legacy sidecars (chain_break_A = 2.5, no chain_split) are
#   still accepted: a scan of every frame of the real backbones (2026-10-18) found no
#   C-N > 2.5 A in the 2,442 mdcath_real_bb domains (bb_10) or the 1,090
#   bioemu_cath_real_bb systems (bb_10), so the 2.5 A split was a no-op there and the old
#   caches equal whole-protein encoding. MISATO (994 of 16,972 complexes with an internal
#   gap inside a chain) is not in this set: its legacy caches are refused.
LEGACY_SPLIT_NOOP = {'mdcath_div', 'bioemu'}


def chain_split_for(dataset: str) -> str:
    """Chain split a baseline cache of `dataset` must record (CHAIN_SPLITS, 2026-10-18)."""
    return CHAIN_SPLITS[canonical_dataset(dataset)]


def split_record(dataset: str) -> dict:
    """Sidecar fields of the chain split of a baseline cache (2026-10-18): chain_split and,
    for MISATO, the chain-break table it was cut with (chain_breaks_record)."""
    ds = canonical_dataset(dataset)
    return {'chain_split': CHAIN_SPLITS[ds],
            'chain_breaks': chain_breaks_record() if ds == 'misato' else None}


def chain_breaks_record(path=None) -> dict:
    """Identity of the MISATO chain-break table for a cache sidecar (2026-10-18, option A):
    file, path, sha256, schema and numbering rule (from its provenance sidecar)."""
    from ensembits.data_versions import read_provenance
    from ensembits.featurize import file_sha256
    p = Path(path or MISATO_CHAIN_BREAKS)
    if not p.exists():
        raise CacheError(f'{p.name} missing: run data_prep/build_misato_chain_breaks.py')
    side = read_provenance(p) or {}
    return {'file': p.name, 'path': _rel(p), 'sha256': file_sha256(p),
            'schema': side.get('schema'), 'numbering': side.get('numbering')}

# Sequence sources a builder may record (anything else is refused by the probes).
SEQUENCE_SOURCES = {'mdcath_div': 'mdcath_real_bb seq, X -> H',
                    'misato': 'misato_real_bb seq (fixed parser)',
                    'bioemu': 'bioemu_cath_extras seq (force-field names normalised)'}


def canonical_dataset(name: str) -> str:
    """Builder dataset names ("mdcath") -> the RMSF dataset names ("mdcath_div")."""
    return {'mdcath': 'mdcath_div'}.get(name, name)


_MISATO_SCAN: dict = {}


def misato_real_bb_scan(directory=None) -> dict:
    """parser_version / K / fps of EVERY misato_real_bb file (cached per run and folder).

    FIXED(preretrain_audit_r2 G8, 2026-10-17): old = a sample of ~50 files; new = every file
    is read (three scalar keys), and the folder's _status.json (written by a full
    build_misato_real_bb.py run) must agree when present.
    """
    d = Path(directory or MISATO_REAL_BB_DIR)
    key = str(d.resolve())
    if key not in _MISATO_SCAN:
        files = sorted(d.glob('*.npz'))
        vers, Ks, fps = set(), set(), set()
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                vers.add(str(z['parser_version']) if 'parser_version' in z.files else None)
                Ks.add(int(z['K']) if 'K' in z.files else None)
                fps.add(str(z['fps']) if 'fps' in z.files else None)
        rec = {'real_bb_dir': str(d), 'n_files': len(files), 'scan': 'all files',
               'parser_version': sorted(vers, key=str), 'K': sorted(Ks, key=str),
               'fps': sorted(fps, key=str)}
        st = d / '_status.json'
        if st.exists():
            sj = json.loads(st.read_text())
            rec['status_json'] = {k: sj.get(k) for k in ('parser_version', 'K', 'fps', 'n_pids')}
            if ([sj.get('parser_version')], [sj.get('K')], [sj.get('fps')]) != \
                    (rec['parser_version'], rec['K'], rec['fps']):
                rec['status_json_disagrees'] = True
        _MISATO_SCAN[key] = rec
    return dict(_MISATO_SCAN[key])


def dataset_input_record(dataset: str) -> dict:
    """Inputs of a baseline builder, for its sidecar: real-backbone folder and, for MISATO,
    the parser versions / K / FPS rule of all its files (misato_real_bb_scan)."""
    dataset = canonical_dataset(dataset)
    if dataset == 'misato':
        return misato_real_bb_scan()
    if dataset == 'mdcath_div':
        man = Path(MDCATH_DIV_SRC) / '_manifest.json'
        from ensembits.featurize import file_sha256
        return {'real_bb_dir': str(MDCATH_REAL_BB_DIR), 'mdcath_div_src': str(MDCATH_DIV_SRC),
                'manifest_sha256': file_sha256(man) if man.exists() else None}
    if dataset == 'bioemu':
        from ensembits.data_versions import read_provenance
        from ensembits.featurize import file_sha256
        ex = read_provenance(BIOEMU_EXTRAS_DIR) or {}
        return {'real_bb_dir': str(BIOEMU_REAL_BB_DIR),
                'layout_sha256': file_sha256(BIOEMU_LAYOUT) if BIOEMU_LAYOUT.exists() else None,
                'extras_dir': str(BIOEMU_EXTRAS_DIR), 'extras_n_ok': ex.get('n_ok'),
                'extras_mismatch': len(ex.get('mismatch') or {})}
    raise ValueError(dataset)


def write_cache_provenance(cache_path, *, dataset: str, builder: str, inputs: dict,
                           n_proteins: int, frames=None, chain_break_A=None,
                           sequence_source=None, failed=None, codebook=None,
                           chain_split=None, **extra) -> Path:
    """Write `<cache>.provenance.json` for a baseline cache (call after the cache is written).

    H4 (2026-10-16): builders pass sequence_source (SEQUENCE_SOURCES[dataset], or
    SEQ_NOT_USED) and frames; the probes require them. Since 2026-10-18 they pass
    chain_split (chain_split_for(dataset)) instead of chain_break_A (kept for old callers).
    """
    from ensembits.data_versions import write_provenance
    from ensembits.featurize import file_sha256
    import time as _time
    info = {'schema': BASELINE_CACHE_SCHEMA, 'dataset': canonical_dataset(dataset),
            'builder': builder, 'inputs': inputs, 'n_proteins': int(n_proteins),
            'frames': frames, 'chain_split': chain_split, 'chain_break_A': chain_break_A,
            'sequence_source': sequence_source, 'failed': failed or {},
            'cache_sha256': file_sha256(cache_path),
            'codebook_sha256': (file_sha256(codebook) if codebook is not None
                                and Path(codebook).exists() else None),
            'codebook_file': Path(codebook).name if codebook is not None else None,
            'created': _time.strftime('%Y-%m-%d %H:%M:%S'), **extra}
    return write_provenance(cache_path, info)


# Sequence source recorded by builders that do not use the sequence (H4, 2026-10-16).
SEQ_NOT_USED = 'not used (structure only)'
# MISATO inputs every MISATO baseline / token cache must come from (K = 8 real backbones,
# Kabsch FPS; build_misato_real_bb.py --K 8).
MISATO_K, MISATO_FPS = 8, 'kabsch'


def check_misato_inputs(inputs: dict, what: str) -> None:
    """CacheError unless the recorded MISATO inputs are the current parser, K = 8, Kabsch."""
    from ensembits.data_versions import MISATO_PARSER_VERSION
    inputs = inputs or {}
    if inputs.get('parser_version') != [MISATO_PARSER_VERSION]:
        raise CacheError(f'{what}: MISATO inputs from parser {inputs.get("parser_version")!r}, '
                         f'expected [{MISATO_PARSER_VERSION!r}]; rebuild it')
    if inputs.get('K') != [MISATO_K] or inputs.get('fps') != [MISATO_FPS]:
        raise CacheError(f'{what}: MISATO inputs K={inputs.get("K")!r} fps={inputs.get("fps")!r}, '
                         f'expected [{MISATO_K}] / [{MISATO_FPS!r}]')
    # G8 (2026-10-17): the inputs must come from a scan of every file, consistent with the
    #   folder's _status.json.
    if inputs.get('scan') != 'all files' or inputs.get('status_json_disagrees'):
        raise CacheError(f'{what}: MISATO inputs were sampled, or disagree with _status.json; '
                         f'rebuild the cache')


def check_cache_provenance(cache_path, dataset: str, *, frames: str, uses_sequence: bool = False,
                           codebook=None, required=None) -> dict:
    """Refuse a baseline cache without a current sidecar; -> record for the result JSON.

    Checks: sidecar exists with the current schema and the dataset; the cache file's sha256
    equals the recorded one (no change after the sidecar); MISATO inputs come from the
    current parser with K = 8 and Kabsch FPS; the chain split (required) is the dataset's
    CHAIN_SPLITS value (2026-10-18; legacy 2.5 Å sidecars only for LEGACY_SPLIT_NOOP); the
    sequence source (required) is the current one when the feature uses the sequence,
    SEQ_NOT_USED otherwise; the frames used (required) are `frames` (e.g. "frame 0",
    "all 8", "first 10 of bb_10"); the codebook (token caches) is the one the tokens were
    made with.
    FIXED(preretrain_audit_r2 H4, 2026-10-16): old = chain split and sequence source were
    only checked when recorded; MISATO K / fps and the frames were not checked.
    """
    from ensembits.data_versions import read_provenance
    from ensembits.featurize import file_sha256
    cache_path = Path(cache_path)
    dataset = canonical_dataset(dataset)
    if not cache_path.exists():
        raise CacheError(f'{cache_path.name} missing')
    side = read_provenance(cache_path)
    if side is None:
        raise CacheError(f'{cache_path.name} has no provenance sidecar (built before 2026-10-16 '
                         f'or outside the codebase); rebuild it with data_prep/baselines/')
    if side.get('schema') != BASELINE_CACHE_SCHEMA:
        raise CacheError(f'{cache_path.name}: sidecar schema {side.get("schema")!r} != '
                         f'{BASELINE_CACHE_SCHEMA!r}; rebuild it')
    if side.get('dataset') != dataset:
        raise CacheError(f'{cache_path.name}: built for {side.get("dataset")!r}, not {dataset!r}')
    sha = file_sha256(cache_path)
    if sha != side.get('cache_sha256'):
        raise CacheError(f'{cache_path.name} changed after its sidecar was written')
    if dataset == 'misato':
        check_misato_inputs(side.get('inputs'), cache_path.name)
    # FIXED(baseline fidelity audit, 2026-10-18): old = chain_break_A must be 2.5 Å; new =
    #   chain_split must be the dataset's CHAIN_SPLITS value (MISATO: 'true_chain_v2', made
    #   from the current misato_chain_breaks.json). A legacy 2.5 Å sidecar is refused (for
    #   new probe runs), except for the single-chain datasets of LEGACY_SPLIT_NOOP, where
    #   that split cut nothing (scan of 2026-10-18).
    want_cs = CHAIN_SPLITS[dataset]
    cs, cb = side.get('chain_split'), side.get('chain_break_A')
    legacy_split = (cs is None and dataset in LEGACY_SPLIT_NOOP and cb is not None
                    and abs(float(cb) - CHAIN_BREAK_A) <= 1e-9)
    if cs != want_cs and not legacy_split:
        raise CacheError(f'{cache_path.name}: chain split {cs!r} (chain_break_A {cb!r}), expected '
                         f'{want_cs!r}; rebuild it with data_prep/baselines/ (README: move the old '
                         f'cache and its result JSONs aside first)')
    cb_rec = None
    if dataset == 'misato':
        cb_rec = chain_breaks_record()
        made = (side.get('chain_breaks') or {}).get('sha256')
        if made != cb_rec['sha256']:
            raise CacheError(f'{cache_path.name} was made from another misato_chain_breaks.json '
                             f'(sha256 {str(made)[:12]}, current {cb_rec["sha256"][:12]}); rebuild it')
    want_ss = SEQUENCE_SOURCES[dataset] if uses_sequence else SEQ_NOT_USED
    if side.get('sequence_source') != want_ss:
        raise CacheError(f'{cache_path.name}: sequence source {side.get("sequence_source")!r}, '
                         f'expected {want_ss!r}')
    if side.get('frames') != frames:
        raise CacheError(f'{cache_path.name}: frames {side.get("frames")!r}, this feature needs '
                         f'{frames!r}')
    # FIXED(preretrain_audit_r2 G7, 2026-10-17): new: no protein of the residue set
    #   (`required`) may come from an ideal-geometry frame (sidecar n_ideal_frames /
    #   ideal_pids; an unknown list counts as a hit).
    if int(side.get('n_ideal_frames') or 0) > 0:
        ideal = set(side.get('ideal_pids') or [])
        if not ideal or required is None or ideal & set(required):
            raise CacheError(f'{cache_path.name}: {side.get("n_ideal_frames")} ideal-geometry frames '
                             f'(proteins {sorted(ideal & set(required or ideal))[:5]}) inside the '
                             f'RMSF residue set; rebuild without --allow-ideal')
    rec = {'file': cache_path.name, 'path': _rel(cache_path), 'sha256': sha,
           'builder': side.get('builder'), 'frames': frames, 'chain_split': want_cs}
    if legacy_split:
        rec['chain_split_legacy_sidecar'] = 'chain_break_A 2.5 (no gap in this dataset)'
    if cb_rec is not None:
        rec['chain_breaks_sha256'] = cb_rec['sha256']
    if codebook is not None:
        cs = file_sha256(codebook)
        if side.get('codebook_sha256') != cs:
            raise CacheError(f'{cache_path.name} was made with another codebook than '
                             f'{Path(codebook).name}')
        rec['codebook_sha256'] = cs
        rec['codebook_path'] = _rel(codebook)          # G11: re-checked by records_changed
    return rec


def _rel(path) -> str:
    """Path relative to REPO_ROOT when inside it (recorded in result JSONs)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def file_record(path) -> dict:
    """Identity of an input file without a baseline sidecar (full sha256)."""
    from ensembits.featurize import file_sha256
    return {'file': Path(path).name, 'path': _rel(path), 'sha256': file_sha256(path)}


def large_file_record(path) -> dict:
    """Cheap identity of a large cache (size, mtime, sha256 of its sidecar if any)."""
    from ensembits.featurize import file_sha256
    p = Path(path)
    st = p.stat()
    side = Path(str(p) + '.provenance.json')
    return {'file': p.name, 'path': _rel(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'sidecar_sha256': file_sha256(side) if side.exists() else None}


def records_changed(records) -> list:
    """Reasons why recorded input files (file_record / large_file_record /
    check_cache_provenance records) differ from the files on disk now.

    FIXED(preretrain_audit_r2 G11, 2026-10-17): also the codebook of token caches
    (codebook_path / codebook_sha256) and the path-less MISATO sequence record (kind
    "misato_real_bb_seq": the parser versions of misato_real_bb are re-scanned); any other
    record without a path cannot be re-checked and counts as changed.
    """
    from ensembits.featurize import file_sha256
    out = []
    for r in records or []:
        if 'codebook_path' in r:
            cp = Path(r['codebook_path']) if Path(r['codebook_path']).is_absolute() else ROOT / r['codebook_path']
            if not cp.exists() or file_sha256(cp) != r.get('codebook_sha256'):
                out.append(f'{cp.name} (codebook of {r.get("file")}) changed')
        if 'path' not in r:
            if r.get('kind') == 'misato_real_bb_seq':
                if misato_real_bb_scan().get('parser_version') != r.get('parser_version'):
                    out.append('misato_real_bb parser versions changed')
            else:
                out.append(f'{r.get("file")}: record cannot be re-checked (no path)')
            continue
        p = Path(r['path']) if Path(r['path']).is_absolute() else ROOT / r['path']
        if not p.exists():
            out.append(f'{r["file"]} missing'); continue
        if 'sha256' in r and file_sha256(p) != r['sha256']:
            out.append(f'{r["file"]} changed (sha256)')
        if 'size' in r:
            st = p.stat()
            if (st.st_size, st.st_mtime_ns) != (r['size'], r['mtime_ns']):
                out.append(f'{r["file"]} changed (size / mtime)')
    return out


# ---------------------------------------------------------------- builder tolerances
def rmsf_set_proteins(dataset: str):
    """Proteins of the common RMSF residue set of `dataset`, or None if not built yet."""
    p = RESIDUE_SET_FILES[canonical_dataset(dataset)]
    return set(load_residue_set(p)['proteins']) if p.exists() else None


def check_builder_failures(failed: dict, attempted, dataset: str,
                           max_fail_frac: float = 0.0, only=None) -> bool:
    """Report the proteins a baseline builder dropped or got wrong; True = within tolerance.

    FIXED(preretrain_audit_r2 F7, 2026-10-16): old = builders skipped failing proteins and
    still exited 0 (the probes then refused the cache, or, before the common set, silently
    scored fewer residues). New = failures of proteins in the RMSF residue set (all attempted
    proteins if the set is not built yet), plus set proteins without any input, are listed;
    their fraction of the set must not exceed `max_fail_frac` (default 0: none allowed).
    """
    req = rmsf_set_proteins(dataset)
    if req is None:
        print(f'[builder] no RMSF residue set for {canonical_dataset(dataset)} yet: every '
              f'attempted protein must succeed', flush=True)
        req = set(attempted)
    if only is not None:            # one shard: only its own proteins
        req = req & set(only)
    bad = {p: w for p, w in failed.items() if p in req}
    for p in sorted(req - set(attempted)):
        bad.setdefault(p, 'no input (not attempted)')
    frac = len(bad) / max(len(req), 1)
    outside = len(set(failed) - req)
    print(f'[builder] {len(bad)} of {len(req)} proteins of the RMSF set failed ({frac:.2%}; '
          f'allowed {max_fail_frac:.2%}); {outside} failures outside the set', flush=True)
    for p in sorted(bad)[:50]:
        print(f'  FAILED {p}: {bad[p]}', flush=True)
    return frac <= max_fail_frac


def finish_cache(out_path, out: dict, failed: dict, attempted, dataset: str, *, builder: str,
                 max_fail_frac: float = 0.0, sharded: bool = False, **prov) -> bool:
    """Tolerance check (F7), then write the npz and its provenance sidecar (F1).

    Returns False (nothing written) when too many proteins of the RMSF set failed. A shard
    is checked on its own proteins and records them (`attempted`) for the merge.
    """
    if not check_builder_failures(failed, attempted, dataset, max_fail_frac,
                                  only=set(attempted) if sharded else None):
        print(f'[error] failures inside the RMSF residue set; {Path(out_path).name} not written',
              flush=True)
        return False
    np.savez(out_path, **out)
    write_cache_provenance(out_path, dataset=dataset, builder=builder,
                           inputs=dataset_input_record(dataset), n_proteins=len(out),
                           failed=failed, sharded=sharded,
                           attempted=sorted(attempted) if sharded else None, **prov)
    print(f'wrote {Path(out_path).name} (+ .provenance.json)', flush=True)
    return True


def snapshot_info(variant_dir):
    """snapshot.json of a downstream/snapshot_checkpoint.py folder, or None (F9)."""
    p = Path(variant_dir) / 'snapshot.json'
    return json.loads(p.read_text()) if p.exists() else None


def check_bioemu_split_layout(splits_path=None) -> None:
    """The BioEmu split must have been built from the current layout_rows.json (G4).

    FIXED(preretrain_audit_r2 G4, 2026-10-17): new. The split builder records
    `layout_sha256` in its sidecar; a split built from another layout (other row order or
    lengths) is refused.
    """
    from ensembits.data_versions import read_provenance
    from ensembits.featurize import file_sha256
    sp = Path(splits_path or BIOEMU_SPLITS)
    prov = read_provenance(sp)
    if prov is None or prov.get('layout_sha256') is None:
        raise CacheError(f'{sp.name}: no provenance sidecar with layout_sha256; rebuild it with '
                         f'data_prep/build_bioemu_rmsf_splits.py')
    if prov['layout_sha256'] != file_sha256(BIOEMU_LAYOUT):
        raise CacheError(f'{sp.name} was built from another layout_rows.json; rebuild it')


def check_desc_sidecar_rows(side: dict, dataset: str, n_rows: int, rs=None) -> None:
    """Descriptor-cache sidecar vs the dataset, the label's rows and the residue set (G1).

    FIXED(preretrain_audit_r2 G1, 2026-10-17): new. ValueError unless the sidecar names
    this dataset (builder name "mdcath" = mdcath_div), records n_rows equal to the label's,
    and none of its ideal-geometry proteins (IDEAL_PIDS) is in the residue set.
    """
    if canonical_dataset(side.get('dataset') or '') != canonical_dataset(dataset):
        raise ValueError(f'descriptor cache built for {side.get("dataset")!r}, not {dataset!r}')
    if side.get('n_rows') != n_rows:
        raise ValueError(f'descriptor cache sidecar records {side.get("n_rows")} rows, the label '
                         f'has {n_rows}')
    if rs is not None:
        bad = set(side.get('IDEAL_PIDS') or []) & set(rs['proteins'])
        if bad:
            raise ValueError(f'descriptor cache uses ideal geometry for {len(bad)} proteins of '
                             f'the residue set (e.g. {sorted(bad)[:3]})')


def check_flat_label_sidecar(path, dataset: str, label_name: str, n_rows: int) -> None:
    """Label sidecar of a flat dataset (mdCATH-div, BioEmu): required for the all-frames label,
    and it must name the label and the row count (G1, 2026-10-17). ValueError otherwise.
    Older labels (converged5, legacy) without a sidecar only print a warning."""
    from ensembits.data_versions import read_provenance
    side = read_provenance(path)
    if side is None:
        if label_name == RMSF_LABEL_DEFAULT:
            raise ValueError(f'{Path(path).name} has no provenance sidecar; rebuild it with '
                             f'data_prep/build_{"bioemu" if dataset == "bioemu" else "mdcath_div"}'
                             f'_rmsf_labels.py')
        print(f'[WARNING] {Path(path).name}: no provenance sidecar (label {label_name!r}, '
              f'reproduction only)', flush=True)
        return
    if side.get('label_name', label_name) != label_name:
        raise ValueError(f'{Path(path).name} is label {side.get("label_name")!r}, not {label_name!r}')
    if side.get('n_rows') is not None and int(side['n_rows']) != int(n_rows):
        raise ValueError(f'{Path(path).name}: sidecar n_rows {side["n_rows"]} != {n_rows}')
    if label_name == RMSF_LABEL_DEFAULT and side.get('label_name') is None:
        raise ValueError(f'{Path(path).name}: sidecar names no label; rebuild it')


def write_codebook(path, arr) -> bool:
    """Write a baseline codebook atomically; keep an identical existing file (returns False).

    FIXED(preretrain_audit_r2 G12, 2026-10-17): old = np.save over the shared codebook on
    every builder call (a later single-dataset call changed the file's mtime / could be read
    half-written, and the token caches' codebook_sha256 records depend on it). New = an
    identical codebook is left untouched; otherwise tmp file + os.replace.
    """
    import os
    path = Path(path)
    arr = np.asarray(arr)
    if path.exists():
        old = np.load(path)
        if old.dtype == arr.dtype and old.shape == arr.shape and np.array_equal(old, arr):
            return False
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    with open(tmp, 'wb') as fh:
        np.save(fh, arr)
    os.replace(tmp, path)
    return True
