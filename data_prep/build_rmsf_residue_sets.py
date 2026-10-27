"""Common residue sets of the RMSF probes (one per dataset, chosen before any method runs).

What it does
------------
Writes the residue set on which EVERY method (Ensembits P=full / P=1 and every
baseline) is trained, validated and tested in the RMSF probes:

  * `--dataset mdcath_div`: every residue of the mdCATH-div domains that have real
    backbone atoms, i.e. a file `data/cached_descriptors/mdcath_real_bb/<pid>.npz`.
    Since the re-extraction (data_prep/build_mdcath_real_bb.py, 2026-10-17) all 2,442
    representatives have one, so the set is every residue (308,293). A domain without
    a file is an error (exit 1) unless `--allow-missing`, which excludes it for every
    method, prints it and records `allow_missing` + the domains in the sidecar
    (2026-10-16: 8 domains, 1,326 residues, before the re-extraction). Rows follow the `_manifest.json`
    "representatives" order with each domain's `ca_5` length (the row order of `rmsf_div*.npy` and of every mdCATH-div
    cache); each kept domain's real-backbone `seq` must have that length.
  * `--dataset misato`: every residue of the complexes that are in
    `data/misato_splits_v2.json` (union of the sequence / structure / random splits),
    have an RMSF label in `misato_rmsf_allframes.npz` (--misato-rmsf) and a K = 8
    real-backbone file
    `misato_real_bb/<PID>.npz`; the label length must equal the backbone length.

  * `--dataset bioemu` (phase 2, 2026-10-16): every residue of the 1,090 systems of
    `bioemu_cath_real_bb/layout_rows.json` (layout order; each bb_10 must have the
    layout length).

The choice depends on the data only (which files exist), never on a method's
features or results. Output = JSON {dataset, rule, proteins: {pid: {row_start,
length}}, excluded: {pid: reason}, sha256} plus the provenance sidecar
`<out>.provenance.json` (inputs, their hashes, counts). The probes record the
set's sha256 in every result JSON; the compilers refuse tables whose rows carry
different sets.

Paper
-----
`2_results.tex` \\label{result:rmsf} and `appendix.tex` \\label{app:baselines}
("for every method"): every method is scored on the same held-out residues.

Origin
------
New (2026-10-16, preretrain_audit_r2 Q3 / Q29c, user decision Q3; same principle
as the ANOVA fix P6). Provenance sidecar as in `build_mdcath_div_rmsf_labels.py`.

Used by the reported model: yes (the residue set of every RMSF probe).

Inputs / outputs
----------------
  reads   data/external/mdcath_div/{_manifest.json,<pid>.npz}   (ENSEMBITS_MDCATH_DIV_SRC)
          data/cached_descriptors/mdcath_real_bb/<pid>.npz      (seq)
          data/misato_splits_v2.json, data/cached_descriptors/misato_rmsf_allframes.npz (+ sidecar)
          data/cached_descriptors/misato_real_bb/<PID>.npz      (bb_8, parser_version)
          output/phase_b/cached_desc/rmsf_div_allframes.npy     (row count check, if present)
          data/cached_descriptors/bioemu_cath_real_bb/{layout_rows.json,<system>.npz}   (bioemu)
  writes  data/cached_descriptors/rmsf_residue_set_mdcath_div.json (+ .provenance.json)
          data/cached_descriptors/rmsf_residue_set_misato.json     (+ .provenance.json)
          data/cached_descriptors/rmsf_residue_set_bioemu.json     (+ .provenance.json)
          refuses to overwrite unless --overwrite

How to run
----------
Before the RMSF probes (after the MISATO labels are rebuilt):
    python data_prep/build_rmsf_residue_sets.py --dataset mdcath_div
    python data_prep/build_rmsf_residue_sets.py --dataset misato
    python data_prep/build_rmsf_residue_sets.py --dataset bioemu     (phase 2)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))   # rmsf_common (shared with the probes)
from ensembits.paths import REPO_ROOT as ROOT, MDCATH_DIV_SRC  # noqa: E402
from ensembits.data_versions import (hash_arrays_by_pid, read_provenance,  # noqa: E402
                                     write_provenance)
from ensembits.featurize import file_sha256  # noqa: E402
import rmsf_common as RC  # noqa: E402

# FIXED(benchmark_audit D3, 2026-10-18): default split file = ensembits.paths.MISATO_SPLITS
#   (data/misato_splits_v2.json; same 4,894 complexes as the old misato_splits.json, so the
#   residue set is unchanged).
from ensembits.paths import MISATO_SPLITS  # noqa: E402
# Labels (user decision 2026-10-16: all MD frames). The MISATO set needs the label of
# every complex; the mdCATH-div label is only used for a row-count check.
MISATO_RMSF = RC.RMSF_LABEL_FILES['misato'][RC.RMSF_LABEL_DEFAULT]
MDCATH_DIV_LABEL = RC.RMSF_LABEL_FILES['mdcath_div'][RC.RMSF_LABEL_DEFAULT]

RULES = {
    # FIXED(mdCATH re-extraction N1, 2026-10-17): a representative without a real-backbone
    #   file is an error unless --allow-missing (was: silently excluded, so a missing
    #   re-extraction fell back to the 2,434-domain set).
    'mdcath_div': ('every residue of the mdCATH-div representatives (all 2,442); each must have '
                   'a real-backbone file (data/cached_descriptors/mdcath_real_bb, written by '
                   'data_prep/build_mdcath_real_bb.py); a missing file is an error unless '
                   '--allow-missing, which excludes the domain (recorded in the sidecar)'),
    'misato': ('every residue of the complexes in data/misato_splits.json (sequence, structure, '
               'random) that have an RMSF label and a misato_real_bb (K=8) file'),
    'bioemu': ('every residue of the BioEmu-CATH systems of bioemu_cath_real_bb/layout_rows.json '
               '(all 1,090, incl. the 18 held out of the tokenizer corpus)'),
}


def build_mdcath_div(src: Path, rb_dir: Path, label_path: Path | None = None,
                     allow_missing: bool = False):
    """-> (residue set dict, provenance dict). Raises ValueError on inconsistent inputs,
    and when a representative has no real-backbone file unless allow_missing."""
    pids, offsets, total = RC.mdcath_div_layout(src)
    proteins, excluded = {}, {}
    for pid in pids:
        s, e = offsets[pid]
        f = rb_dir / f'{pid}.npz'
        if not f.exists():
            # 2026-10-17: none after data_prep/build_mdcath_real_bb.py; an error unless
            #   allow_missing (checked after the loop, so every missing domain is listed)
            excluded[pid] = 'no real-backbone file (run data_prep/build_mdcath_real_bb.py)'
            continue
        with np.load(f, allow_pickle=False) as z:
            n = len(str(z['seq']))
        if n != e - s:
            raise ValueError(f'{pid}: mdcath_real_bb seq has {n} residues, ca_5 has {e - s}')
        proteins[pid] = {'row_start': int(s), 'length': int(e - s)}
    if excluded and not allow_missing:
        raise ValueError(f'{len(excluded)} of {len(pids)} mdCATH-div representatives have no '
                         f'real-backbone file in {rb_dir} ({sorted(excluded)[:10]}); run '
                         f'data_prep/build_mdcath_real_bb.py --only-missing, or pass '
                         f'--allow-missing to exclude them for every method')
    rs = RC.make_residue_set('mdcath_div', proteins, RULES['mdcath_div'],
                             n_rows_total=total, excluded=excluded)
    prov = {'source': str(src), 'manifest_sha256': file_sha256(Path(src) / '_manifest.json'),
            'real_bb_dir': str(rb_dir), 'n_representatives': len(pids), 'n_rows_total': total,
            'row_order': '_manifest.json representatives, ca_5 lengths',
            'allow_missing': bool(allow_missing), 'missing_real_bb': sorted(excluded)}
    if label_path is not None and Path(label_path).exists():
        n_lab = int(np.load(label_path, mmap_mode='r').shape[0])
        if n_lab != total:
            raise ValueError(f'{Path(label_path).name} has {n_lab} rows, the layout {total}')
        prov['label_file'] = str(label_path)
        prov['label_rows'] = n_lab
    return rs, prov


def build_bioemu(layout: Path, rb_dir: Path, label_path: Path | None = None):
    """-> (residue set dict, provenance dict). Every system of the layout; its bb_10 must
    have the layout length."""
    names, offsets, total = RC.bioemu_layout(layout)
    proteins = {}
    for s_ in names:
        st, en = offsets[s_]
        with np.load(rb_dir / f'{s_}.npz', allow_pickle=False) as z:
            L = int(z['bb_10'].shape[1])
        if L != en - st:
            raise ValueError(f'{s_}: bb_10 has {L} residues, layout_rows.json {en - st}')
        proteins[s_] = {'row_start': int(st), 'length': int(en - st)}
    rs = RC.make_residue_set('bioemu', proteins, RULES['bioemu'], n_rows_total=total)
    prov = {'layout': str(layout), 'layout_sha256': file_sha256(layout), 'real_bb_dir': str(rb_dir),
            'n_rows_total': total, 'row_order': 'layout_rows.json'}
    if label_path is not None and Path(label_path).exists():
        n_lab = int(np.load(label_path, mmap_mode='r').shape[0])
        if n_lab != total:
            raise ValueError(f'{Path(label_path).name} has {n_lab} rows, the layout {total}')
        prov['label_file'] = str(label_path); prov['label_rows'] = n_lab
    return rs, prov


def build_misato(splits_path: Path, rmsf_path: Path, rb_dir: Path, allow_old_parser=False,
                 allow_label_failures=False):
    """-> (residue set dict, provenance dict). Raises ValueError on inconsistent inputs.

    FIXED(preretrain_audit_r2 K4, 2026-10-17): new: a label built with --max-fail-frac > 0
    (complexes dropped by the label builder) is refused unless allow_label_failures.
    """
    lp0 = read_provenance(rmsf_path) or {}
    if float(lp0.get('max_fail_frac') or 0) > 0 and not allow_label_failures:
        raise ValueError(f'{Path(rmsf_path).name} was built with max_fail_frac='
                         f'{lp0["max_fail_frac"]} ({lp0.get("n_failed")} complexes failed); pass '
                         f'--allow-label-failures to accept (they are excluded for every method)')
    from ensembits.data_versions import MISATO_PARSER_VERSION
    s = json.loads(Path(splits_path).read_text())
    pids = sorted({p.upper() for sp in RC.MISATO_SPLIT_NAMES
                   for k in ('train', 'val', 'test') for p in s[sp].get(k, [])})
    labels = np.load(rmsf_path)
    lab = set(labels.files)
    proteins, excluded, versions = {}, {}, set()
    for pid in pids:
        f = rb_dir / f'{pid}.npz'
        if pid not in lab:
            excluded[pid] = 'no RMSF label'
            continue
        if not f.exists():
            excluded[pid] = 'no misato_real_bb file'
            continue
        with np.load(f, allow_pickle=False) as z:
            L = int(z['bb_8'].shape[1])
            v = str(z['parser_version']) if 'parser_version' in z.files else None
        versions.add(v)
        if v != MISATO_PARSER_VERSION and not allow_old_parser:
            raise ValueError(f'{f.name} was written by MISATO parser {v!r}, expected '
                             f'{MISATO_PARSER_VERSION!r} (code_fix §4); rebuild misato_real_bb '
                             f'or pass --allow-old-parser')
        n_lab = int(labels[pid].shape[0])
        if n_lab != L:
            raise ValueError(f'{pid}: RMSF label has {n_lab} residues, misato_real_bb {L} '
                             f'(stale label? rebuild with build_misato_rmsf_labels.py)')
        proteins[pid] = {'row_start': None, 'length': L}
    rs = RC.make_residue_set('misato', proteins, RULES['misato'], excluded=excluded)
    lp = read_provenance(rmsf_path) or {}
    prov = {'splits_file': str(splits_path), 'splits_sha256': file_sha256(splits_path),
            'split_names': list(RC.MISATO_SPLIT_NAMES), 'n_split_pids': len(pids),
            'rmsf_file': str(rmsf_path),
            'label_max_fail_frac': lp.get('max_fail_frac'), 'label_n_failed': lp.get('n_failed'),
            'rmsf_labels_sha256': lp.get('labels_sha256') or hash_arrays_by_pid(
                {p: labels[p] for p in labels.files}),
            'rmsf_label_provenance_found': bool(lp),
            'real_bb_dir': str(rb_dir), 'parser_versions': sorted(versions, key=str)}
    return rs, prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=RC.DATASETS)
    ap.add_argument('--out', default=None, help='default: rmsf_common.RESIDUE_SET_FILES[dataset]')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--mdcath-src', default=str(MDCATH_DIV_SRC))
    ap.add_argument('--mdcath-real-bb', default=str(RC.MDCATH_REAL_BB_DIR))
    ap.add_argument('--mdcath-label', default=str(MDCATH_DIV_LABEL),
                    help='row-count check only (skipped if absent)')
    ap.add_argument('--misato-splits', default=str(MISATO_SPLITS))
    ap.add_argument('--misato-rmsf', default=str(MISATO_RMSF))
    ap.add_argument('--misato-real-bb', default=str(RC.MISATO_REAL_BB_DIR))
    ap.add_argument('--bioemu-layout', default=str(RC.BIOEMU_LAYOUT))
    ap.add_argument('--bioemu-real-bb', default=str(RC.BIOEMU_REAL_BB_DIR))
    ap.add_argument('--allow-label-failures', action='store_true',
                    help='misato: accept a label built with --max-fail-frac > 0 (K4)')
    ap.add_argument('--allow-missing', action='store_true',
                    help='mdcath_div: exclude representatives without a real-backbone file '
                         '(default: error); recorded in the sidecar')
    ap.add_argument('--allow-old-parser', action='store_true',
                    help='accept misato_real_bb files written before the code_fix §4 parser fix')
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else RC.RESIDUE_SET_FILES[args.dataset]
    if out.exists() and not args.overwrite:
        print(f'{out} exists; pass --overwrite to rebuild (every RMSF result made with the old '
              f'set must then be rerun)'); return 1
    try:
        if args.dataset == 'mdcath_div':
            rs, prov = build_mdcath_div(Path(args.mdcath_src), Path(args.mdcath_real_bb),
                                        Path(args.mdcath_label), args.allow_missing)
        elif args.dataset == 'bioemu':
            rs, prov = build_bioemu(Path(args.bioemu_layout), Path(args.bioemu_real_bb),
                                    RC.RMSF_LABEL_FILES['bioemu'][RC.RMSF_LABEL_DEFAULT])
        else:
            rs, prov = build_misato(Path(args.misato_splits), Path(args.misato_rmsf),
                                    Path(args.misato_real_bb), args.allow_old_parser,
                                    args.allow_label_failures)
    # FIXED(preretrain_audit_r2 F11, 2026-10-16): old = a missing input file raised a
    #   traceback; new = a clean [error] line and exit 1.
    except (ValueError, FileNotFoundError, KeyError, OSError) as e:
        print(f'[error] {type(e).__name__}: {e}'); return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rs, indent=1))
    # FIXED(preretrain_audit_r2 F7, 2026-10-16): the excluded proteins (method-independent,
    #   e.g. MISATO complexes whose all-frames label failed) are counted by reason, printed
    #   prominently and recorded, so the paper's counts come from this sidecar.
    from collections import Counter
    by_reason = dict(Counter(rs['excluded'].values()))
    bar = '=' * 78
    print(f'{bar}\n[{args.dataset}] EXCLUDED for every method: {len(rs["excluded"])} proteins '
          f'{by_reason}\n[{args.dataset}] common residue set: {rs["n_proteins"]} proteins, '
          f'{rs["n_residues"]} residues\n{bar}')
    write_provenance(out, {'builder': 'data_prep/build_rmsf_residue_sets.py',
                           'dataset': args.dataset, 'output': str(out), 'rule': rs['rule'],
                           'sha256': rs['sha256'], 'n_proteins': rs['n_proteins'],
                           'n_residues': rs['n_residues'], 'n_excluded': len(rs['excluded']),
                           'n_excluded_by_reason': by_reason,
                           'created': time.strftime('%Y-%m-%d %H:%M:%S'), **prov})
    print(f'wrote {out}: {rs["n_proteins"]} proteins, {rs["n_residues"]} residues, '
          f'{len(rs["excluded"])} excluded, sha256 {rs["sha256"][:12]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
