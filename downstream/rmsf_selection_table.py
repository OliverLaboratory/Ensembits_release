"""Validation-only table for choosing among trained tokenizers (reconstruction loss + RMSF).

What it does
------------
For a list of trained variants (`output/<variant>/`, finished runs or frozen snapshots
of running ones made by `downstream/snapshot_checkpoint.py`), collects the numbers the
authors use to select among tokenizers (λ sweep, ablations), all on VALIDATION data:

  * best validation reconstruction loss: `best_val_recon` of `config.json` (the
    trainer's early-stopping criterion), cross-checked with history.json (the
    evaluation that wrote best.pt, trainer's rule: improvement > 1e-5); a mismatch is
    flagged (⚠). Run status: finished, or running / snapshot of <run> (flagged ⚠,
    numbers may still change). The loss settings that make the losses comparable
    (recon_loss_type, eval_p, n_val_residues) are compared and a warning is printed
    when they differ;
  * the RMSF probe, P = full and P = 1: pooled val Spearman at the selected epoch
    (`best_val_spearman`) and per-protein val Spearman (`val_spearman_per_protein_mean`),
    mean ± sample std (ddof = 1) over the seeds.

Test numbers are never read into the table. A cell whose seeds are all NaN because the
probe predicted a constant with every residue mapped to one token
(rmsf_common.constant_prediction_seed) is shown as "—†" with the footnote "every residue
maps to one token; Spearman undefined" and its seeds are not missing; any other NaN seed
is a problem (authors' decision 2026-10-18; was: dropped by the mean, K5). The script exits 1 without writing the
table (and lists every gap) unless, for every variant and every requested
(dataset, P) cell: all seeds of `--seeds` (default 0–9) exist; every JSON carries
the expected RMSF label (`--rmsf-label`, default allframes) with ONE label content
hash, ONE common residue set and ONE descriptor-cache identity across all variants;
the probe settings (max_epochs, patience, min_delta, batch_size) are identical
across all files; and the checkpoint block names the variant's current best.pt.

Datasets: mdCATH-div only by default. BioEmu-CATH (`--include-bioemu`) is shown
with a printed warning: its validation domains overlap the tokenizer's training
split (73 of 91 val domains for bioemu_cath_misato_e0; the number is read from the
BioEmu split sidecar when present), so it is not a clean selection signal.

No selection rule is hard-coded: the table only lists the numbers; the authors
decide how to weigh the reconstruction loss against the RMSF probe.

Paper
-----
Not described in the paper (model selection for `appendix.tex` \\label{app:ablations}
and the E0 / λ sweep runs; infrastructure only).

Origin
------
New (2026-10-16, user request: evaluation uses the validation loss AND the RMSF probe).
Reworked the same day (preretrain_audit_r2 F10): full seed sets, identical settings and
hashes across variants, run status / snapshots, history cross-check, BioEmu opt-in.

Used by the reported model: yes (choosing the checkpoint), no numbers of it enter the paper.

Inputs / outputs
----------------
Reads (relative to REPO_ROOT):
  output/<variant>/{config.json,history.json,best.pt[,snapshot.json]}
  output/probe_rmsf_div/<vtag>[_P1inf][_seed<s>]__<tag>.json       (mdcath_div)
  output/probe_rmsf_bioemu/<vtag>[_P1inf][_seed<s>]__<tag>.json    (--include-bioemu)
  output/phase_b/cached_desc/bioemu/splits.json.provenance.json     (overlap warning)
Writes: stdout; with --out, a markdown file.

How to run
----------
    python downstream/rmsf_selection_table.py --variants run_a,run_b,snap_run_c_ep120 \\
        [--seeds 0-9] [--tag test] [--rmsf-label allframes] [--include-bioemu] [--out table.md]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
import rmsf_common as RC   # noqa: E402

P_MODES = (('P=full', ''), ('P=1', '_P1inf'))
VAL_KEYS = (('pooled', 'best_val_spearman'), ('per-protein', 'val_spearman_per_protein_mean'))
SETTINGS = RC.PROBE_SETTINGS     # max_epochs, patience, min_delta, batch_size, selection
IMPROVE = 1e-5          # trainer's improvement threshold (train/train.py)
FLAG = '⚠'


def tag_of(v):
    return v.replace('final_model_', '').replace('final_model', '') or v


def parse_seeds(text: str) -> list:
    if '-' in text and ',' not in text:
        a, b = text.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(',') if x.strip()]


def agg(vs):
    """(mean, sample std or None, n) over the finite values, or None."""
    vs = [v for v in vs if v is not None and v == v]
    return (st.mean(vs), st.stdev(vs) if len(vs) > 1 else None, len(vs)) if vs else None


class Constant(tuple):
    """All seeds constant predictions, one token (2026-10-18): printed CONSTANT_MARK."""
    constant = True


def fmt(t):
    if t is None:
        return '—'
    if getattr(t, 'constant', False):
        return f'{RC.CONSTANT_MARK} (n={t[2]})'
    m, s, n = t
    return (f'{m:.3f}±{s:.3f}' if s is not None else f'{m:.3f}') + f' (n={n})'


def recon_summary(vdir: Path) -> dict:
    """Best validation reconstruction loss, run status and the loss settings."""
    cfg = json.loads((vdir / 'config.json').read_text()) if (vdir / 'config.json').exists() else {}
    snap = RC.snapshot_info(vdir)
    out = {'best_val_recon': cfg.get('best_val_recon'), 'status': cfg.get('status'),
           'snapshot_of': (snap or {}).get('snapshot_of'),
           'recon_loss_type': cfg.get('recon_loss_type'), 'eval_p': cfg.get('eval_p'),
           'n_val_residues': cfg.get('n_val_residues'), 'hist_best': None, 'hist_epoch': None,
           'mismatch': False}
    hp = vdir / 'history.json'
    if hp.exists():
        h = json.loads(hp.read_text())
        best = float('inf')
        for e, v in zip(h.get('eval_epochs') or [], h.get('val_recon') or []):
            if v < best - IMPROVE:
                best, out['hist_best'], out['hist_epoch'] = float(v), float(v), e
    if out['best_val_recon'] is None:
        out['best_val_recon'] = out['hist_best']
    elif out['hist_best'] is not None and abs(out['best_val_recon'] - out['hist_best']) > \
            1e-6 * max(1.0, abs(out['hist_best'])):
        out['mismatch'] = True
    return out


def status_of(r: dict) -> str:
    if r['snapshot_of']:
        return f'{FLAG} snapshot of {r["snapshot_of"]} ({r["status"] or "?"})'
    if r['status'] in (None, 'finished'):
        return 'finished' if r['status'] else 'finished (legacy config)'
    return f'{FLAG} {r["status"]}'


def collect(variants, datasets, tag, label, seeds):
    """-> (rows, problems). rows: [(variant, recon dict, {(dataset, P, kind): agg})]."""
    from ensembits.featurize import checkpoint_provenance
    rows, problems = [], []
    table = {}          # dataset -> {key: value} that every file must share
    settings = set()
    # FIXED(preretrain_audit_r2 G9, 2026-10-17): new: the results must also match the CURRENT
    #   residue-set file and label file (content hash) and their recorded input caches must
    #   still be on disk unchanged (rmsf_common.records_changed), like the compiler.
    cur_set = {ds: RC.current_residue_set_sha256(ds) for ds in datasets}
    cur_splits = {ds: RC.current_splits_sha256(ds) for ds in datasets}     # K1 (2026-10-17)
    cur_label = {}
    for ds in datasets:
        lp = RC.RMSF_LABEL_FILES[ds].get(label)
        if lp is not None and lp.exists():
            import numpy as np
            cur_label[ds] = RC.label_content_sha256(np.load(lp))
    for v in variants:
        vdir = ROOT / 'output' / v
        if not (vdir / 'best.pt').exists():
            problems.append(f'{v}: no output/{v}/best.pt'); continue
        ck = checkpoint_provenance(vdir)
        cells = {}
        for ds in datasets:
            out_dir = ROOT / 'output' / RC.RESULT_DIRS[ds]
            for pname, suf in P_MODES:
                recs, missing = [], []
                for s in seeds:
                    f = out_dir / f'{tag_of(v)}{suf}{"" if s == 0 else f"_seed{s}"}__{tag}.json'
                    if f.exists():
                        recs.append((f.name, json.loads(f.read_text())))
                    else:
                        missing.append(s)
                if missing:
                    problems.append(f'{v} {ds} {pname}: missing seeds {missing}')
                t = table.setdefault(ds, {})
                for name, js in recs:
                    problems += [f'{v} {ds} {name}: {w}'
                                 for w in RC.records_changed(js.get('feature_caches'))]
                    # FIXED(preretrain_audit_r2 K1, 2026-10-17): the CURRENT split file
                    if cur_splits.get(ds) is not None and js.get('splits_sha256') != cur_splits[ds]:
                        problems.append(f'{v} {ds} {name}: made with another split file')
                    for key in ('residue_set_sha256', 'rmsf_label_sha256', 'feature_caches',
                                'splits_sha256'):
                        val = json.dumps(js.get(key), sort_keys=True)
                        if t.setdefault(key, (val, f'{v}/{name}'))[0] != val:
                            problems.append(f'{v} {ds} {name}: {key} differs from '
                                            f'{t[key][1]}')
                    settings.add(tuple(js.get(k) for k in SETTINGS))
                problems += [f'{v} {ds} {pname}: {p}' for p in RC.check_rmsf_group(
                    recs, dataset=ds, expect_label=label, checkpoint=ck, is_model=True,
                    residue_set_sha256=cur_set.get(ds), label_sha256=cur_label.get(ds))]
                for kname, key in VAL_KEYS:
                    # FIXED(constant-prediction rule, 2026-10-18): all seeds one-token constant
                    #   predictions -> "—†"; any other NaN seed is a problem (was: dropped).
                    if recs and all(RC.constant_prediction_seed(js, key) for _, js in recs):
                        cells[(ds, pname, kname)] = Constant((float('nan'), None, len(recs)))
                        continue
                    vals = [js.get(key) for _, js in recs]
                    nan = [name for (name, _), x in zip(recs, vals) if x is None or x != x]
                    if nan:
                        problems.append(f'{v} {ds} {pname} {kname}: NaN in {nan} (not a '
                                        f'one-token constant prediction)')
                    cells[(ds, pname, kname)] = agg(vals)
        rows.append((v, recon_summary(vdir), cells))
    if len(settings) > 1:
        problems.append(f'probe settings {SETTINGS} differ across files: {sorted(settings, key=str)}')
    return rows, problems


def markdown(rows, datasets, tag, label) -> list:
    cols = [(ds, p, k) for ds in datasets for p, _ in P_MODES for k, _ in VAL_KEYS]
    md = ['# Validation-only selection table', '',
          f'Validation numbers only (no test numbers). RMSF probe files `__{tag}`, label '
          f'`{label}`; RMSF = val Spearman of the epoch selected by early stopping '
          f'(pooled = all val residues; per-protein = mean over val proteins with ≥ '
          f'{RC.PER_PROTEIN_MIN_RESIDUES} residues), mean ± sample std over seeds (n = seeds). '
          f'{FLAG} = unfinished run / snapshot, or config best_val_recon ≠ history.json. '
          'No selection rule is applied here.', '',
          '| variant | status | best val recon (epoch) | ' + ' | '.join(
              f'{d} {p} {k}' for d, p, k in cols) + ' |',
          '|---|---|---|' + '---|' * len(cols)]
    for v, r, cells in rows:
        rec = '—' if r['best_val_recon'] is None else f'{r["best_val_recon"]:.4f}'
        if r['hist_epoch'] is not None:
            rec += f' ({r["hist_epoch"]})'
        if r['mismatch']:
            rec += f' {FLAG} history {r["hist_best"]:.4f}'
        md.append(f'| {v} | {status_of(r)} | {rec} | '
                  + ' | '.join(fmt(cells.get(c)) for c in cols) + ' |')
    if any(getattr(t, 'constant', False) for _, _, cells in rows for t in cells.values()):
        md += ['', RC.CONSTANT_NOTE + '.']
    return md


def bioemu_overlap_note() -> str:
    from ensembits.data_versions import read_provenance
    prov = read_provenance(RC.BIOEMU_SPLITS) or {}
    ov = (prov.get('tokenizer_corpus_overlap') or {}).get('val')
    if ov:
        return (f'{ov["tokenizer_train"]} of {ov["n_domains"]} BioEmu val domains are in the '
                f'tokenizer training split')
    return 'BioEmu val domains overlap the tokenizer training split (73 of 91 for bioemu_cath_misato_e0)'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variants', required=True,
                    help='comma-separated output/<variant> names (finished runs or snapshots)')
    ap.add_argument('--seeds', default='0-9', help='seed set every cell must have (default 0-9)')
    ap.add_argument('--include-bioemu', action='store_true',
                    help='also show BioEmu-CATH (val overlaps the tokenizer training split)')
    ap.add_argument('--tag', default='test', help='result-file suffix __<tag> (default test)')
    ap.add_argument('--rmsf-label', default=RC.RMSF_LABEL_DEFAULT)
    ap.add_argument('--out', default=None, help='also write the markdown here')
    a = ap.parse_args(argv)
    variants = [x.strip() for x in a.variants.split(',') if x.strip()]
    datasets = ['mdcath_div'] + (['bioemu'] if a.include_bioemu else [])
    if a.include_bioemu:
        print(f'[WARNING] BioEmu-CATH is not a clean selection signal: {bioemu_overlap_note()}.',
              file=sys.stderr)
    rows, problems = collect(variants, datasets, a.tag, a.rmsf_label, parse_seeds(a.seeds))
    if problems:
        print(f'[error] {len(problems)} problems; table not written:', file=sys.stderr)
        for p in problems[:80]:
            print(f'  {p}', file=sys.stderr)
        return 1
    keys = {(r['recon_loss_type'], r['eval_p'], r['n_val_residues']) for _, r, _ in rows}
    if len(keys) > 1:
        print(f'[warn] the validation losses are not directly comparable: (recon_loss_type, '
              f'eval_p, n_val_residues) differ across variants: {sorted(keys, key=str)}',
              file=sys.stderr)
    for v, r, _ in rows:
        if r['mismatch']:
            print(f'[warn] {v}: config best_val_recon {r["best_val_recon"]} != history.json '
                  f'{r["hist_best"]}', file=sys.stderr)
    md = markdown(rows, datasets, a.tag, a.rmsf_label)
    print('\n'.join(md))
    if a.out:
        Path(a.out).write_text('\n'.join(md) + '\n')
        print(f'wrote {a.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
