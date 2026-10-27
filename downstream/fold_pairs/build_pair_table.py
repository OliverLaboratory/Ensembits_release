"""Build the fold-matched pair-test table (markdown + LaTeX rows) from the token comparison.

What it does
------------
Reads `output/fold_pairs/token_comparison.json` (written by `bioemu_token_compare.py`)
and the two verification files `verification_{divergent,similar}.json`, and writes one
table per position set (ALL aligned positions, DYNAMIC = top-quartile RMSF positions):

  per tokenizer: codebook size, mean chance-adjusted agreement on the divergent and on
  the similar pairs (with their 95% bootstrap CIs), Delta = adj(similar) - adj(divergent)
  with its 95% CI, and for the baselines Delta(Ensembits) - Delta(baseline) with its
  paired-bootstrap 95% CI, for Ensembits P=full and P=1;

plus the counts the caption needs: valid pairs per set (verification), pairs used per
set (comparison), dropped pairs, and the floor population (systems, residues per
tokenizer). Nothing is recomputed; numbers are copied from the JSON and rounded.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Infrastructure only (formats the numbers of that table).

Origin
------
New file (2026-10-16, code_fix P21). Replaces the two pair sections of
`downstream/_compile_rebuttal_results.py`, which read the old rebuttal outputs in
`output/rebuttal_analyses/`.

Used by the reported model: yes (Ensembits rows of the pair-test table).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/fold_pairs/token_comparison.json
          output/fold_pairs/verification_{divergent,similar}.json
  writes  output/fold_pairs/pair_table.md   (markdown tables + LaTeX tabular rows)

How to run
----------
    python downstream/fold_pairs/build_pair_table.py [--comparison <json>] [--out <md>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fp_common as C   # noqa: E402

ENS_PREFIX = 'Ensembits'


def _n_valid(path: Path):
    """(valid pairs, total verified) of a verification JSON, or (None, None)."""
    if not Path(path).exists():
        return None, None
    obj = json.load(open(path))
    recs = obj['pairs'] if isinstance(obj, dict) else obj
    return sum(1 for r in recs if r.get('valid')), len(recs)


def _ci(x, plus=False):
    f = (lambda v: f'{v:+.3f}') if plus else (lambda v: f'{v:.3f}')
    return f'[{f(x[0])}, {f(x[1])}]'


def build_tables(comp: dict, ver_counts: dict) -> tuple[list[str], list[str]]:
    """-> (markdown lines, LaTeX lines) for every position set in comp['contrast']."""
    md, tex = [], []
    floor = comp.get('floor', {})
    sets = comp.get('sets', {})
    codebook = {r['tokenizer']: r['codebook'] for r in sets.get('divergent', {}).get('rows', [])}
    md.append('## Fold-matched pair test')
    md.append('')
    md.append('Counts: ' + '; '.join(
        f'{s}: {ver_counts.get(s, (None, None))[0]} valid of {ver_counts.get(s, (None, None))[1]} '
        f'verified, {sets.get(s, {}).get("n_pairs")} used ({len(sets.get(s, {}).get("dropped", []))} dropped)'
        for s in C.SELECTIONS) + f'; floor population: {floor.get("n_systems")} systems.')
    md.append('')
    if comp.get('checkpoint'):
        ck = comp['checkpoint']
        md.append(f'Ensembits checkpoint: `{ck.get("variant")}` (best.pt '
                  f'{str(ck.get("best_pt_sha256"))[:12]}).')
        md.append('')
    contrast = comp.get('contrast')
    if not isinstance(contrast, dict):
        md.append(f'*Contrast not computed: {contrast}*')
        return md, tex
    for label, c in contrast.items():
        per = c['per_tokenizer']; emb = c.get('ensembits_minus_baseline', {})
        ens_names = [n for n in per if n.startswith(ENS_PREFIX)]
        md.append(f'### {label} positions (n_similar = {c["n_similar"]}, n_divergent = '
                  f'{c["n_divergent"]}, {c["n_boot"]} bootstrap resamples)')
        md.append('')
        hdr = ['tokenizer', 'codebook', 'adj. agr. divergent', 'adj. agr. similar',
               'Δ = sim − div [95% CI]'] + [f'Δ({e}) − Δ(b) [95% CI]' for e in ens_names] + \
              ['floor residues']
        md.append('| ' + ' | '.join(hdr) + ' |')
        md.append('|' + '---|' * len(hdr))
        tex.append(f'% {label} positions: tokenizer & K & adj(div) & adj(sim) & Delta [CI] & '
                   + ' & '.join(f'Delta({e})-Delta(b) [CI]' for e in ens_names) + r' \\')
        for n, x in per.items():
            fl = floor.get('per_tokenizer', {}).get(n, {}).get('n_residues')
            cells = [n, str(codebook.get(n, '—')),
                     f'{x["mean_divergent"]:.3f} {_ci(x["ci_divergent"])}',
                     f'{x["mean_similar"]:.3f} {_ci(x["ci_similar"])}',
                     f'{x["delta"]:+.3f} {_ci(x["ci_delta"], True)}']
            tcells = [n.replace('=', '$=$'), str(codebook.get(n, '--')),
                      f'{x["mean_divergent"]:.3f}', f'{x["mean_similar"]:.3f}',
                      f'${x["delta"]:+.3f}$ [{x["ci_delta"][0]:+.3f}, {x["ci_delta"][1]:+.3f}]']
            for e in ens_names:
                d = emb.get(f'{e} - {n}')
                if d is None:
                    cells.append('—'); tcells.append('--')
                else:
                    cells.append(f'{d["diff"]:+.3f} {_ci(d["ci"], True)}')
                    tcells.append(f'${d["diff"]:+.3f}$ [{d["ci"][0]:+.3f}, {d["ci"][1]:+.3f}]')
            cells.append(str(fl) if fl is not None else '—')
            md.append('| ' + ' | '.join(cells) + ' |')
            tex.append(' & '.join(tcells) + r' \\')
        md.append('')
    return md, tex


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--comparison', default=str(C.COMPARISON))
    ap.add_argument('--divergent', default=str(C.verification_path('divergent')))
    ap.add_argument('--similar', default=str(C.verification_path('similar')))
    ap.add_argument('--out', default=str(C.OUT / 'pair_table.md'))
    args = ap.parse_args()
    comp_p = Path(args.comparison)
    if not comp_p.exists():
        print(f'[pair-table] {comp_p} missing; run bioemu_token_compare.py first', file=sys.stderr)
        return 1
    comp = json.load(open(comp_p))
    ver = {'divergent': _n_valid(args.divergent), 'similar': _n_valid(args.similar)}
    md, tex = build_tables(comp, ver)
    if comp.get('interpretation'):
        md += ['', f'*{comp["interpretation"]}*', '']
    md += ['## LaTeX rows', '', '```latex'] + tex + ['```', '']
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(md) + '\n')
    print(f'[pair-table] wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
