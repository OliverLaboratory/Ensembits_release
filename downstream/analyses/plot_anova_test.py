"""Figure fig:anova_test: per-token s_1 means and the within-protein permutation null.

What it does
------------
Reads the outputs of `token_anova.py` (JSON, `<stem>.residues.npz`, `<stem>.nulls.npz`)
and draws, for each Ensembits variant (P = 10 row), one figure with two panels:
  (a) histogram of s_1 over the residue set with the per-token mean of every token that
      has at least `--rug-min` residues as a tick (display only; the statistics use every
      token); inset: n, M, eta^2, omega^2, F(df), the circular-shift permutation p and the
      held-out within-protein R^2 with its topology-cluster bootstrap CI (no parametric p);
  (b) the null distributions of eta^2 (within-protein circular shift and, for reference,
      the plain shuffle) on a log axis, with the observed eta^2.
Nothing is recomputed here.

Paper
-----
`appendix.tex` app:token_anova, Figure fig:anova_test (`img/ANOVAtest.pdf`; the new file is
written to `output/token_anova/figures/<variant>/` and swapped in by the authors).

Origin
------
Port of the figure block of `experiments/shared_states_v2/test_token_dynamics.py`
(anova_audit A5, 2026-10-18; fix round 2: "p < 10^-3 (0/1,000)" form, s_1/s_2 titles). Changed: the inset shows omega^2, the permutation p of the
circular-shift null and the held-out R^2 instead of the parametric F p-value; panel (b)
shows the circular-shift null next to the plain-shuffle null on a log axis instead of
"real eta^2 is N x the null mean"; the rug omits tokens with fewer than `--rug-min`
residues (their means are noise; was: tokens with >= 80 residues after dropping the rest).

Used by the reported model: yes (fig:anova_test).

Inputs / outputs
----------------
reads   <out>.json, <out stem>.residues.npz, <out stem>.nulls.npz (token_anova.py)
writes  <fig-dir>/<variant>/ANOVAtest.pdf and .png

How to run
----------
    python downstream/analyses/plot_anova_test.py --anova-json output/token_anova/token_anova.json
(token_anova.py calls it at the end unless --no-figures.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT  # noqa: E402

FIG_DIR = REPO_ROOT / 'output' / 'token_anova' / 'figures'
BLUE, ORANGE, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#8a8985', '#0b0b0b', '#52514e'


def _p_text(pc):
    """Circular-shift p: "p < 10^-3 (0/1,000)" when no draw reaches the observed value."""
    b, n = pc.get('b'), pc['n']
    if b == 0:
        e = int(np.floor(np.log10(n)))
        return f'$p < 10^{{-{e}}}$ (0/{n:,})'
    return f"$p$ = {pc['p']:.3g} ({b}/{n:,})" if b is not None else f"$p$ = {pc['p']:.3g}"


def draw(s1, tokens, null_circ, null_shuf, anova, heldout, out_dir: Path, variant: str,
         target_label='s_1', rug_min: int = 30):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), dpi=150)

    ax = axes[0]
    hi = float(np.percentile(s1, 99.5))
    ax.hist(s1[s1 <= hi], bins=80, range=(0, hi), color='#c6d9f1', edgecolor='white', linewidth=0.4)
    ax.set_xlim(0, hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _, inv, cnt = np.unique(tokens, return_inverse=True, return_counts=True)
    means = np.bincount(inv.ravel(), weights=s1) / cnt
    shown = means[cnt >= rug_min]
    ax2 = ax.twinx()
    ax2.scatter(shown[shown <= hi], np.full(int((shown <= hi).sum()), 0.03), marker='|', s=60,
                color=ORANGE, alpha=0.45, linewidths=0.8,
                label=f'per-token mean ({len(shown):,} tokens with $\\geq${rug_min} residues)')
    ax2.set_ylim(0, 1); ax2.set_yticks([])
    ax2.legend(loc='upper center', fontsize=7.5, frameon=False)
    what = 'motion amplitude' if target_label == 's_1' else 'motion shape'
    ax.set_xlabel(f'{what} ${target_label}$ (locally aligned; axis cut at the 99.5th percentile)')
    ax.set_ylabel('residues')
    ax.set_title(f'(a) ${target_label}$ distribution and per-token means', fontsize=9.5, loc='left')
    pc = anova['perm_circular']
    ax.text(0.97, 0.80,
            f"$n$ = {anova['N']:,}, $M$ = {anova['M']:,} tokens\n"
            f"$\\eta^2$ = {anova['eta2']:.3f}, $\\omega^2$ = {anova['omega2']:.3f}\n"
            f"$F$({anova['df_between']:,}, {anova['df_within']:,}) = {anova['F']:.1f}\n"
            f"within-protein circular shift: {_p_text(pc)}\n"
            f"held-out within-protein $R^2$ = {heldout['r2']:.3f}\n"
            f"   95% CI [{heldout['ci95'][0]:.3f}, {heldout['ci95'][1]:.3f}]",
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#d0cfca', alpha=0.95))

    ax = axes[1]
    lo = min(null_circ.min(), null_shuf.min()) * 0.7
    top = anova['eta2'] * 1.4
    bins = np.geomspace(lo, top, 90)
    ax.hist(null_shuf, bins=bins, color=GRAY, alpha=0.8,
            label=f'plain shuffle (mean {null_shuf.mean():.4f})')
    ax.hist(null_circ, bins=bins, color=BLUE, alpha=0.85,
            label=f'within-protein circular shift (mean {null_circ.mean():.4f})')
    ax.axvline(anova['eta2'], color=ORANGE, lw=2.2, label=f"observed $\\eta^2$ = {anova['eta2']:.3f}")
    ax.set_xscale('log')
    ax.set_xlabel(f'$\\eta^2$ of ${target_label}$ (log scale)')
    ax.set_ylabel(f'null draws (n = {len(null_circ):,} each)')
    ax.set_title('(b) Permutation nulls of $\\eta^2$', fontsize=9.5, loc='left')
    ax.legend(fontsize=7.5, frameon=False, loc='upper center')
    for a in axes:
        a.spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ('pdf', 'png'):
        p = out_dir / f'ANOVAtest.{ext}'
        tmp = out_dir / f'.ANOVAtest.tmp.{ext}'
        fig.savefig(tmp, bbox_inches='tight', format=ext)
        tmp.replace(p)
        paths.append(p)
    plt.close(fig)
    return paths


def figures_from_outputs(json_path: Path, fig_dir: Path = FIG_DIR, rug_min: int = 30) -> list:
    js = json.loads(Path(json_path).read_text())
    stem = Path(json_path).with_suffix('')
    R = np.load(str(stem) + '.residues.npz')
    N = np.load(str(stem) + '.nulls.npz')
    t = js['feature']
    T = js['analysis']['targets'][t]
    out = []
    for m in js['methods']:
        if not (m['key'].startswith('ens:') and m.get('p_frames') == 10):
            continue
        k = m['key']
        a = T['anova'][k]
        if 'perm_circular' not in a:
            raise SystemExit(f'{json_path}: no permutation null for {k} (run with --n-perm > 0)')
        out += draw(R[f'target__{t}'].astype(np.float64), R[f'tokens__{k}'],
                    N[f'{t}|{k}|circular'], N[f'{t}|{k}|shuffle'], a, T['heldout_r2'][k],
                    Path(fig_dir) / m['variant'], m['variant'],
                    target_label='s_1' if t == 's1' else 's_1/s_2', rug_min=rug_min)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--anova-json', required=True)
    ap.add_argument('--fig-dir', default=str(FIG_DIR))
    ap.add_argument('--rug-min', type=int, default=30)
    a = ap.parse_args()
    for p in figures_from_outputs(Path(a.anova_json), Path(a.fig_dir), a.rug_min):
        print(p)


if __name__ == '__main__':
    main()
