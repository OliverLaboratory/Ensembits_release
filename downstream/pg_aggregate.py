"""ProteinGym table with alpha chosen by leave-one-UniProt-out CV, the -n_mut control and paired tests.

What it does
------------
Reads the per-variant files written by `pg_score_dihedral.py` (Ensembits) and
`pg_score_baselines.py` (baselines) for the given output tags, and for the fixed
assay list (default `downstream/pg_assays_wt_le200.txt`, 96 assays):
  1. checks provenance: every tag has the current schema (pg_version), scored every
     assay of the list, and used the same ESMFold structures and CSV for each assay
     (content sha256), and every per-variant npz matches the sha256 recorded in its
     per-assay JSON; the list, structure set and CSV set sha256 go into the output;
  2. aligns the methods variant by variant (intersection of mutant ids; DMS, ESM2 and
     n_mut must agree exactly across tags);
  3. per assay and predictor computes the Spearman of the score alone, on single
     mutants only (assays with >= 10 singles), the z-blend with ESM2 at every alpha of
     the grid, and the parameter-free rank-sum blend rank(score) + rank(ESM2).
     Predictors: each method's -score (score (a) by default), ESM2-650M alone, and the
     control -n_mut (minus the number of mutated positions: 70% of the variants are
     multi-mutants and a sum over positions grows with the mutation count);
  4. chooses alpha by leave-one-UniProt-out CV (`pg_common.louo_alpha_cv`): for each
     protein, alpha is the grid value with the best mean Spearman over the other
     proteins' assays; the same rule for every tokenizer and for -n_mut; reports the
     distribution of the chosen alphas, and, for reference only, the test-selected
     alpha and the gap between a fixed alpha (--fixed-alpha, 0.3 in the preprint) and
     the test-best alpha;
  5. aggregates: unweighted mean over assays (the paper's metric), mean over UniProt
     entries (each protein weighted once, ProteinGym's convention), Tsuboyama /
     other assays, single mutants only;
  6. paired tests over assays (Wilcoxon signed-rank, paired t) and over proteins
     (Wilcoxon on per-UniProt means), Holm-adjusted over the whole family: every
     predictor's CV blend vs ESM2 alone; every method vs -n_mut (alone and blended);
     every --primary method vs every other method (alone, CV blend, rank-sum).
     (Exploratory since final_fix_list A5: JSON key `tests`, not in the table.)
  7. (final_fix_list A5, 2026-10-19) the approved table: rows = tokenizers, ESM2-650M,
     -n_mut (no inverse-folding / structure-LM rows); columns = alone (assay mean
     [per-UniProt mean]), singles only, Tsuboyama / other, + ESM2 (CV alpha) with the
     alpha distribution, Delta vs ESM2 alone with a cluster-bootstrap 95% CI (cluster =
     UniProt entry; `--n-boot`, `--boot-seed`); the PRE-SPECIFIED test family
     (`--family`, default primary + ESM2 vs ESM2 alone and primary + ESM2 vs each other
     tokenizer + ESM2): Wilcoxon signed-rank on the per-UniProt mean differences,
     Holm-adjusted over that family only, with the claim rule "win only if Delta >= 0.01
     and Holm p < 0.05"; every predictor on the multi-mutant assays (55 of the 96: the
     -n_mut control is constant on single-mutant-only assays); an appendix table with
     the rank-sum blend and the within-n_mut Spearman (per assay: Spearman inside each
     n_mut stratum with >= 10 variants, averaged with stratum-size weights).
Writes a JSON (all per-assay values) and a markdown file with the tables.

Lambda blind (final_fix_list A1 / A5): a tag whose sidecar checkpoint is not a baseline
("baseline:<name>") is an Ensembits tag; the aggregate refuses to read any Ensembits tag
unless `--primary` names the chosen tag (the check runs on the sidecars, before any
per-assay value is loaded). With --print-table off it prints nothing but file names.

Paper
-----
`2_results.tex`, \\paragraph{Mutation Effect Prediction} and Table
\\label{tab:proteingym_ensemble}: the camera-ready protocol replaces "alpha = 0.3,
the grid-best alpha for 4/5 tokenizers" (chosen on the same 96 test assays) with the
leave-one-UniProt-out alpha, adds the rank-sum blend and the -n_mut control, and
discloses the subset (benchmark_audit D9; preretrain_audit_r2 Q31 / Q32).

Origin
------
New (2026-10-19, benchmark_audit D9). Replaces the alpha-grid means of
`_compile_rebuttal_results.py` (PG rows) and the ProteinGym part of
`significance_tests.py` (which read the invalid released baseline JSONs) for the
camera-ready table; uses `significance_tests.holm`.

Used by the reported model: yes (table rows of every tokenizer and the controls).

FIXED(final_fix_list A5, 2026-10-19): the approved table (point 7), the pre-specified Holm
family, the cluster-bootstrap CIs, the multi-mutant subset, the appendix table (rank-sum,
within-n_mut Spearman) and the lambda-blind refusal are new; the scoring files
(pg_score.py, pg_common.py, pg_score_baselines.py) are unchanged. Old: one table with
every column, Holm over all exploratory tests on the assay-level Wilcoxon, no CI, no
refusal of Ensembits tags.

Inputs / outputs
----------------
Reads (REPO_ROOT/output/proteingym):
  per_assay_pg_<tag>.assays_scored.json, per_assay_pg_<tag>/<assay>.json,
  per_variant_pg_<tag>/<assay>.npz            for each --method tag
  downstream/pg_assays_wt_le200.txt          (or --assay-list)
Writes (REPO_ROOT/output/proteingym):
  pg_d9_<name>.json   {meta (tags, provenance, grid, list sha), per_assay, summary, alpha, tests}
  pg_d9_<name>.md     the table

How to run
----------
    python downstream/pg_aggregate.py --name baselines --family tokenizers_vs_esm2 \\
        --method bl_3di_d9 --method bl_aminoaseed_d9 --method bl_esm3struct_d9 \\
        --method bl_protoken_d9 --label bl_3di_d9=3Di --label bl_aminoaseed_d9=AminoAseed \\
        --label bl_esm3struct_d9=ESM3struct --label bl_protoken_d9=ProToken
    (after the lambda decision: add --method bm_<chosen>_rv --primary bm_<chosen>_rv
     --label bm_<chosen>_rv=Ensembits; the default --family is then
     primary_vs_esm2,primary_vs_tokenizers)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT          # noqa: E402
import pg_common as C                                  # noqa: E402
from pg_score import PG_VERSION                        # noqa: E402

PG_OUT = ROOT / 'output' / 'proteingym'
ESM2 = 'ESM2-650M'
NMUT = '-n_mut'
MIN_SINGLES = 10
# FIXED(final_fix_list A5, 2026-10-19): new constants of the approved table.
MIN_STRATUM = 10                 # within-n_mut Spearman: strata with >= 10 variants
CLAIM_MIN_DELTA = 0.01           # "win" only if Delta >= 0.01 and Holm p < 0.05
CLAIM_ALPHA = 0.05
# Pre-specified test families (--family, comma-separated). Every comparison is on the
# "+ ESM2 (CV alpha)" column except '*_vs_esm2' (blend vs ESM2 alone).
FAMILY_ITEMS = {
    'primary_vs_esm2': 'primary + ESM2 (CV alpha) vs ESM2 alone',
    'primary_vs_tokenizers': 'primary + ESM2 vs each other tokenizer + ESM2 (CV alpha)',
    'tokenizers_vs_esm2': 'each tokenizer + ESM2 (CV alpha) vs ESM2 alone',
    'tokenizers_vs_nmut': 'each tokenizer + ESM2 vs -n_mut + ESM2 (CV alpha)',
}
# Q3: added tokenizers_vs_nmut so "primary + ESM2 vs -n_mut + ESM2" is pre-specified
DEFAULT_FAMILY = ('primary_vs_esm2', 'primary_vs_tokenizers', 'tokenizers_vs_nmut')
N_BOOT = 10000


class ProvenanceError(RuntimeError):
    pass


class BlindError(RuntimeError):
    """An Ensembits tag was requested without --primary (lambda selection is blind to PG)."""


def read_sidecar(tag: str, out_dir: Path = PG_OUT) -> dict:
    """The `per_assay_pg_<tag>.assays_scored.json` sidecar (ProvenanceError if missing)."""
    side_p = out_dir / f'per_assay_pg_{tag}.assays_scored.json'
    if not side_p.exists():
        raise ProvenanceError(f'{side_p.name} missing (run pg_score_* for {tag})')
    return json.loads(side_p.read_text())


def is_ensembits_sidecar(side: dict) -> bool:
    """True unless the sidecar's checkpoint is a baseline ('baseline:<name>')."""
    v = str((side.get('checkpoint') or {}).get('variant') or '')
    return not v.startswith('baseline:')


def check_blind(tags, primary, out_dir: Path = PG_OUT) -> list[str]:
    """FIXED(final_fix_list A5, 2026-10-19): lambda-blind gate, on the sidecars only.

    Returns the Ensembits tags; raises BlindError when one is requested without --primary,
    or when --primary is not one of the requested tags.

    A.4 fix: also refuses Ensembits tags not in --primary, and checks that each Ensembits
    tag's best_pt_sha256 matches the current output/<variant>/best.pt of an _rv run.
    """
    ens = [t for t in tags if is_ensembits_sidecar(read_sidecar(t, out_dir))]
    bad = [p for p in primary if p not in tags]
    if bad:
        raise BlindError(f'--primary {bad} is not among the --method tags')
    if ens and not primary:
        raise BlindError(f'Ensembits tags {ens} requested without --primary: the Ensembits '
                         f'ProteinGym aggregate stays blocked until lambda is chosen (pass '
                         f'--primary <chosen tag>)')
    # A.4 fix: refuse Ensembits tags not in --primary
    extra_ens = [t for t in ens if t not in primary]
    if extra_ens:
        raise BlindError(f'Ensembits tags {extra_ens} not in --primary: every Ensembits tag '
                         f'must be explicitly listed as --primary')
    # A.4 fix: check best_pt_sha256 against the _rv run checkpoint (when available)
    for t in ens:
        side = read_sidecar(t, out_dir)
        ckpt = side.get('checkpoint') or {}
        variant = ckpt.get('variant', '')
        side_sha = ckpt.get('best_pt_sha256')
        if side_sha and len(str(side_sha)) == 64:
            rv_best = ROOT / 'output' / variant / 'best.pt'
            if rv_best.exists():
                cur_sha = C.file_sha256(rv_best)
                if cur_sha != side_sha:
                    raise BlindError(f'{t}: sidecar best_pt_sha256 {side_sha[:12]}… '
                                     f'!= output/{variant}/best.pt {cur_sha[:12]}…')
            else:
                raise BlindError(f'{t}: output/{variant}/best.pt not found '
                                 f'(cannot verify Ensembits tag checkpoint)')
    return ens


# ---------------------------------------------------------------- loading
def load_tag(tag: str, assays: list[str], out_dir: Path = PG_OUT) -> tuple[dict, dict]:
    """({assay: per-variant arrays}, sidecar) of one tag; raises ProvenanceError."""
    side = read_sidecar(tag, out_dir)
    if side.get('pg_version') != PG_VERSION:
        raise ProvenanceError(f'{tag}: pg_version {side.get("pg_version")!r} != {PG_VERSION!r}')
    missing = sorted(set(assays) - set(side.get('assays_scored', [])))
    if missing:
        raise ProvenanceError(f'{tag}: {len(missing)} assays not scored, e.g. {missing[:3]}')
    data = {}
    for a in assays:
        rec = json.loads((out_dir / f'per_assay_pg_{tag}' / f'{a}.json').read_text())
        if rec.get('pg_version') != PG_VERSION:
            raise ProvenanceError(f'{tag}/{a}: old per-assay schema')
        npz = out_dir / f'per_variant_pg_{tag}' / f'{a}.npz'
        if C.file_sha256(npz) != rec.get('per_variant_sha256'):
            raise ProvenanceError(f'{tag}/{a}: per-variant npz does not match its JSON sha256')
        with np.load(npz, allow_pickle=False) as z:
            data[a] = {k: z[k] for k in z.files}
        data[a]['_structure_sha256'] = rec['inputs']['structure_sha256']
        data[a]['_csv_sha256'] = rec['inputs']['csv_sha256']
    return data, side


def check_provenance(sides: dict, data: dict, assays: list[str], allow_mismatch=False) -> dict:
    """Same assay list and same structure / CSV per assay across tags; returns meta."""
    # A tag scored on a superset of the list is fine (load_tag checks every listed assay
    # was scored); what must agree is the structure and CSV content of each assay.
    problems = []
    list_sha = C.assay_list_sha256(assays)
    for a in assays:
        for key in ('_structure_sha256', '_csv_sha256'):
            vals = {tag: data[tag][a][key] for tag in data}
            if len(set(vals.values())) > 1:
                problems.append(f'{a}: {key.strip("_")} differs across tags')
    if problems and not allow_mismatch:
        raise ProvenanceError('; '.join(problems[:5]) + (f' (+{len(problems) - 5})'
                                                          if len(problems) > 5 else ''))
    any_tag = next(iter(data))
    return {'assay_list_sha256': list_sha,
            'structure_set_sha256': C.structure_set_sha256(
                {a: data[any_tag][a]['_structure_sha256'] for a in assays}),
            'csv_set_sha256': C.structure_set_sha256(
                {a: data[any_tag][a]['_csv_sha256'] for a in assays}),
            'provenance_problems': problems,
            'checkpoints': {tag: s.get('checkpoint') for tag, s in sides.items()}}


def align(data: dict, assay: str, score_key: str) -> dict:
    """Common variants of one assay across tags: {mutant, fitness, esm2, n_mut, scores{tag}}."""
    tags = list(data)
    common = set(data[tags[0]][assay]['mutant'].tolist())
    for t in tags[1:]:
        common &= set(data[t][assay]['mutant'].tolist())
    mut = np.array(sorted(common))
    out = {'mutant': mut, 'scores': {}}
    for t in tags:
        d = data[t][assay]
        idx = {m: i for i, m in enumerate(d['mutant'].tolist())}
        sel = np.array([idx[m] for m in mut], dtype=np.int64)
        for k in ('fitness', 'esm2', 'n_mut'):
            v = np.asarray(d[k])[sel]
            if k in out and not np.array_equal(out[k], v):
                raise ProvenanceError(f'{assay}: {k} differs between tags on common variants')
            out[k] = v
        out['scores'][t] = np.asarray(d[score_key], dtype=np.float64)[sel]
    out['n_dropped'] = {t: int(len(data[t][assay]['mutant']) - len(mut)) for t in tags}
    return out


# ---------------------------------------------------------------- metrics
def per_assay_metrics(al: dict, preds: dict, grid, fixed_alpha) -> dict:
    """{predictor: {alone, singles, blend[grid], ranksum, fixed}} for one aligned assay."""
    y, e = al['fitness'], al['esm2']
    single = al['n_mut'] == 1
    res = {}
    for name, p in preds.items():
        r = {'alone': C.spearman(p, y)[0], 'constant': C.spearman(p, y)[1]}
        if single.sum() >= MIN_SINGLES:
            rho_s, const_s = C.spearman(p[single], y[single])
            # B.17: print "—" for -n_mut singles (constant predictor on single mutants)
            r['singles'] = None if (name == NMUT and const_s) else rho_s
        else:
            r['singles'] = None
        r['within_nmut'] = within_nmut_spearman(p, y, al['n_mut'])   # A5 appendix
        if name != ESM2:
            r['blend'] = [float(v) for v in C.blend_spearman(p, e, y, grid)]
            r['ranksum'] = C.ranksum_spearman(p, e, y)
            if fixed_alpha is not None:
                r['fixed'] = float(C.blend_spearman(p, e, y, (fixed_alpha,))[0])
        res[name] = r
    return res


def within_nmut_spearman(pred, y, n_mut, min_n: int = MIN_STRATUM):
    """Spearman inside each n_mut stratum with >= min_n variants, averaged with stratum-size
    weights (None when no stratum qualifies). Removes the mutation-count effect: a score
    that only counts mutations (-n_mut) is constant inside every stratum and scores 0."""
    n_mut = np.asarray(n_mut)
    rs, ws = [], []
    for k in np.unique(n_mut):
        m = n_mut == k
        if m.sum() >= min_n:
            rs.append(C.spearman(np.asarray(pred)[m], np.asarray(y)[m])[0])
            ws.append(int(m.sum()))
    return float(np.average(rs, weights=ws)) if rs else None


def cluster_bootstrap_ci(values: dict, n_boot: int = N_BOOT, seed: int = 0,
                         group_of=C.uniprot_of, level: float = 0.95):
    """Percentile CI of the unweighted mean over assays of {assay: value}, resampling
    UniProt clusters (all assays of a protein together) with replacement."""
    keys = sorted(k for k, v in values.items() if v is not None)
    if not keys:
        return None
    groups = sorted({group_of(k) for k in keys})
    gi = {g: i for i, g in enumerate(groups)}
    sums = np.zeros(len(groups)); cnt = np.zeros(len(groups))
    for k in keys:
        sums[gi[group_of(k)]] += float(values[k]); cnt[gi[group_of(k)]] += 1
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b0 in range(0, n_boot, 1000):                     # chunks keep memory small
        idx = rng.integers(len(groups), size=(min(1000, n_boot - b0), len(groups)))
        out[b0:b0 + len(idx)] = sums[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.quantile(out, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def _diff(x: dict, y: dict) -> dict:
    return {k: x[k] - y[k] for k in x if k in y and x[k] is not None and y[k] is not None}


def _mean(vals):
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def aggregate(values: dict) -> dict:
    """{assay: value} -> mean over assays, over UniProt, Tsuboyama / other."""
    by_u = {}
    for a, v in values.items():
        if v is not None:
            by_u.setdefault(C.uniprot_of(a), []).append(v)
    m, n = _mean(values.values())
    mu, nu = _mean([np.mean(v) for v in by_u.values()])
    mt, nt = _mean([v for a, v in values.items() if C.is_tsuboyama(a)])
    mo, no = _mean([v for a, v in values.items() if not C.is_tsuboyama(a)])
    return {'mean': m, 'n': n, 'mean_uniprot': mu, 'n_uniprot': nu,
            'mean_tsuboyama': mt, 'n_tsuboyama': nt, 'mean_other': mo, 'n_other': no}


def paired(name: str, x: dict, y: dict) -> dict:
    """Paired tests of two {assay: value} maps over shared assays and over UniProt means."""
    keys = sorted(k for k in x if k in y and x[k] is not None and y[k] is not None)
    d = np.array([x[k] - y[k] for k in keys], dtype=np.float64)
    def wil(diff):
        diff = diff[diff != 0]
        return float(stats.wilcoxon(diff).pvalue) if len(diff) >= 1 else 1.0
    if len(keys) < 2:
        p_t = None
    elif not np.any(d):                          # identical values: no difference
        p_t = 1.0
    else:
        p_t = float(stats.ttest_rel([x[k] for k in keys], [y[k] for k in keys]).pvalue)
    by_u = {}
    for k, v in zip(keys, d):
        by_u.setdefault(C.uniprot_of(k), []).append(v)
    du = np.array([np.mean(v) for v in by_u.values()])
    return {'comparison': name, 'n_assays': len(keys), 'n_uniprot': len(du),
            'delta': float(d.mean()) if len(d) else None,
            'p_wilcoxon': wil(d), 'p_ttest': p_t,
            'p_wilcoxon_uniprot': wil(du)}


def resolve_family(family, primary) -> list[str]:
    """The pre-specified family items (default DEFAULT_FAMILY); ValueError if unknown or if
    a primary item is requested without --primary."""
    items = list(DEFAULT_FAMILY if family is None else family)
    unknown = [i for i in items if i not in FAMILY_ITEMS]
    if unknown:
        raise ValueError(f'unknown --family items {unknown}; choose from {sorted(FAMILY_ITEMS)}')
    if not primary and any(i.startswith('primary') for i in items):
        raise ValueError(f'--family {",".join(items)} needs --primary (baselines only: pass e.g. '
                         f'--family tokenizers_vs_esm2)')
    return items


def run(tags: list[str], labels: dict, assays: list[str], grid, fixed_alpha=None,
        primary=(), score_key='score_a', out_dir: Path = PG_OUT, allow_mismatch=False,
        family=None, n_boot: int = N_BOOT, boot_seed: int = 0) -> dict:
    """Everything above; returns the result dict (also used by the tests).

    family: pre-specified test family (FAMILY_ITEMS keys; None = DEFAULT_FAMILY).
    """
    # FIXED(final_fix_list A5, 2026-10-19): lambda-blind gate and family check before any
    #   per-assay value is read.
    ens_tags = check_blind(tags, primary, out_dir)
    fam = resolve_family(family, primary)
    data, sides = {}, {}
    for t in tags:
        data[t], sides[t] = load_tag(t, assays, out_dir)
    meta = check_provenance(sides, data, assays, allow_mismatch)
    names = [labels.get(t, t) for t in tags]
    per_assay, n_dropped = {}, {}
    for a in assays:
        al = align(data, a, score_key)
        preds = {labels.get(t, t): -al['scores'][t] for t in tags}    # higher = fitter
        preds[NMUT] = -al['n_mut'].astype(np.float64)
        preds[ESM2] = al['esm2']
        per_assay[a] = per_assay_metrics(al, preds, grid, fixed_alpha)
        per_assay[a]['_n_variants'] = int(len(al['mutant']))
        per_assay[a]['_n_multi'] = int((al['n_mut'] > 1).sum())
        n_dropped[a] = al['n_dropped']

    blended = names + [NMUT]
    alpha, summary = {}, {}
    for nm in blended + [ESM2]:
        col = {'alone': {a: per_assay[a][nm]['alone'] for a in assays},
               'singles': {a: per_assay[a][nm]['singles'] for a in assays},
               'within_nmut': {a: per_assay[a][nm]['within_nmut'] for a in assays}}
        if nm != ESM2:
            cv = C.louo_alpha_cv({a: per_assay[a][nm]['blend'] for a in assays}, grid)
            col['blend_cv'] = cv['rho_cv']
            col['ranksum'] = {a: per_assay[a][nm]['ranksum'] for a in assays}
            if fixed_alpha is not None:
                col['blend_fixed'] = {a: per_assay[a][nm]['fixed'] for a in assays}
            ag = list(cv['alpha_by_group'].values())
            full = cv['rho_full_by_alpha']
            fixed_gap = None
            if fixed_alpha is not None:
                fixed_gap = float(max(full) - np.mean(list(col['blend_fixed'].values())))
            alpha[nm] = {'by_group': cv['alpha_by_group'], 'counts': dict(sorted(Counter(ag).items())),
                         'median': float(np.median(ag)), 'min': float(min(ag)), 'max': float(max(ag)),
                         'test_selected_alpha': cv['alpha_full'],
                         'mean_by_alpha_test': full,
                         'fixed_alpha_gap_to_test_best': fixed_gap}
        summary[nm] = {k: aggregate(v) for k, v in col.items()}
        summary[nm]['_per_assay'] = col
        summary[nm]['n_constant_alone'] = sum(per_assay[a][nm]['constant'] for a in assays)

    # ---- paired tests ----
    tests = []
    e2 = summary[ESM2]['_per_assay']['alone']
    for nm in blended:
        tests.append(paired(f'{nm} + ESM2 (CV alpha) vs ESM2 alone', summary[nm]['_per_assay']['blend_cv'], e2))
        tests.append(paired(f'{nm} + ESM2 (rank-sum) vs ESM2 alone', summary[nm]['_per_assay']['ranksum'], e2))
    for nm in names:
        for key, what in (('alone', 'alone'), ('blend_cv', '+ESM2 CV')):
            tests.append(paired(f'{nm} vs -n_mut ({what})', summary[nm]['_per_assay'][key],
                                summary[NMUT]['_per_assay'][key]))
    prim = [labels.get(t, t) for t in primary]
    for p in prim:
        for nm in names:
            if nm == p:
                continue
            for key, what in (('alone', 'alone'), ('blend_cv', '+ESM2 CV'), ('ranksum', 'rank-sum')):
                tests.append(paired(f'{p} vs {nm} ({what})', summary[p]['_per_assay'][key],
                                    summary[nm]['_per_assay'][key]))
    if tests:
        from significance_tests import holm
        for r, h in zip(tests, holm([r['p_wilcoxon'] for r in tests])):
            r['p_wilcoxon_holm'] = h

    # ---- FIXED(final_fix_list A5, 2026-10-19): Delta vs ESM2 with cluster-bootstrap CIs,
    #   the pre-specified family (Holm over it only, per-UniProt Wilcoxon), the multi-mutant
    #   subset.
    boot = dict(n_boot=n_boot, seed=boot_seed)
    for nm in blended:
        d = _diff(summary[nm]['_per_assay']['blend_cv'], e2)
        dr = _diff(summary[nm]['_per_assay']['ranksum'], e2)
        summary[nm]['delta_vs_esm2'] = {**aggregate(d), 'ci95': cluster_bootstrap_ci(d, **boot)}
        summary[nm]['delta_ranksum_vs_esm2'] = {**aggregate(dr),
                                               'ci95': cluster_bootstrap_ci(dr, **boot)}
    pa = lambda nm, key: summary[nm]['_per_assay'][key]
    comps = []   # (family item, comparison, x, y)
    for item in fam:
        if item == 'primary_vs_esm2':
            comps += [(item, f'{p} + ESM2 (CV alpha) vs ESM2 alone', pa(p, 'blend_cv'), e2) for p in prim]
        elif item == 'primary_vs_tokenizers':
            comps += [(item, f'{p} + ESM2 vs {nm} + ESM2 (CV alpha)', pa(p, 'blend_cv'),
                       pa(nm, 'blend_cv')) for p in prim for nm in names if nm != p]
        elif item == 'tokenizers_vs_esm2':
            comps += [(item, f'{nm} + ESM2 (CV alpha) vs ESM2 alone', pa(nm, 'blend_cv'), e2)
                      for nm in names]
        elif item == 'tokenizers_vs_nmut':
            comps += [(item, f'{nm} + ESM2 vs -n_mut + ESM2 (CV alpha)', pa(nm, 'blend_cv'),
                       pa(NMUT, 'blend_cv')) for nm in names]
    family_tests = []
    for item, name, x, y in comps:
        r = paired(name, x, y)
        r['family_item'] = item
        r['ci95_delta'] = cluster_bootstrap_ci(_diff(x, y), **boot)
        family_tests.append(r)
    if family_tests:
        from significance_tests import holm
        for r, h in zip(family_tests, holm([r['p_wilcoxon_uniprot'] for r in family_tests])):
            r['p_holm'] = h
            r['claim'] = bool(r['delta'] is not None and r['delta'] >= CLAIM_MIN_DELTA
                              and h < CLAIM_ALPHA)
    multi = [a for a in assays if per_assay[a]['_n_multi'] > 0]
    sub = lambda m: {a: m[a] for a in multi}
    multi_summary = {}
    for nm in blended + [ESM2]:
        ms = {'alone': aggregate(sub(pa(nm, 'alone')))}
        if nm != ESM2:
            d = _diff(sub(pa(nm, 'blend_cv')), sub(e2))
            ms['blend_cv'] = aggregate(sub(pa(nm, 'blend_cv')))
            ms['delta_vs_esm2'] = {**aggregate(d), 'ci95': cluster_bootstrap_ci(d, **boot)}
        multi_summary[nm] = ms

    meta.update({'ensembits_tags': ens_tags, 'primary': prim,
                 'family': {'items': fam, 'definitions': {i: FAMILY_ITEMS[i] for i in fam},
                            'test': 'Wilcoxon signed-rank on per-UniProt mean differences, '
                                    'Holm-adjusted over the family only',
                            'claim_rule': f'Delta >= {CLAIM_MIN_DELTA} and Holm p < {CLAIM_ALPHA}'},
                 'bootstrap': {'B': n_boot, 'seed': boot_seed, 'cluster': 'UniProt entry',
                               'statistic': 'unweighted mean over assays', 'ci': 'percentile 95%'},
                 'within_nmut_rule': f'Spearman inside each n_mut stratum with >= {MIN_STRATUM} '
                                     f'variants, stratum-size weighted mean per assay',
                 'n_multi_mutant_assays': len(multi)})
    meta.update({'tags': tags, 'labels': labels, 'score_key': score_key,
                 'alpha_grid': list(grid), 'fixed_alpha': fixed_alpha, 'n_assays': len(assays),
                 'pg_version': PG_VERSION, 'n_dropped_by_alignment': n_dropped,
                 'cv': 'leave-one-UniProt-out (UniProt = first two fields of the assay id); '
                       'alpha = argmax of the unweighted mean Spearman over the other '
                       "proteins' assays; ties -> smallest alpha"})
    return {'meta': meta, 'summary': summary, 'alpha': alpha, 'tests': tests,
            'family_tests': family_tests, 'multi_mutant': multi_summary,
            'per_assay': per_assay}


def to_markdown(res: dict) -> str:
    """FIXED(final_fix_list A5, 2026-10-19): the approved table, the pre-specified family,
    the multi-mutant subset and the appendix table. Old: one table with every column and the
    exploratory tests (Holm over all of them; those stay in the JSON key `tests`)."""
    S, A, m = res['summary'], res['alpha'], res['meta']
    f = lambda x: '—' if x is None else f'{x:.3f}'
    fs = lambda x: '—' if x is None else f'{x:+.3f}'
    ci = lambda c: '' if not c else f' [{c[0]:+.3f}, {c[1]:+.3f}]'
    prim = set(m.get('primary') or [])
    order = ([nm for nm in S if nm in prim] + [nm for nm in S if nm not in prim | {ESM2, NMUT}]
             + [ESM2, NMUT])
    disp = lambda nm: f'**{nm}** (primary)' if nm in prim else nm
    L = [f"# ProteinGym (final_fix_list A5): {m['n_assays']} assays, score `{m['score_key']}`", '',
         'Spearman, unweighted mean over assays [mean over UniProt entries]. Singles: single '
         f'mutants only (assays with >= {MIN_SINGLES}). Tsuboyama / other: the {S[ESM2]["alone"]["n_tsuboyama"]} '
         f'Tsuboyama stability assays / the {S[ESM2]["alone"]["n_other"]} others. + ESM2: z-blend with '
         'ESM2-650M, CV alpha = leave-one-UniProt-out (alpha: median [min, max] over the '
         f"{len(next(iter(A.values()))['by_group']) if A else 0} folds). Δ vs ESM2: + ESM2 minus "
         'ESM2 alone, mean over assays, 95% CI = UniProt-cluster bootstrap '
         f"(B = {m['bootstrap']['B']}).", '',
         '| predictor | alone | singles only | alone Tsuboyama / other | + ESM2 (CV α) | '
         '+ ESM2 Tsuboyama / other | α | Δ vs ESM2 [95% CI] |',
         '|---|---|---|---|---|---|---|---|']
    for nm in order:
        s, al = S[nm], S[nm]['alone']
        row = (f"| {disp(nm)} | {f(al['mean'])} [{f(al['mean_uniprot'])}] | "
               f"{f(s['singles']['mean'])} | {f(al['mean_tsuboyama'])} / {f(al['mean_other'])} | ")
        if nm in A:
            b, d = s['blend_cv'], s['delta_vs_esm2']
            row += (f"{f(b['mean'])} [{f(b['mean_uniprot'])}] | {f(b['mean_tsuboyama'])} / "
                    f"{f(b['mean_other'])} | {A[nm]['median']:.2f} [{A[nm]['min']:.2f}, "
                    f"{A[nm]['max']:.2f}] | {fs(d['mean'])}{ci(d['ci95'])} |")
        else:
            row += '— | — | — | — |'
        L.append(row)
    fam = m['family']
    L += ['', f"## Pre-specified test family ({', '.join(fam['items'])})", '',
          f"{fam['test']}; win only if {fam['claim_rule']}. Δ = mean over assays of the paired "
          'difference, 95% CI = UniProt-cluster bootstrap.', '',
          '| comparison | assays / UniProt | Δ [95% CI] | p Wilcoxon (UniProt means) | p Holm | win |',
          '|---|---|---|---|---|---|']
    for r in res.get('family_tests', []):
        L.append(f"| {r['comparison']} | {r['n_assays']} / {r['n_uniprot']} | {fs(r['delta'])}"
                 f"{ci(r['ci95_delta'])} | {r['p_wilcoxon_uniprot']:.2g} | {r['p_holm']:.2g} | "
                 f"{'yes' if r['claim'] else 'no'} |")
    if not res.get('family_tests'):
        L.append('| (empty family) | | | | | |')
    MM = res.get('multi_mutant', {})
    n_multi = m.get('n_multi_mutant_assays')
    L += ['', f'## Multi-mutant assays ({n_multi} of {m["n_assays"]})', '',
          '-n_mut is constant (Spearman 0) on the single-mutant-only assays, so its 96-assay '
          'mean understates it; on the assays with multi-mutants it is a real predictor.', '',
          '| predictor | alone | + ESM2 (CV α) | Δ vs ESM2 [95% CI] |', '|---|---|---|---|']
    for nm in order:
        if nm not in MM:
            continue
        x = MM[nm]
        if nm == ESM2:
            L.append(f"| {disp(nm)} | {f(x['alone']['mean'])} | — | — |")
        else:
            L.append(f"| {disp(nm)} | {f(x['alone']['mean'])} | {f(x['blend_cv']['mean'])} | "
                     f"{fs(x['delta_vs_esm2']['mean'])}{ci(x['delta_vs_esm2']['ci95'])} |")
    fx = m['fixed_alpha'] is not None
    L += ['', '## Appendix: rank-sum blend and within-n_mut Spearman', '',
          'Rank-sum: rank(score) + rank(ESM2), no parameter. Within-n_mut: ' + m['within_nmut_rule']
          + ' (removes the mutation-count signal; -n_mut scores 0).'
          + (f" Fixed α = {m['fixed_alpha']}: the preprint's α, chosen on the test assays "
             '(reference only).' if fx else ''), '',
          '| predictor | + ESM2 rank-sum | Δ rank-sum vs ESM2 [95% CI] | within-n_mut (alone) |'
          + (' + ESM2 fixed α |' if fx else ''),
          '|---|---|---|---|' + ('---|' if fx else '')]
    for nm in order:
        s = S[nm]
        wn = s['within_nmut']
        if nm in A:
            d = s['delta_ranksum_vs_esm2']
            row = (f"| {disp(nm)} | {f(s['ranksum']['mean'])} | {fs(d['mean'])}{ci(d['ci95'])} | "
                   f"{f(wn['mean'])} (n={wn['n']}) |")
            if fx:
                row += f" {f(s['blend_fixed']['mean'])} |"
        else:
            row = f"| {disp(nm)} | — | — | {f(wn['mean'])} (n={wn['n']}) |" + (' — |' if fx else '')
        L.append(row)
    nd = m.get('n_dropped_by_alignment') or {}
    tot = sum(sum(v.values()) for v in nd.values())
    L += ['', f"Variants dropped by aligning the tags (not scored by every tag): {tot}. "
              f"Exploratory tests (every pair, assay-level Wilcoxon, Holm over all of them): JSON "
              f"key `tests`, not for claims.",
          '', f"assay list sha256 `{m['assay_list_sha256']}`; structure set sha256 "
              f"`{m['structure_set_sha256']}`; CSV set sha256 `{m['csv_set_sha256']}`."]
    return '\n'.join(L) + '\n'


def parse_grid(s: str):
    if ':' in s:
        lo, hi, step = (float(x) for x in s.split(':'))
        n = int(round((hi - lo) / step))
        return tuple(round(lo + k * step, 6) for k in range(n + 1))
    return tuple(float(x) for x in s.split(','))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', action='append', required=True, help='output tag (repeatable)')
    ap.add_argument('--label', action='append', default=[], help='TAG=display label')
    ap.add_argument('--primary', action='append', default=[],
                    help='the chosen Ensembits tag (required for any Ensembits tag: lambda blind)')
    ap.add_argument('--family', default=None,
                    help='pre-specified Holm family, comma-separated items of '
                         f'{sorted(FAMILY_ITEMS)} (default: {",".join(DEFAULT_FAMILY)})')
    ap.add_argument('--n-boot', type=int, default=N_BOOT, help='cluster-bootstrap resamples')
    ap.add_argument('--boot-seed', type=int, default=0)
    ap.add_argument('--name', required=True, help='output stem: pg_d9_<name>.{json,md}')
    ap.add_argument('--assay-list', default=str(C.ASSAY_LIST))
    ap.add_argument('--alpha-grid', default='0:1:0.05', help='lo:hi:step or a,b,c')
    ap.add_argument('--fixed-alpha', type=float, default=0.3,
                    help='also report this alpha (preprint value; chosen on the test assays); '
                         'negative = off')
    ap.add_argument('--score-key', default='score_a')
    ap.add_argument('--out-dir', default=str(PG_OUT))
    ap.add_argument('--allow-provenance-mismatch', action='store_true')
    ap.add_argument('--print-table', action='store_true', help='print the markdown table')
    args = ap.parse_args(argv)
    labels = dict(l.split('=', 1) for l in args.label)
    assays = C.read_assay_list(args.assay_list)
    try:
        res = run(args.method, labels, assays, parse_grid(args.alpha_grid),
                  None if args.fixed_alpha < 0 else args.fixed_alpha, args.primary,
                  args.score_key, Path(args.out_dir), args.allow_provenance_mismatch,
                  family=None if args.family is None else [x.strip() for x in args.family.split(',') if x.strip()],
                  n_boot=args.n_boot, boot_seed=args.boot_seed)
    except (ProvenanceError, BlindError, ValueError) as e:
        print(f'[abort] {e}', file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    C.atomic_write_json(out / f'pg_d9_{args.name}.json', res, indent=1)
    md = to_markdown(res)
    C.atomic_write_text(out / f'pg_d9_{args.name}.md', md)
    print(f'wrote {out / f"pg_d9_{args.name}.json"} and .md')
    if args.print_table:
        print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
