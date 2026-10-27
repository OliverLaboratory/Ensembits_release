"""Compare tokenizers on fold-matched pairs: token agreement on divergent vs similar dynamics.

What it does
------------
For each VALID pair of `verification_divergent.json` and of `verification_similar.json`:

  1. TM-align the two medoid structures (C-alpha of the representative frames used for
     tokenization) -> residue correspondence. Pairs with < 30 aligned residues are
     dropped (as in the original).
  2. Two position sets over the aligned residues:
       ALL      every aligned position;
       DYNAMIC  aligned positions whose pairwise-max per-residue RMSF (C-alpha cache,
                frames superposed onto frame 0) is in the top quartile.
  3. Per tokenizer: raw agreement obs = fraction of positions where A and B get the same
     token, and the chance-adjusted (Cohen-style) agreement
         adj = (obs - p) / (1 - p),   p = sum_t f(t)^2,
     where f(t) is the token frequency over the FLOOR population
     (`bioemu_token_tokenize.py --population floor`): the same systems and the same
     kind of input for every tokenizer. Only systems tokenized by all tokenizers enter
     the floor. adj = 0 is chance for that alphabet, 1 = identical tokens everywhere.
  4. Contrast per tokenizer: Delta = mean adj(similar) - mean adj(divergent), with a 95%
     bootstrap CI over pairs (10,000 resamples, pairs resampled within each set
     independently, seed 0). For each baseline b and each Ensembits mode e (P=full,
     P=1): Delta(e) - Delta(b), with the CI from the SAME resampled pairs for both
     tokenizers (paired bootstrap). Done for ALL and DYNAMIC.

Interpretation: a dynamics-faithful tokenizer agrees MORE on similar-dynamics pairs than
on divergent ones (large positive Delta); a random or noisy alphabet has Delta ~ 0. The
fold is matched in both sets (TM >= 0.7, seq_id <= 0.30), so Delta isolates dynamics.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Section 5 of `rebuttal/fold_dynamics_mining.md` (tokenizer comparison).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/bioemu_token_compare.py`. Kept: the
TM-align correspondence on the medoids, the ALL / DYNAMIC sets (top quartile of the
pairwise-max RMSF), `rmsf`, raw and chance-adjusted agreement, the collision
probability `coll` (now `chance_collision`, same Counter computation). Changed (FIXED
code_fix C15 markers): the floor comes from one shared population tokenized for this
test (the original read mdCATH caches for the baselines and the MISATO cache for
Ensembits); both pair sets and the Delta contrast with paired bootstrap CIs; 3Di tokens
come from the tokenization step (real backbone) instead of being computed here from
C-alpha; systems instead of domains; paths from `ensembits.paths.REPO_ROOT`.
Fixes 2026-10-16: the Ensembits token files of both populations must come from one
checkpoint (code_fix P17).

Used by the reported model: yes (Ensembits P=1 / P=full rows, reported checkpoint).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/fold_pairs/verification_{divergent,similar}.json
          output/fold_pairs/medoids/<system>.npz
          output/fold_pairs/bioemu_ca/<system>.npz
          output/fold_pairs/tokens/{pairs,floor}/<tokenizer>.npz, manifest.json
  writes  output/fold_pairs/token_comparison.json

How to run
----------
    python downstream/fold_pairs/bioemu_token_compare.py [--n-boot 10000] [--seed 0]
"""
import os, sys, json, argparse, collections
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import fold_dynamics_miner as M   # noqa: E402  tmalign, internal_align (pins 1 BLAS thread)
import numpy as np                 # noqa: E402
import _fp_common as C             # noqa: E402

# (display name, token key, codebook size)
TOKS = [('3Di', '3di', 20), ('AminoAseed', 'aminoaseed', 512), ('ESM3-struct', 'esm3struct', 4096),
        ('ProToken', 'protoken', 512), ('Ensembits P=1', 'ens_P1', 2048),
        ('Ensembits P=full', 'ens_Pfull', 2048)]
ENS_NAMES = ('Ensembits P=full', 'Ensembits P=1')
DYN_Q = 0.75  # top quartile of pairwise-max RMSF = "dynamic" residues
N_BOOT = 10000
SEED = 0
INTERPRETATION = (
    'Delta = mean adjusted agreement on similar-dynamics pairs minus that on divergent-dynamics '
    'pairs (both sets fold-matched: TM >= 0.7, seq_id <= 0.30). A dynamics-faithful tokenizer '
    'agrees MORE on similar-dynamics pairs than on divergent ones (large positive Delta); a random '
    'or noisy alphabet has Delta ~ 0. Delta(Ensembits) - Delta(baseline) > 0 with a CI excluding 0 '
    'means Ensembits separates the two sets more than that baseline (paired bootstrap over pairs).')


# ---------------------------------------------------------------- statistics
def chance_collision(arrays):
    """Collision probability p = sum_t f(t)^2 of the pooled tokens (the original `coll`)."""
    cnt = collections.Counter()
    for a in arrays:
        cnt.update(np.asarray(a).ravel().tolist())
    tot = sum(cnt.values()); return sum((x / tot) ** 2 for x in cnt.values())


def adjusted_agreement(obs, p):
    """Cohen-style chance-adjusted agreement (obs - p) / (1 - p)."""
    return (obs - p) / (1 - p)


def bootstrap_contrast(adj_sim, adj_div, names, n_boot=N_BOOT, seed=SEED, ens_names=ENS_NAMES):
    """Delta = mean(adj_sim) - mean(adj_div) per tokenizer with percentile 95% CIs.

    adj_sim / adj_div: {name: (n_sim,) / (n_div,) per-pair adjusted agreements, the
    same pairs in the same order for every tokenizer}. Each bootstrap draw resamples
    the similar pairs and the divergent pairs independently (with replacement); the
    SAME draw is used for every tokenizer, so differences between tokenizers are
    paired. Returns a dict with per-set means, Delta per tokenizer, and
    Delta(e) - Delta(b) for every Ensembits mode e in `ens_names` and baseline b.
    """
    S = np.stack([np.asarray(adj_sim[n], dtype=np.float64) for n in names])   # (T, n_sim)
    D = np.stack([np.asarray(adj_div[n], dtype=np.float64) for n in names])   # (T, n_div)
    ns, nd = S.shape[1], D.shape[1]
    rng = np.random.default_rng(seed)
    idx_s = rng.integers(0, ns, size=(n_boot, ns))
    idx_d = rng.integers(0, nd, size=(n_boot, nd))
    ms = S[:, idx_s].mean(-1)                    # (T, n_boot) resampled set means
    md = D[:, idx_d].mean(-1)
    bd = ms - md                                 # (T, n_boot) resampled Delta

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = {'n_similar': ns, 'n_divergent': nd, 'n_boot': n_boot, 'seed': seed,
           'per_tokenizer': {}, 'ensembits_minus_baseline': {}}
    for t, n in enumerate(names):
        out['per_tokenizer'][n] = {
            'mean_similar': float(S[t].mean()), 'ci_similar': ci(ms[t]),
            'mean_divergent': float(D[t].mean()), 'ci_divergent': ci(md[t]),
            'delta': float(S[t].mean() - D[t].mean()), 'ci_delta': ci(bd[t]),
            'p_boot_delta_le_0': float((bd[t] <= 0).mean())}
    for e in ens_names:
        if e not in names:
            continue
        ie = names.index(e)
        for b in names:
            if b in ens_names:
                continue
            ib = names.index(b)
            diff = bd[ie] - bd[ib]
            out['ensembits_minus_baseline'][f'{e} - {b}'] = {
                'diff': float((S[ie].mean() - D[ie].mean()) - (S[ib].mean() - D[ib].mean())),
                'ci': ci(diff), 'p_boot_le_0': float((diff <= 0).mean())}
    return out


# ---------------------------------------------------------------- data
def rmsf(system):
    ca = np.load(os.path.join(C.CA_CACHE, system + '.npz'))['ca'].astype(np.float64)  # (F,L,3)
    A = M.internal_align(ca)   # superpose all frames to frame-0
    return np.sqrt(((A - A.mean(0)) ** 2).sum(2).mean(0))  # (L,) per-residue Ca std


def load_tokens(tok_dir, population, keys):
    out = {}
    for k in keys:
        p = Path(tok_dir) / population / f'{k}.npz'
        if not p.exists():
            raise FileNotFoundError(f'{p} missing; run bioemu_token_tokenize.py --population {population}')
        d = np.load(p)
        out[k] = {s: d[s] for s in d.files}
    return out


def valid_pairs(path):
    obj = json.load(open(path))
    recs = obj['pairs'] if isinstance(obj, dict) else obj
    return [r for r in recs if r['valid']]


def pair_metrics(r, T, toks, p_chance, RM, medoid_ca):
    """Per-tokenizer raw / adjusted agreement of one pair on ALL and DYNAMIC, or None."""
    a, b = r['A'], r['B']
    res = M.tmalign(medoid_ca[a].astype(np.float64), medoid_ca[b].astype(np.float64))
    if res is None or len(res[3]) < 30:
        return None, 'TM-align on medoids failed or < 30 aligned'
    iA = np.array([c[0] for c in res[3]]); iB = np.array([c[1] for c in res[3]])
    rmax = np.maximum(RM[a][iA], RM[b][iB])
    thr = np.quantile(rmax, DYN_Q); dyn = rmax >= thr
    row = {'A': a, 'B': b, 'tm_medoid_max': round(res[1], 3), 'n_aligned': int(len(iA)),
           'n_dynamic': int(dyn.sum())}
    for name, key, _ in toks:
        ta = T[key][a][iA]; tb = T[key][b][iB]; eq = (ta == tb)
        ch = p_chance[name]
        ag = float(eq.mean()); agd = float(eq[dyn].mean())
        row[name] = {'raw_all': ag, 'adj_all': adjusted_agreement(ag, ch),
                     'raw_dyn': agd, 'adj_dyn': adjusted_agreement(agd, ch)}
    return row, None


def summarise(rows, toks, p_chance, field):
    out = []
    for name, _, cbk in toks:
        v = np.array([r[name]['adj_' + field] for r in rows]); rw = np.array([r[name]['raw_' + field] for r in rows])
        out.append({'tokenizer': name, 'codebook': cbk, 'emp_chance': round(p_chance[name], 5),
                    'raw_agree': round(float(rw.mean()), 4) if len(rw) else None,
                    'adj_agree': round(float(v.mean()), 4) if len(v) else None,
                    'adj_std': round(float(v.std()), 4) if len(v) else None})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tokenizers', default=','.join(k for _, k, _ in TOKS),
                    help='comma list of token keys to compare (default all)')
    ap.add_argument('--tokens-dir', default=str(C.TOKENS_DIR))
    ap.add_argument('--divergent', default=str(C.verification_path('divergent')))
    ap.add_argument('--similar', default=str(C.verification_path('similar')))
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=str(C.COMPARISON))
    args = ap.parse_args()
    keys = [k for k in args.tokenizers.split(',') if k]
    toks = [t for t in TOKS if t[1] in keys]
    names = [t[0] for t in toks]

    # 1. Chance floor from one shared population.
    # FIXED(code_fix C15, 2026-10-15): old: p from each tokenizer's own cache of a different
    #   dataset (mdCATH frame-0 caches for the baselines, the MISATO cache for both
    #   Ensembits modes). New: p from the floor population of this test, restricted to
    #   the systems that every compared tokenizer tokenized, so all floors come from the
    #   same proteins and the same single-structure / bb_10 inputs as the pair tokens.
    F = load_tokens(args.tokens_dir, 'floor', keys)
    floor_sys = sorted(set.intersection(*[set(F[k]) for k in keys]))
    if not floor_sys:
        raise SystemExit('no system was tokenized by every tokenizer in the floor population')
    p_chance, floor_info = {}, {}
    for name, key, cbk in toks:
        p = chance_collision([F[key][s] for s in floor_sys])
        p_chance[name] = p
        floor_info[name] = {'p_chance': p, 'effective_alphabet': 1.0 / p, 'uniform_1_over_K': 1.0 / cbk,
                            'n_systems_tokenized': len(F[key]),
                            'n_residues': int(sum(len(F[key][s]) for s in floor_sys))}
    print(f'[fold-pairs-cmp] floor: {len(floor_sys)} systems common to {len(keys)} tokenizers', flush=True)

    # 2. Per-pair agreement in both sets.
    T = load_tokens(args.tokens_dir, 'pairs', keys)
    sets = {'divergent': valid_pairs(args.divergent), 'similar': valid_pairs(args.similar)}
    systems = sorted({s for V in sets.values() for r in V for s in (r['A'], r['B'])})
    RM, medoid_ca = {}, {}
    for s in systems:
        mp = C.MEDOID_DIR / f'{s}.npz'
        if mp.exists():
            medoid_ca[s] = np.load(mp)['ca']
        RM[s] = rmsf(s)
    result = {'interpretation': INTERPRETATION, 'dyn_quartile': DYN_Q,
              'floor': {'n_systems': len(floor_sys), 'per_tokenizer': floor_info},
              'sets': {}, 'contrast': {}}
    man_p = Path(args.tokens_dir) / 'pairs' / 'manifest.json'
    if man_p.exists():
        result['tokenizer_inputs'] = {k: v.get('input') for k, v in json.load(open(man_p)).get('tokenizers', {}).items()}
    # FIXED(code_fix P17, 2026-10-16): new. The Ensembits token files of both populations
    #   must come from one checkpoint (their <file>.provenance.json); recorded in the output.
    ckpts = {}
    for pop in ('pairs', 'floor'):
        for k in keys:
            side = Path(args.tokens_dir) / pop / f'{k}.npz.provenance.json'
            if k.startswith('ens_') and side.exists():
                ckpts[f'{pop}/{k}'] = json.load(open(side)).get('checkpoint')
    shas = {(c or {}).get('best_pt_sha256') for c in ckpts.values()}
    if len(shas) > 1:
        raise SystemExit(f'Ensembits token files come from different checkpoints: '
                         f'{ {k: (c or {}).get("variant") for k, c in ckpts.items()} }; re-tokenize')
    if ckpts:
        result['checkpoint'] = next(iter(ckpts.values()))
    per_set_rows = {}
    for sname, V in sets.items():
        rows, dropped = [], []
        for r in V:
            a, b = r['A'], r['B']
            miss = [k for k in keys if a not in T[k] or b not in T[k]]
            if miss or a not in medoid_ca or b not in medoid_ca:
                dropped.append({'A': a, 'B': b, 'why': f'missing tokens/medoid ({miss})'}); continue
            if any(len(T[k][s]) != len(RM[s]) for k in keys for s in (a, b)):
                dropped.append({'A': a, 'B': b, 'why': 'token length != C-alpha cache length'}); continue
            row, why = pair_metrics(r, T, toks, p_chance, RM, medoid_ca)
            if row is None:
                dropped.append({'A': a, 'B': b, 'why': why}); continue
            rows.append(row)
        per_set_rows[sname] = rows
        result['sets'][sname] = {'n_valid_pairs': len(V), 'n_pairs': len(rows), 'dropped': dropped,
                                 'rows': summarise(rows, toks, p_chance, 'all'),
                                 'rows_dynamic': summarise(rows, toks, p_chance, 'dyn'),
                                 'pairs': rows}
        print(f'[fold-pairs-cmp] {sname}: {len(rows)} pairs ({len(dropped)} dropped)', flush=True)

    # 3. Delta contrast with paired bootstrap CIs.
    # Caveat: pairs share systems (the original 20 divergent pairs span 34 domains), so
    #   resampling pairs as if independent understates the CI; a cluster bootstrap over
    #   systems would be more conservative. p_chance is treated as fixed.
    if per_set_rows['similar'] and per_set_rows['divergent']:
        for field, label in (('all', 'ALL'), ('dyn', 'DYNAMIC')):
            adj_s = {n: [r[n]['adj_' + field] for r in per_set_rows['similar']] for n in names}
            adj_d = {n: [r[n]['adj_' + field] for r in per_set_rows['divergent']] for n in names}
            result['contrast'][label] = bootstrap_contrast(adj_s, adj_d, names, args.n_boot, args.seed)
    else:
        result['contrast'] = 'not computed: a set has no usable pair'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, 'w'), indent=1)
    print(f'[fold-pairs-cmp] -> {args.out}')
    print(INTERPRETATION)
    if isinstance(result['contrast'], dict):
        c = result['contrast']['ALL']['per_tokenizer']
        print(f'{"tokenizer":<18}{"adj(sim)":>10}{"adj(div)":>10}{"Delta":>9}  95% CI')
        for n in names:
            x = c[n]
            print(f'{n:<18}{x["mean_similar"]:>10.3f}{x["mean_divergent"]:>10.3f}{x["delta"]:>9.3f}'
                  f'  [{x["ci_delta"][0]:.3f}, {x["ci_delta"][1]:.3f}]')
        for k, x in result['contrast']['ALL']['ensembits_minus_baseline'].items():
            print(f'  {k:<36} {x["diff"]:+.3f}  [{x["ci"][0]:+.3f}, {x["ci"][1]:+.3f}]')


if __name__ == '__main__':
    main()
