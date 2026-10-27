# downstream/baselines: MISATO baseline probes

Probes that produce the baseline rows of the EC, GO, binding site and binding
affinity tables (`appendix.tex` app:downstreams, app:experiments) and the
launcher that also runs the baseline RMSF probes (`2_results.tex` result:rmsf).
They use the same heads and training protocol as the Ensembits probes in
`downstream/tokenize_misato_*.py`, on fixed per-residue baseline features built
by `data_prep/baselines/`. Each file has a header with its paper reference and
origin (`submission_exp/src/...`). Outputs go to `REPO_ROOT/output/`.

| File | Writes | Notes |
|---|---|---|
| `ec_classify_conv1d_misato.py` | `output/ec_conv1d_misato/<feature>[__<split>][__<tag>][__seed<s>]__{d1,d2,d3,summary}.json` | defines `load_features` (all baseline features), imported by the other two |
| `conv1d_GO_misato.py` | `output/go_conv1d_misato/go_<feature>_<scope>[_<split>][_<tag>][_seed<s>].json` | GO labels and class selection from `downstream/go_labels.py` (MF v2) |
| `conv1d_binding_baselines.py` | `output/binding_misato/baseline_<feature>[_aa][_lig][_<tag>][_seed<s>].json` | schedule and splits from `downstream/tokenize_misato_binding_eval.py` |
| `run_misato_baselines.py` | all of the above with `--tag stab`, plus `output/probe_rmsf_misato/baseline_*__test.json` and `output/probe_rmsf_div/baseline_*__test.json` | GPU pool, skips existing outputs, `--dry-run` (Q4: rescue opt-in via `--tasks ...,rescue`) |

Inputs (2026-10-19): every probe standardises the head input by default, with the same rule
as the Ensembits probes (`downstream/function_common.py`): lookup features (aa one-hot, the
AminoAseed / ESM3struct / ProToken codebooks, the mini3di centroids of 3Di / Vote_3Di) through
their table standardised per dimension over its rows; continuous features (ProtProfileMD,
random, PCA eigenvalues, pooled features) per dimension over the residues of the training
items of each split / CV rotation; the MACCS ligand input unchanged. `--no-input-standardisation`
feeds the old raw inputs (tag + `rawin`; refused by the compilers). The temporary features
`aminoaseed_std` / `esm3struct_std` were removed (now the default of `aminoaseed` /
`esm3struct`). `load_features_checked` returns the rule as a fourth value.

## Run order

1. Build the caches (`data_prep/baselines/README.md`).
2. `python downstream/baselines/run_misato_baselines.py --dry-run` to see the jobs, then
   `python downstream/baselines/run_misato_baselines.py --gpus 0,1,2,3`:
   EC / GO / BIND with the stabilised recipe (`--tag stab --warmup 5 --min-epochs 20
   --patience 12 --cosine`; binding with `--task both --use-ligand-maccs`), the MISATO
   and mdCATH-div RMSF baseline probes (Q4: no seed rescue by default; add
   `rescue` to `--tasks` to opt in).
3. Compile the tables: `python downstream/_compile_rebuttal_results.py`.

A single probe can be run on its own, e.g.

```bash
python downstream/baselines/ec_classify_conv1d_misato.py --feature esm3struct --split structure \
    --seed 0 --depths 1,2,3 --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine
```
