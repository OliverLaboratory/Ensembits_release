# Ensembits — reproduction repo

## What is Ensembits?

**Ensembits** is a Residual VQ-VAE structural tokenizer for protein
**conformational ensembles**. Given a per-residue *set* of local
geometric descriptors across `P` conformations (MD snapshots, predicted
ensembles, etc.), Ensembits compresses each residue to a small tuple of
discrete tokens. The first level (`L_0`, |𝒱|=2048) is the primary
alphabet that downstream probes consume.

This repository is the **reproduction package** for the paper:

- Ships the production tokenizer (`ours/combined/ESM3`, trained on
  combined mdCATH-div + MISATO with the 192-D ESM3 K=16 Affine3D
  descriptor).
- Vendors a clean re-implementation of every baseline tokenizer used in
  the paper (`3di_tokens`, `mini3di K=8` / `vote_3di`, `protprofile_K`,
  `aminoaseed`, `esm3struct`, `protoken`, `aa`, `random`).
- Includes all downstream probes — RMSF, EC, GO, binding-site,
  binding-affinity (ligand-conditioned), and ProteinGym mutation-effects
  — and the sweep driver that produces [`all_baselines_summary.md`](all_baselines_summary.md).

Token caches and downstream JSONs from this repo are bit-/exactly-
identical to the canonical paper run for everything that's
deterministic, and statistically indistinguishable from canonical for
the stochastic conv1d-head probes. The full audit lives in
[`tests/REPRODUCTION.md`](tests/REPRODUCTION.md).

## Repo structure

```text
ensembits/                    # core tokenizer library (model, descriptor, loader)

baselines/                    # baseline tokenizer implementations
├── _backbone.py              # real-bb cache lookup, ideal-geometry recon, chain split, tetrahedral Cβ
├── aa.py / random.py         # sequence-only / null baselines
├── mini3di.py                # Foldseek 3Di tokens, plurality-vote, protprofile_K
├── aminoaseed.py             # StructTokenBench VQ-VAE wrapper
├── esm3struct.py             # ESM3 structure-track encoder wrapper
├── protoken.py               # ProToken-1.0 TF SavedModel wrapper
└── esm2.py                   # ESM2-650M pseudo-LL scorer (PG α-blend only)

probes/                       # downstream evaluators (consume .npz token caches)
├── probe_rmsf.py             # per-residue Cα RMSF MLP probe
├── probe_ec.py               # EC depth 1/2/3 (conv1d head)
├── probe_go.py               # GO top-50 multi-label (conv1d head)
├── probe_binding_site.py     # per-residue BCE → AUROC / AP
├── probe_binding_affinity.py # per-protein -log K_d/K_i (mandatory MACCS ligand)
├── probe_proteingym.py       # PG mutation-effect Spearman + ESM2 α-blend
├── compute_rmsf.py           # build per-residue RMSF labels from an ensemble
├── per_residue_local_pca.py  # 10 Å Cα-ball local-Kabsch s_1 (Token-Dynamics response)
├── build_local_s1_cache.py   # corpus of ensembles → flat s_1 cache
├── anova_token_dynamics.py   # η² of s_1 grouped by token id (no probe fit)
└── _conv1d_head.py           # shared trunk + heads + I/O helpers

scripts/                      # runnable entry points
├── setup_data_links.py       # symlink canonical caches into data/{tokens,codebooks,...}/
├── build_real_bb.py          # per-pid (P, L, 4, 3) bb_K + cb_K cache builder
├── build_baseline_tokens.py  # 3di / vote_3di / aminoaseed / esm3struct / protoken caches
├── build_protprofile_foldseek.py  # 60-D ProtProfile via foldseek+mmseqs
├── build_pg_atom14.py        # ESMFold-fold ProteinGym variants → atom14
├── build_pg_token_cache.py   # per-tokenizer PG wt/mut token caches
├── tokenize_with_ensembits.py     # encode an ensemble with the shipped checkpoint
├── train.py                  # one-shot tokenizer trainer (CLI)
└── run_sweep_and_compile.py  # the sweep driver — re-runs the full grid

tests/                        # audit + verification
├── verify_tokenization.py            # bit-identity vs canonical caches
├── verify_summary_tables.py          # canonical .md cells vs underlying JSONs
├── compare_compiled_vs_canonical.py  # repro .md vs canonical .md (+ t-test, CSV)
└── REPRODUCTION.md                   # numbers-don't-match audit & guidance

ckpt/combined_esm3/           # shipped tokenizer (best.pt + config + stats)
data/                         # symlinks to canonical caches (built by setup_data_links.py)
runs/                         # per-(probe, baseline, split, seed) JSON outputs
all_baselines_summary.md      # compiled output of the sweep driver
```

## Environment setup

Two equivalent ways to set up the env. Both target Python 3.11 and share
[`pyproject.toml`](pyproject.toml) as the source of truth for deps.

Three optional extras toggle baselines that need external repos /
weights:

- `[protoken]` — TensorFlow + JAX (ProToken 1.0 SavedModel)
- `[aminoaseed]` — `omegaconf` (AminoAseed wrapper)
- `[esm2]` — `transformers` (ESM2 pseudo-LL for the PG α-blend; usually
  optional because ProteinGym ships pre-computed scores)

### Option 1 — `uv` (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # install uv

cd ensembits-repro
uv venv --python 3.11
source .venv/bin/activate

uv pip install -e .                                    # core
uv pip install -e ".[protoken,aminoaseed,esm2]"        # everything
```

### Option 2 — conda / mamba

```bash
conda env create -f environment.yml                    # core + pytorch-cuda=12.1
conda activate ensembits-repro
pip install -e ".[protoken,aminoaseed,esm2]"           # extras (optional)
```

Change `pytorch-cuda=12.1` in [`environment.yml`](environment.yml) to
match your driver if needed (`nvidia-smi`).

### External assets (~9 GB)

Token caches, label files, splits, and the shipped checkpoint are
large and live outside the repo. The fastest path is the single
dereferenced zip that mirrors the in-repo `data/` and `ckpt/` layout:

```bash
curl -L -o ensembits_repro_data.zip \
    "https://zenodo.org/records/20152240/files/ensembits_repro_data.zip?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ1YzkxYjI4LTBlY2EtNGM4Mi04ZTE2LTUyYjBlYWJjMWIxMiIsImRhdGEiOnt9LCJyYW5kb20iOiIzYTEzYTBhYTljMTUwZTk4NmI4OTBhYWRkNWI4OWJhYSJ9.qfijHJglR9dlIHWReDW9t9Zo5cVUXI3fAasblenQQ8x8xLoRq96suHLZVKh-vD6_GLN8k3AJR6FSavftn6Jlkg"
unzip ensembits_repro_data.zip      # → populates data/ + ckpt/combined_esm3/
```

> 🗄️ Dataset DOI: [10.5281/zenodo.20152240](https://doi.org/10.5281/zenodo.20152240) — restricted-access record during peer review; the long tokenized URL above is the reviewer-facing share link.

After unzipping you can run every script in the repo directly — no
symlink setup needed.

See [`MANIFEST.md`](MANIFEST.md) for the per-file inventory, raw-source
rebuild instructions, and external-baseline dependencies (AminoAseed +
ProToken repos / checkpoints; foldseek / mmseqs binaries for
`protprofile_K`; etc.). If you have the canonical caches locally on a
shared filesystem,
[`scripts/setup_data_links.py`](scripts/setup_data_links.py) wires them
in via symlinks instead of needing the zip.

## Quick smoke test (all the scripts)

After `setup_data_links.py` succeeds, this set of small commands
exercises every script in the repo on one protein / one seed / one
split:

```bash
# 1. tokens are bit-identical to canonical (5 MISATO pids, all tokenizers)
python tests/verify_tokenization.py

# 2. encode one ensemble end-to-end with the shipped checkpoint
#    (--dataset picks which real_bb cache to read; --limit 1 stops after 1 pid)
python -m scripts.tokenize_with_ensembits \
    --model ckpt/combined_esm3 --dataset mdcath --limit 1 \
    --out /tmp/smoke_tokens.npz

# 3. one RMSF probe seed (≈ 20 s on a single H100)
#    The integer-token cache needs its matching codebook;
#    setup_data_links.py auto-derives data/codebooks/ours_combined_esm3.npy
#    from best.pt's L0 weights.
python -m probes.probe_rmsf \
    --features data/tokens/ours_combined_esm3_misato.npz \
    --codebook data/codebooks/ours_combined_esm3.npy \
    --labels   data/labels/rmsf_misato.npz \
    --splits   data/splits/misato_splits.json --split-name sequence \
    --out /tmp/_smoke_rmsf.json --seed 0

# 4. one ProteinGym Spearman + α-blend
python -m probes.probe_proteingym \
    --wt-tokens  data/pg/wt_tokens/combined_esm3.npz \
    --mut-tokens data/pg/mut_tokens/combined_esm3.npz \
    --codebook   data/pg/codebooks/combined_esm3.npy \
    --dms data/pg/dms_scores.json --esm2 data/pg/esm2_scores.json \
    --out /tmp/_smoke_pg.json

# 5. sweep driver dry-run (enumerate every (probe, baseline, split, seed))
python -m scripts.run_sweep_and_compile --dry-run --depths 1
```

Each step should finish without error and write the expected output
file. If you only have a subset of the external assets (e.g. no
ProteinGym), skip the corresponding step.

## How to train / evaluate

### Train the tokenizer

You bring a per-residue **descriptor cache** of shape
`(N_residues, P, D)` and a `splits.json` mapping `'train'` and `'val'`
to lists of integer indices into the first axis of that cache. The
paper's production recipe:

```bash
python -m scripts.train \
    --desc      data/cached_descriptors/<your-descriptor>.npy \
    --splits    data/splits.json \
    --out       ckpt/<run_name> \
    --codebook-sizes 2048,128,128 \
    --hidden-dim 256 --latent-dim 128 \
    --n-encoder-layers 4 --n-decoder-layers 3 \
    --n-queries 8 --n-heads 4 \
    --batch-size 4096 --lr 1e-3 --weight-decay 1e-5 \
    --random-p-range 1,10 \
    --consistency-mse-weight 0.1 --consistency-distill-fixed-max \
    --warmup-steps 1000 --kmeans-init \
    --max-epochs 300 --patience 40 --eval-every 5 \
    --seed 42
```

Final-run targets on mdCATH-div H-superfamily (~246 k train residues,
~31 k val): best val recon ≈ 0.41, `L_0` utilisation ≈ 93 %, `L_1`/`L_2`
at 100 %. The shipped checkpoint at `ckpt/combined_esm3/` was trained
with the combined mdCATH-div + MISATO corpus and the 192-D ESM3 K=16
Affine3D descriptor.

`python -m scripts.train --help` enumerates every override.

### Encode an ensemble with a trained tokenizer

```bash
# Tokenize every pid in the chosen real_bb cache:
python -m scripts.tokenize_with_ensembits \
    --model ckpt/combined_esm3 --dataset mdcath \
    --out  /tmp/tokens.npz

# Or stop after one pid for a quick check:
python -m scripts.tokenize_with_ensembits \
    --model ckpt/combined_esm3 --dataset mdcath --limit 1 \
    --out  /tmp/smoke_tokens.npz
```

`--dataset` picks the real-bb cache (`data/{mdcath,misato}_real_bb/`).
For single-frame inference (SFTD), add `--p-inference 1`. The output is
an `.npz` keyed by pid → `(L,) int64` primary tokens. The matching
`L0` codebook is at `data/codebooks/ours_combined_esm3.npy` (auto-
derived from `best.pt` by `setup_data_links.py`).

### Run a single downstream probe

Each probe takes a feature cache (`.npz` of `pid → (L,)` int tokens or
`pid → (L, D)` float features), an optional codebook, the labels, and
the split:

```bash
# RMSF (per-residue regression)
python -m probes.probe_rmsf --features data/tokens/<tag>_misato.npz \
    [--codebook data/codebooks/<tag>.npy] \
    --labels  data/labels/rmsf_misato.npz \
    --splits  data/splits/misato_splits.json --split-name structure \
    --out     runs/rmsf_misato/<tag>__structure__seed0.json --seed 0

# EC depth-1
python -m probes.probe_ec --features ... --labels data/labels/misato_ec.json \
    --depth 1 --split-name sequence --out ... --seed 0

# Binding-site, GO, ligand-aware affinity all follow the same pattern;
# see probes/probe_*.py docstrings for required flags.
```

### Re-run the full sweep

```bash
python -m scripts.run_sweep_and_compile --depths 1,2,3
```

Iterates every (baseline × task × split × seed), writes one JSON per
cell under `runs/`, and emits a freshly-compiled
[`all_baselines_summary.md`](all_baselines_summary.md). ~24 GPU-hours
on a single A100/H100; shard across N GPUs with
`--shard I/N` and `CUDA_VISIBLE_DEVICES=i`.

Sub-flags: `--tasks rmsf,ec,go,bs,aff,pg` (subset), `--seeds 0,1`
(seed subset), `--limit 5` (debug cap), `--compile-only` (skip running;
recompile the .md from existing JSONs).

## Reproduction & verification

If your re-run produces numbers that disagree with
[`all_baselines_summary.md`](all_baselines_summary.md), the audit and
the explanation live in **[`tests/REPRODUCTION.md`](tests/REPRODUCTION.md)**.
Short version:

- Tokens are bit-equal; ProteinGym is bit-equal.
- A small residual on conv1d-head probes (≲ 0.05 on micro-AP / top-1)
  is expected from cuDNN non-determinism and is **not** a repro bug.
- Run `python tests/compare_compiled_vs_canonical.py` to get a per-cell
  diff + Welch t-test dumped to
  [`submission_exp/audits/repro_inconsistencies.csv`](../submission_exp/audits/repro_inconsistencies.csv).
- `tests/REPRODUCTION.md` lists the cuDNN flags that recover bit-exact
  determinism if you need it.

## Citation

[paper bibtex placeholder]

## License

MIT (see `LICENSE`).
