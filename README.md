# Ensembits

Code for the paper "Ensembits: tokenizing protein conformational ensembles" (NeurIPS 2026).

Ensembits is a learned tokenizer that maps a variable-size ensemble of protein conformations
(P backbone frames per residue) to a single discrete token per residue via a residual
vector quantizer (RVQ). The token captures the conformational diversity at each site and
can be decoded back to P backbone frames.

**Reported model:** BioEmu-CATH + MISATO corpus, 256-D descriptor, RVQ [2048, 128, 128],
SFTD with lambda = 10. Checkpoint in `ckpt/bioemu_cath_misato_lam10/`.

## Installation

```bash
pip install -r requirements.txt
```

Python >= 3.10 is required. Core dependencies: PyTorch, numpy, scipy, h5py, mdtraj,
matplotlib, scikit-learn. Optional: `esm` (ESM3-struct baseline only), `gemmi`
(CryptoBench structure stage only).

## Layout

```text
ensembits/                 core library (model, descriptor, losses, loading)
  descriptor.py            256-D per-residue, per-frame descriptor
  geometry.py              backbone frames, kNN, relative frames
  featurize.py             load checkpoint, tokenize ensembles
  paths.py                 CODE_ROOT, REPO_ROOT, CKPT_DIR
  tokenizer/
    vqvae.py               SetTransformerEncoder, SetDecoder, VectorQuantizer
    rvqvae.py              RVQVAETokenizer (the reported model)
train/                     trainer and launcher
  train.py                 training loop
  train_bioemu_misato.sh   launcher with the reported recipe
data_prep/                 build real-backbone caches, labels, training corpus
  baselines/               baseline token/feature cache builders
inference/                 tokenize any ensemble (PDB or .npy)
downstream/                downstream tasks: RMSF, EC, GO, binding, affinity, ProteinGym
  analyses/                token ANOVA, codebook t-SNE, figures
  fold_pairs/              fold-matched pair test
  cryptobench/             cryptic binding site benchmark
train/retrain_static/      StructTokenBench static-baseline retraining
tests/                     pytest suite
checks/                    reproduction checks
ckpt/bioemu_cath_misato_lam10/  reported checkpoint
output/figures/            rendered token exemplar figures (PDB/PNG/PDF)
```

## Data

Data and outputs live outside this folder, at `REPO_ROOT` (default: two levels up;
override with `ENSEMBITS_REPO_ROOT`). Third-party inputs go under
`REPO_ROOT/data/external/` (mdCATH-div, ProteinGym, StructTokenBench), each overridable
by an environment variable in `ensembits/paths.py`.

**PDBbind-derived labels** (binding site, affinity): the label files cannot be
redistributed. Use the builders to regenerate them from PDBbind source data:
- `python data_prep/build_misato_binding_site_labels.py`
- `python data_prep/build_misato_affinity_labels.py`

## Quick start

### Tokenize an ensemble

```bash
python inference/quantize_ensemble.py \
    --pdb my_ensemble.pdb \
    --model-dir ckpt/bioemu_cath_misato_lam10 \
    --out tokens.npz
```

### Train (GPU)

The exact command that trained the reported lambda = 10 checkpoint:

```bash
RUN=ensembits_bm_lam10 LAMBDA=10 bash train/train_bioemu_misato.sh
```

This reads the descriptor cache from `output/bioemu_cath_misato_e0/cached_desc/`
(built by `data_prep/build_bioemu_misato_descriptor.py`). Flags: SFTD with
lambda = 10, IUPAC dihedrals, Hungarian matching on the loss, standard RVQ residuals,
full-branch codebook updates, k-means init. See the script header for all options
(SFTD=off, LAMBDA, EXTRA, DRYRUN, RESUME).

### Run the downstream battery

```bash
python downstream/_run_model_battery.py \
    --variant ensembits_bm_lam10_rv \
    --pgtag rv_lam10 \
    --gpus 0,1,2,3
```

### Tests

```bash
python -m pytest tests/ -q
```

## Reproducing tables and figures

### Main tables (function tasks, RMSF, ProteinGym)

```bash
# Compile all results into a markdown table
python downstream/_compile_rebuttal_results.py \
    --variant ensembits_bm_lam10_rv \
    --out camera_ready/rv_valmetrics_lam10.md

# Select camera-ready rows
python downstream/compile_camera_ready_tables.py \
    --variant ensembits_bm_lam10_rv \
    --src camera_ready/rv_valmetrics_lam10.md \
    --out camera_ready/camera_ready_tables.md
```

### Ablation tables (VQ and SFTD)

```bash
# Reconstruction loss at different P
python downstream/analyses/recon_by_p.py --device cuda:0
# Token usage (CPU)
python downstream/analyses/token_usage_by_p.py
# The compiler reads these automatically (defaults: _rv files)
```

### Token ANOVA

```bash
# Chosen lambda = 10
python downstream/analyses/token_anova.py \
    --out output/token_anova/token_anova_rv.json

# All lambda variants
python downstream/analyses/token_anova.py \
    --variants lambda_sweep \
    --out output/token_anova/token_anova_rv.json
```

### Codebook t-SNE

```bash
python downstream/analyses/plot_codebook_tsne.py \
    --counts-cache output/codebook_tsne/counts_lam10.json \
    --out output/codebook_tsne/tSNE_codes.png
```

### Function dot plot (replaces radar)

```bash
python downstream/analyses/plot_function_dotplot.py
```

### Token exemplar figures

```bash
python downstream/analyses/token_exemplar_export.py \
    --variants output/ensembits_bm_lam10_rv \
    --device cuda:0
```

### Fold-pair test

```bash
python downstream/fold_pairs/run_fold_pairs.py \
    --model-dir output/ensembits_bm_lam10_rv \
    --dry-run
```

### CryptoBench

```bash
# See downstream/cryptobench/README.md for the full pipeline:
# build apo structures, tokenize, featurize, probe, aggregate
python downstream/cryptobench/run_cryptobench_rv.py --gpus 0,1,2,3
```

### Per-protein significance tests

```bash
python downstream/protein_level_stats.py \
    --family rv_protein_stats_val_family.json \
    --split test \
    --out output/protein_stats_test.json
```

## Reproduction checks

```bash
python checks/check_tokens_match_original.py     # descriptor + tokens match original pipeline
python checks/check_dihedral_reparam.py           # dihedral reparametrisation is exact
python checks/check_geometry_vs_esm.py            # geometry.py matches ESM3 utilities
python checks/check_matching.py                   # loss vs L2 matching comparison
```

## RMSF pipeline (full run order)

The RMSF probes require real-backbone caches, RMSF labels, residue sets, baseline
caches and descriptor caches. The full build order is documented in
`downstream/analyses/README.md` (section "Run order after the retrain").

Key steps:
1. Build MISATO K=8 backbones: `python data_prep/build_misato_real_bb.py --K 8`
2. Build RMSF labels (all MD frames): `build_{mdcath_div,misato,bioemu}_rmsf_labels.py`
3. Build common residue sets: `build_rmsf_residue_sets.py --dataset {mdcath_div,misato,bioemu}`
4. Build descriptor caches: `build_esm3desc_dihedral_perdataset.py --dataset {mdcath,bioemu}`
5. Build baseline caches: `build_mini3di_misato.py`, `build_aminoaseed_tokens.py`, etc.
6. Run Ensembits probes: `_run_model_battery.py --variant <run> --tasks PRETOK,RMDIV,RMMIS,RMBIO`
7. Run baseline probes: `baselines/run_misato_baselines.py --tasks rmsf_misato,rmsf_div,rmsf_bioemu`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ENSEMBITS_REPO_ROOT` | two levels above this folder | data/ and output/ location |
| `ENSEMBITS_EXTERNAL` | `$REPO_ROOT/data/external` | third-party inputs |
| `CB_GEMMI_PYTHON` | `python` | Python with gemmi for CryptoBench |
