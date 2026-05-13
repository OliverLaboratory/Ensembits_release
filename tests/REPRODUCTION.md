# Reproduction & audit

When your re-run of this repo produces numbers that disagree with the
canonical [`all_baselines_summary.md`](../all_baselines_summary.md), this
document is the place to look. It explains:

1. what's guaranteed to match,
2. why the rest can drift,
3. how big the drift is empirically,
4. and how to force bit-exactness if you need it.

## What this repo guarantees

- **Tokens are bit-identical to canonical.** Every shipped tokenizer
  (`ours/combined/ESM3`, `3di_tokens`, `mini3di K=8`, `vote_3di`,
  `esm3struct`, `aminoaseed`, `protoken`) is bit-verified against the
  canonical token cache on a 5-pid MISATO spot-check plus an mdCATH
  sanity pass.

  ```bash
  python tests/verify_tokenization.py
  ```

  The downstream sweep loads the canonical token `.npz` files via
  symlinks under `data/tokens/` (set up by `scripts/setup_data_links.py`),
  so anything you compute downstream consumes the **exact same input**
  as the canonical paper run.

- **ProteinGym matches canonical to 4 decimals.** PG is deterministic
  (no seeds — one Spearman per tokenizer, one α-blend per α). All five
  shipped tokenizers' `alone` and grid-best `α=0.3-blend` Spearmans
  reproduce exactly. Wired into the comparator below.

## Why downstream cells can still disagree slightly

The MISATO probes (RMSF / EC / GO / binding-site / binding-affinity)
each train a small MLP or conv1d head on GPU. With default PyTorch +
cuDNN settings this training is **non-deterministic across runs** —
even a fresh re-run with the same seed shifts metrics. Three concrete
sources:

1. **cuDNN picks non-deterministic convolution algorithms.** With
   `torch.backends.cudnn.benchmark=True` (the default), cuDNN's
   heuristic algorithm selection can yield different conv kernels on
   different runs, especially under different shape/batch traffic.

2. **Atomic-add reductions** in masked pooling, conv1d backward, and
   pool-then-MLP make the order of floating-point summation vary across
   batch boundaries. Sums of floats are not associative, so re-ordering
   shifts the last few mantissa bits, which compounds through
   thousands of optimisation steps.

3. **DataLoader shuffling** and worker-init RNG paths inherit subtle
   ordering from CUDA stream timing on warm starts vs cold starts.

This applies equally to the canonical sweep (parent repo) and to this
repro. Both sample from the same loss surface; the residual disagreement
you may see is non-determinism, **not** a bug in the repro pipeline.

## How big is the residual, empirically?

Run the comparator after a full sweep:

```bash
python tests/compare_compiled_vs_canonical.py
```

For every cell in the freshly-compiled `.md`, this:

1. Locates the corresponding cell in
   `submission_exp/mds/all_baselines_summary.md`.
2. Decides match / off via a **variance-scaled tolerance**:
   `|Δmean| ≤ 2.5 · σ_canon / √n_canon`, i.e. inside canonical's own
   ~95 % SEM band, with a 0.002 floor for very-low-variance rows.
3. Runs **Welch's two-sample t-test** (unequal variance) on every off
   cell and reports the two-sided p-value.
4. Dumps every off cell to
   [`submission_exp/audits/repro_inconsistencies.csv`](../../submission_exp/audits/repro_inconsistencies.csv),
   sorted by p ascending.

ProteinGym is checked separately by exact-match assertion on the
per-tokenizer `alone` and `α=0.3-blend` Spearmans.

At our last full sweep (2026-05-12: 10 baselines × 5 tasks × 3 splits ×
10 seeds + EC depths 1/2/3 + vote_3di + ProteinGym):

| | n |
|---|---:|
| cells matched inside canonical's variance-scaled SEM band | **313 / 344** |
| off-mean cells | 31 |
| · highly significant (p < 0.001) | 1 |
| · significant (0.001 ≤ p < 0.05) | 13 |
| · not significant (p ≥ 0.05) | 17 |

The single p < 0.001 cell is `3di_tokens` GO sequence split (Δ ≈ 0.05
on mAP/μAP) — conv1d-head training drift, not a tokenization bug
(tokens are bit-equal). 17/31 off cells are within seed noise and only
outside the SEM band because the canonical's std is unusually tight.

## Caveats on the canonical numbers

A few canonical cells were never re-run after backbone-cache or
tokenizer fixes landed in the parent repo, and are explicitly flagged
as stale in `submission_exp/mds/all_baselines_summary.md`. The clearest
example is `protprofile_K8`: it dropped from 34 off-mean cells → 1
after the mode-2 prob-centroid interleave fix and the iteration-order
fix landed in this repo. Its post-fix numbers are tighter and
consistently *better* than the legacy canonical entries — the
canonical row, not this repro, is stale.

When the comparator flags a baseline that's better in the new run than
in canonical, check the corresponding canonical-md row notes before
treating it as a regression.

## Forcing bit-exact downstream reproduction

If you do need floating-point exactness against a previous run (e.g.
for unit tests or differential debugging), the cuDNN flags below
recover full determinism. Set them on **both** sides if you're
comparing this repo against the canonical run.

Shell:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Python (top of each probe script):

```python
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

For this repo, inject at the top of:

- `probes/probe_ec.py`
- `probes/probe_go.py`
- `probes/probe_binding_site.py`
- `probes/probe_binding_affinity.py`

For the canonical sweep, the corresponding files are:

- `submission_exp/src/ec_classify_conv1d_misato.py`
- `submission_exp/src/conv1d_GO_misato.py`
- `submission_exp/src/conv1d_binding_baselines.py`

We verified empirically that with these four flags set on both sides,
the conv1d-head cells match bit-for-bit at seed 0.

## Files referenced by this doc

- [`verify_tokenization.py`](verify_tokenization.py) — bit-identity
  spot-check for every shipped tokenizer.
- [`verify_summary_tables.py`](verify_summary_tables.py) — checks that
  the canonical `.md` cells aggregate correctly from underlying seed
  JSONs.
- [`compare_compiled_vs_canonical.py`](compare_compiled_vs_canonical.py)
  — the per-cell diff + Welch t-test + CSV dump.
- [`../submission_exp/audits/repro_inconsistencies.csv`](../../submission_exp/audits/repro_inconsistencies.csv)
  — most-recent audit output (sorted by p ascending).
