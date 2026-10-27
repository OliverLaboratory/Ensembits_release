# Fold-matched pair test (same fold, divergent vs similar dynamics)

Paper: appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Method write-up: `rebuttal/fold_dynamics_mining.md` in the research repo.

## What the test measures

Do tokens carry dynamics beyond fold? We take pairs of BioEmu-CATH domains that share
a fold (TM-score >= 0.7 over >= 50 aligned residues) but are unrelated in sequence
(identity <= 30% over the aligned positions), and split them by how their dominant
collective motions compare:

* **divergent** set: the PC1 motions of A and B are nearly orthogonal (cos1 <= 0.30),
* **similar** set (control): the PC1 motions agree (cos1 >= 0.70) and the covariance
  overlap over 10 modes is >= 0.5.

In both sets each protein's PC1 must reproduce across disjoint groups of independent
runs (>= 0.6), so neither "divergent" nor "similar" is a sampling artifact. Selection is
tokenizer-free (C-alpha coordinates, TM-align, PCA only); tokenizers are applied after
the pairs are fixed.

For each pair and tokenizer we measure, at the TM-aligned positions of the two medoid
structures, how often A and B receive the same token, chance-corrected with the
tokenizer's own collision probability:

    adj = (obs - p) / (1 - p),   p = sum_t f(t)^2 over the floor population

on ALL aligned positions and on the DYNAMIC subset (top quartile of the pairwise-max
per-residue RMSF). The contrast is

    Delta = mean adj(similar) - mean adj(divergent)

with a 95% bootstrap CI over pairs (10,000 resamples, each set resampled independently,
seed 0), and for every baseline b: Delta(Ensembits P=full) - Delta(b) and
Delta(Ensembits P=1) - Delta(b), with the same resampled pairs for both tokenizers.

**Interpretation.** A dynamics-faithful tokenizer agrees MORE on similar-dynamics pairs
than on divergent ones (large positive Delta). A random or noisy alphabet has Delta ~ 0.
Because the fold is matched in both sets, a fold-only tokenizer should also show a small
Delta; a positive Ensembits-minus-baseline difference whose CI excludes 0 means Ensembits
separates the two dynamics regimes more than that baseline.

## Inputs given to each tokenizer

The same for the pair systems and the floor population:

| tokenizer | input |
|---|---|
| 3Di (mini3di) | medoid frame, real N/CA/C, mini3di's virtual (tetrahedral) C-beta from the real N/CA/C |
| AminoAseed, ESM3-struct | medoid frame, real N/CA/C (`tokenize_protein(..., bb=...)` of `data_prep/baselines/`) |
| ProToken | medoid frame, real N/CA/C/O, topology sequence |
| Ensembits P=1 | medoid frame, real N/CA/C |
| Ensembits P=full | the 10 standard frames `bb_10` (`data_prep/build_bioemu_real_bb.py`) |

Medoid = the frame of run 1 closest to the run-1 mean C-alpha (rule of the original
analysis; superposition was added in the port, FIXED(C15)). The floor
population is every system of `data/cached_descriptors/bioemu_cath_real_bb/` (or a seeded
`--floor-limit` subset); only systems tokenized by every tokenizer enter the floor.

## Run order

From the codebase root (`ensembits_code/`):

```
python downstream/fold_pairs/run_fold_pairs.py --dry-run          # prints the commands below
```

0. prerequisite: `python data_prep/build_bioemu_real_bb.py --root <REPO_ROOT>/data/reviewer_3_special_corpus/bioemu_cath/extracted --out <REPO_ROOT>/data/cached_descriptors/bioemu_cath_real_bb` (or `run_fold_pairs.py --with-real-bb`)
1. `bioemu_extract_ca.py --workers 32 --per-run 30`: C-alpha cache, one file per system
2. `bioemu_fold_dynamics_miner.py --workers 64 --len-tol 0.15 --select divergent`: scores every length-compatible pair once, writes the top 25 divergent candidates
3. `bioemu_fold_dynamics_miner.py --select similar`: reuses the scores, top 50 similar candidates
4. `verify_fold_dynamics_pair.py --mode divergent` and `--mode similar`
5. `bioemu_token_tokenize.py --population pairs`
6. `bioemu_token_tokenize.py --population floor [--floor-limit N]`
7. `bioemu_token_compare.py`

Steps 1 to 4 are CPU only and need mdtraj and TM-align (`ensembits.paths.TMALIGN_BIN`:
env `ENSEMBITS_TMALIGN`, else `TMalign` or `USalign` on PATH). Steps 5 and 6 need a GPU
and the tokenizer weights; `--tokenizers protoken` lets ProToken (TensorFlow + JAX) run
in its own environment.

## Outputs (under `REPO_ROOT/output/fold_pairs/`)

| file | content |
|---|---|
| `bioemu_ca/<system>.npz` | `ca` (F, L, 3), `runlens`, `system`, `pid`, `deposit`, `n_full_bb` |
| `scored_pairs.json` | `{meta, records}`: every scored pair (TM, ncommon, cos1..3, rmsip, fvarA/B) |
| `candidates_{divergent,similar}.json` | filters, ranking, top-N pairs sent to verification |
| `verification_{divergent,similar}.json` | thresholds and per-pair verification records with `valid` |
| `medoids/<system>.npz` | medoid real backbone `bb` (L, 4, 3), `ca`, `seq`, `frame`, `run` |
| `real_bb_fallback/<system>.npz` | `bb_10` computed for systems missing from the standard cache |
| `tokens/{pairs,floor}/<tokenizer>.npz`, `manifest.json` | system -> (L,) tokens; inputs and failures |
| `token_comparison.json` | floor collision probabilities, per-set rows (ALL, DYNAMIC), per-pair rows, Delta contrast with CIs |

## Files

| file | origin (research repo) |
|---|---|
| `_fp_common.py` | new (paths, system / domain helpers) |
| `bioemu_extract_ca.py` | `submission_exp/src/rebuttal_analyses/bioemu_extract_ca.py` |
| `fold_dynamics_miner.py` | `submission_exp/src/rebuttal_analyses/fold_dynamics_miner.py` (numeric core only) |
| `bioemu_fold_dynamics_miner.py` | `submission_exp/src/rebuttal_analyses/bioemu_fold_dynamics_miner.py` |
| `verify_fold_dynamics_pair.py` | `submission_exp/src/rebuttal_analyses/verify_fold_dynamics_pair.py` |
| `bioemu_token_tokenize.py` | `submission_exp/src/rebuttal_analyses/bioemu_token_tokenize.py` |
| `bioemu_token_compare.py` | `submission_exp/src/rebuttal_analyses/bioemu_token_compare.py` |
| `run_fold_pairs.py` | new (orchestrator) |

Approved changes are marked `FIXED(code_fix C15, 2026-10-15)` in the code; open
design caveats are marked `Caveat`.

## Caveats

* Pairs can share systems, so pairs are not independent samples; the bootstrap over
  pairs treats them as such.
* The ensembles are the BioEmu-CATH simulations (GROMACS trajectories of the Zenodo
  release). The Ensembits checkpoint was trained on BioEmu-CATH (including the bb_10
  frames of these systems); the baselines were not.
* The thresholds (0.6, 0.30, 0.70, 0.5, 0.30) are fixed before tokenization and are CLI
  options of the verifier.
* Ensembits numbers depend on the checkpoint (`--model-dir`); re-run after the retrain.
