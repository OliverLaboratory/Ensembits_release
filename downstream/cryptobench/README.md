# CryptoBench: cryptic binding sites from a single apo structure

Per-residue prediction of **cryptic** ligand binding sites (pockets that are closed or
malformed in the ligand-free structure and open on binding). Input: one apo structure
(backbone N/CA/C, so P = 1). Labels: residues that contact the ligand in the matched holo
structures. A frozen representation (Ensembits level-0 code vector, or a baseline
tokenizer's code vector) feeds a small per-residue probe.

Status: **not yet in the paper**. The authors decided (2026-10-15) that it goes into the
main text: `camera_ready/require_fix.md` #24 (text) and `camera_ready/code_fix.md` E3
(re-run with the reported checkpoint and open checks). Status 2026-10-22: the camera-ready
port is done and re-audited (`camera_ready/cryptobench_audit/PORT.md`, `REAUDIT_CB1..CB6*.md`,
`FIXLIST_20261022.md`); the full run (`run_cryptobench_rv.py`) has not been launched yet.

## Benchmark (external, not copied here)

- Paper: V. Škrhák, M. Novotný, C. P. Feidakis, R. Krivák, D. Hoksza, *CryptoBench: cryptic
  protein–ligand binding sites dataset and benchmark*, Bioinformatics 41(1):btae745, 2025,
  doi:[10.1093/bioinformatics/btae745](https://doi.org/10.1093/bioinformatics/btae745).
- Data: OSF project [pz4a9](https://osf.io/pz4a9/) (doi:10.17605/OSF.IO/PZ4A9). The local
  copy was downloaded from `https://osf.io/download/c5vp8/`.
- Code: <https://github.com/skrhakv/CryptoBench>, MIT licence (© 2024 Vít Škrhák). A clone
  sits in the research repo at `rebuttal/CryptoBench/` (commit `9a3432f`, 2026-04-30). None
  of its code is used by this pipeline; it documents how the dataset was built and has a
  tutorial notebook. Only the OSF data files are used.

Data needed under `REPO_ROOT/data/cryptobench/` (override with `--cryptobench-dir`):
`dataset.json` (apo id → holo partners, chains, pocket residues, ligand, pRMSD,
`is_main_holo_structure`), `folds.json` (`test`, `train-0` … `train-3`; apo ids clustered at
≤ 10 % sequence identity between subsets), `cif-files/<pdb>.cif` (5,005 mmCIF, about 1.2 GB).
`noncryptic-pockets.json` is used since 2026-10-21 for the per-residue `nc` flag and the
pre-declared sensitivity mask (residues lining a non-cryptic pocket, labelled 0, masked).

## Pipeline (camera-ready port, 2026-10-21)

Protocol fixed before any run of the retrained (`_rv`) checkpoints: see
`camera_ready/cryptobench_audit/DECISIONS.md` (authors' answers) and `PORT.md` (what was
ported, every deviation, test results). Everything below is launched by ONE runner,
`run_cryptobench_rv.py` (skips outputs that exist and verify; `--dry-run` lists the jobs).

| step | script | env | output (run layout) |
|---|---|---|---|
| 1 structures + labels | `cryptobench_build_apo.py` | python with `gemmi` (`--gemmi-python`) | `cryptobench_<set>/*.npz` (bb, y, seq, auth, icode, cb, label_seq, o, nc), `<set>_manifest.json` |
| 1b non-cryptic mask | `cryptobench_common.py noncryptic-mask` | CPU | `<set>_noncryptic_mask.json` |
| 2a Ensembits tokens (primary readout) | `cryptobench_tokenize.py` | GPU | `cryptobench_tokens_ens_<run>_rv.npz` + codebook + provenance |
| 2a' Ensembits z / q / q0 (secondary) | `cryptobench_featurize.py` | GPU | `cryptobench_dense_ens_<run>_rv.npz` + provenance |
| 2b baselines (whole chain, label_seq index) | `cryptobench_tokenize_baselines.py` (`cryptobench_tokenize_esm3struct.py` = wrapper) | GPU (mini3di CPU) | `cryptobench_tokens_<tok>_wc_apo.npz` + codebook + provenance |
| 2d ESM-2 3B embeddings (descriptive row, `--esm2-row on`) | `cryptobench_esm2.py` | GPU | `cryptobench_dense_esm2_3b.npz` + provenance |
| 3 MLP probe (primary) | `cryptobench_mlp_probe.py` | GPU | `mlp/<row>__<readout>__seed<s>.json` + `.preds.npz`, `mlp/<row>__<readout>.json` |
| 3b conv1d probe (secondary) | `cryptobench_probe.py` | GPU | `conv/...` (same layout) |
| 4 statistics | `../protein_level_stats.py` on `cryptobench_families_v1.json` | CPU | `stats/cryptobench_families_v1_results.json` |
| 5 completeness | `run_cryptobench_rv.py` | CPU | `completeness.json` (counts only) |
| (optional) diagnostics | `cryptobench_diagnose.py` | any | reads the rebuttal-era `preds_*.npz` only; superseded by step 4 |
| cluster map (documented generator of the pre-registered file) | `cryptobench_test_clusters.py` | CPU, mmseqs | `{apo: cluster}` JSON |
| check: corpus overlap | `check_training_overlap.py` | CPU, seconds | `overlap_check.json` |

Every job skips only when its output carries the current input fingerprint (manifest,
checkpoint, input file, code, arguments); the authors' tentative choices of 2026-10-22 are
runner options (`--modified-residues`, `--esm2-row`, `--conv-standardize`, `--selection`,
`--aa-order`, `--esm2-layer` (default 33, the coordinator's tentative choice = upstream
pLM-NN's layer), defaults = tentative choice) passed only to the jobs they affect.
Modified polymer residues take their parent from the PDB Chemical Component Dictionary
(`data/external/components.cif.gz`), with the entry's `_pdbx_struct_mod_residue` as fallback.
No script prints a CryptoBench metric value (sealed until the lambda choice, DECISIONS.md
CB8); the results are in the JSON files only. Shared helpers: `cryptobench_common.py`.

Sets: `apo` (cryptic task: apo input, union of all holo pockets; the camera-ready runs),
`apo_main` (apo input, representative pocket only), `holo` (holo input, its own pocket;
open-pocket control).

### Locations (`_cb_paths.py`)

New runs never overwrite the rebuttal files. By default everything goes to a run folder:

- structures, manifests, tokens: `REPO_ROOT/data/cached_descriptors/cryptobench_runs/<run_tag>/`
- probe / diagnose / overlap outputs: `REPO_ROOT/output/cryptobench/runs/<run_tag>/`

`--run-tag` defaults to the name of `ensembits.paths.CKPT_DIR` (since 2026-10-16, code_fix P22;
the run of 2026-10-15 is `--run-tag camera_ready`). `cryptobench_probe.py --tokens` has no default
any more (it was the old flagship's `dih256_fapeH`). Ensembits token files carry a
`<file>.provenance.json` checkpoint block, copied into the probe JSON. `--legacy-layout` points at the rebuttal locations
(`data/cached_descriptors/cryptobench_*`, `data/cryptobench/*_manifest.json`,
`output/cryptobench/`); use it only to read them. Each location also has its own flag
(`--structs-root`, `--manifest-dir`, `--tokens-dir`, `--out-dir`, `--cryptobench-dir`);
relative paths resolve against the current directory if they exist there, else REPO_ROOT.

### Camera-ready run (from `camera_ready/ensembits_code`)

```bash
PY=python                                   # analysis environment
$PY downstream/cryptobench/run_cryptobench_rv.py --run-tag rv_20261021 \
    --gemmi-python <python with gemmi> --gpus 0,1,2 --dry-run   # list the 56 jobs
$PY downstream/cryptobench/run_cryptobench_rv.py --run-tag rv_20261021 \
    --gemmi-python <python with gemmi> --gpus 0,1,2
```

### End to end, rebuttal-era commands (kept for the inventory below; superseded)

(Since 2026-10-21 the probes need `--row`, the baseline tokenizers write `<tok>_wc_<set>`
tags and refuse nothing silently; use the runner above. These lines document how the
rebuttal outputs were made.)


```bash
PY=python      # tokenize / probe / diagnose / overlap
PYG=<python with gemmi>                    # build step only
D=downstream/cryptobench

# 1. structures + labels (deterministic; identical to the rebuttal build)
$PYG $D/cryptobench_build_apo.py --workers 32
$PYG $D/cryptobench_build_apo.py --workers 32 --side apo  --main-holo-only --out-suffix _main
$PYG $D/cryptobench_build_apo.py --workers 32 --side holo --main-holo-only

# 2. tokens. Ensembits default = the reported checkpoint (ensembits.paths.CKPT_DIR)
$PY $D/cryptobench_tokenize.py --set apo --tag ensembits --device cuda:0
#    baselines do not depend on the Ensembits checkpoint: either re-run them ...
$PY $D/cryptobench_tokenize_esm3struct.py --set apo --tag esm3struct --device 0
$PY $D/cryptobench_tokenize_baselines.py --tokenizer aminoaseed --set apo --device 0
$PY $D/cryptobench_tokenize_baselines.py --tokenizer protoken   --set apo --device 1
$PY $D/cryptobench_tokenize_baselines.py --tokenizer mini3di    --set apo
#    ... or probe the rebuttal caches with --tokens-dir data/cached_descriptors

# 3. probe, 5 seeds each (labels match diagnose.py's METHODS)
$PY $D/cryptobench_probe.py --set apo --tokens ensembits      --label ensembits  --seeds 5
$PY $D/cryptobench_probe.py --set apo --tokens aminoaseed_apo --label aminoaseed --seeds 5
$PY $D/cryptobench_probe.py --set apo --tokens esm3struct     --label esm3struct --seeds 5
$PY $D/cryptobench_probe.py --set apo --tokens protoken_apo   --label protoken   --seeds 5
$PY $D/cryptobench_probe.py --set apo --tokens mini3di_apo    --label mini3di    --seeds 5
$PY $D/cryptobench_probe.py --set apo --features aa     --label aa_onehot  --seeds 5
$PY $D/cryptobench_probe.py --set apo --tokens ensembits --features random --label random_tok --seeds 5

# 4. diagnostics + leakage check
$PY $D/cryptobench_diagnose.py
$PY $D/check_training_overlap.py
```

## Inventory: what produced which number

### A. The transcribed table (`rebuttal/cryptobench_summary.md`, `rebuttal/response_cryptobench.md`)

Ensembits level 0 0.6357 AUROC, latent 0.6697, AminoAseed 0.6102, ESM3-struct 0.5983,
aa 0.5818, 3Di 0.5684, 1-frame ablations 0.6123 / 0.6148, SFTD off 0.6374.

- **Produced by a collaborator, outside this repo.** Numbers were transcribed from a
  shared page, not reproduced here. According to that page the scripts were
  `cryptobench_repr.py` (features → probe → summarize), `repr_stats.py` (Welch tests),
  the collaborator's own `cryptobench_probe.py` (structure stage + token caches; a
  different file from the one here) and `cryptic_pocket_probe.py` (tokenizer registry),
  in the collaborator's own working tree. (Updated 2026-10-20: they were located, read
  only, in the collaborator's tree `ensembits-work/submission_exp/src/`, with outputs; paths
  and sha256 in `camera_ready/cryptobench_plan.md`. The camera-ready port is built on them:
  `cryptobench_mlp_probe.py`, `cryptobench_featurize.py`, `cryptobench_esm2.py`.)
- Probe, as described on that page (CryptoBench's pLM-NN head):
  `Linear(d→256) → ReLU → Dropout(0.3)` × 2 → `Linear(256→2)` → softmax; per residue, no
  neighbour context; inputs standardised on train mean/std; Adam, lr 1e-4, weight decay
  1e-5; batch 4,096 residues; cross-entropy with class weights [1, 17.72]; ≤ 80 epochs,
  early stopping on val AUROC with patience 15, best state restored; seeds 0–4; pooled
  AUROC/AUPRC plus per-protein AUROC/AUPRC (chains with both classes); Welch t-tests
  across seeds.
- Ensembits inputs per row: latent z (pre-quantisation), summed q over 3 RVQ levels
  ("code vector"), and the level-0 codebook vector of `encode()`.
- Checkpoints (all **mdCATH + MISATO**, none is the reported BioEmu-CATH + MISATO model):
  `output/dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT` (flagship), `…_PRODSPLIT_noSFTD`,
  `output/dih256_singleframe_mdcathMISATO` (1-frame, single-branch trainer),
  `output/dih256_fapeH_P1train_mdcathMISATO` (1-frame, flagship architecture). All four
  folders exist in the research repo.
- Test set on that page: 261 chains, 73,910 residues, 3,992 positives. The local build
  (step 1) gives 261 chains, 73,974 residues, 3,996 positives: the collaborator used a
  different structure parser.
- "Effective bits" (10.31 / 10.55 / 10.55 / 10.62): the page does not say how they were
  computed.

### B. The local rebuttal run (`rebuttal/cryptobench.md`)

Produced by the scripts copied here (legacy layout), with the old flagship
(`cryptobench_tokenize.py` default `--model-dir` was `…_PRODSPLIT`, tag `dih256_fapeH`;
no tokenization log exists to confirm the model folder).

| result | tokens (data/cached_descriptors/) | probe output (output/cryptobench/) |
|---|---|---|
| Ensembits P=1, apo: AUROC 0.698, AUPRC 0.176, per-protein 0.648 | `cryptobench_tokens_dih256_fapeH.npz` (cryptobench_tokenize.py) | `probe_apo_ensembits_tokens.json`, `preds_apo_ensembits_tokens.npz` |
| AminoAseed 0.701 / 0.166 / 0.663 | `cryptobench_tokens_aminoaseed_apo.npz` (tokenize_baselines) | `probe_apo_aminoaseed_tokens.json` |
| ESM3-struct 0.696 / 0.150 / 0.655 | `cryptobench_tokens_esm3struct.npz` (tokenize_esm3struct) | `probe_apo_esm3struct_tokens.json` |
| ProToken 0.718 / 0.148 / 0.657 | `cryptobench_tokens_protoken_apo.npz` (tokenize_baselines) | `probe_apo_protoken_tokens.json` |
| 3Di 0.672 / 0.129 / 0.633 | `cryptobench_tokens_mini3di_apo.npz` (tokenize_baselines) | `probe_apo_mini3di_tokens.json` |
| aa one-hot 0.651 / 0.105 / 0.653 | none | `probe_apo_aa_onehot_aa.json` |
| random 0.500 / 0.054 / 0.493 | `dih256_fapeH` ids, shuffled | `probe_apo_random_tok_random.json` |
| §4 calibration vs localization, bootstrap, Wilcoxon | | `diagnose_summary.json` (cryptobench_diagnose.py; tables on stdout only) |
| §5 apo_main vs holo (Ensembits, ESM3-struct) | `…_dih256_fapeH_{apo_main,holo}.npz`, `…_esm3struct_{apo_main,holo}.npz` | `probe_{apo_main,holo}_{ensembits,esm3struct}_tokens.json` |

The files without a set prefix (`probe_dih256_fapeH_tokens.json`, `probe_esm3struct_tokens.json`,
`probe_aa_onehot_aa.json`, `probe_random_tok_random.json`) are an earlier run of the same
probe with identical numbers. Structure caches: `data/cached_descriptors/cryptobench_{apo,apo_main,holo}/`
(1,276 / 1,274 / 1,289 chains), manifests `data/cryptobench/{apo,apo_main,holo}_manifest.json`.

This run uses a **different probe** (3 × conv1d, k = 5, AdamW lr 1e-3) and reaches the
opposite conclusion (Ensembits behind AminoAseed on localization, paired per-protein AUROC
−0.027, p = 6.7e-4). Both runs used the old flagship.

### C. Not copied

- `submission_exp/src/rebuttal_analyses/cryptobench_ligand_maccs.py` (MACCS-167 of the
  CryptoBench ligands, rdkit). No script reads its output, and the output
  (`data/cached_descriptors/cryptobench_ligand_maccs.npz`) does not exist, so it was never run.
- The upstream CryptoBench repo (referenced above).

### What was missing to reproduce the transcribed table (status 2026-10-22)

1. The MLP probe: ported (`cryptobench_mlp_probe.py`, bitwise equal to the collaborator's
   training loop on identical inputs, REAUDIT CB1 / CB6); Welch tests replaced by the
   protein-level cluster bootstrap (`../protein_level_stats.py`).
2. Latent and 3-level code-vector features: `cryptobench_featurize.py` (z / q / q0).
3. The collaborator's structure stage: replaced by ours (MSE kept, polymer residues only,
   modified residues mapped to their parent; PORT.md section 2.1 / 9).
4. "Effective bits": entropy of the level-0 token ids over the test residues
   (`cryptobench_common.effective_bits`; equals the collaborator's entropy of distinct q0
   rows when the codebook rows are distinct, as for both lambda candidates).
5. For the reported model: 1-frame BioEmu-CATH + MISATO checkpoints (none exist; not run).

## Leakage check (run 2026-10-15, `check_training_overlap.py`)

PDB-id overlap between CryptoBench and the reported model's training corpus
(`output/reviewer_3_special_corpus_MISATO/cached_desc/`: 16,972 MISATO proteins, 1,057
BioEmu-CATH domains from 1,020 PDB ids; splits are protein-level):

| CryptoBench test | MISATO | of which in the tokenizer's train split | BioEmu-CATH (PDB id) |
|---|---|---|---|
| apo ids (222) | 11 | 10 | 1 (1g59, same chain) |
| holo ids (858) | 88 | 72 | 0 |
| apo entries with apo or any holo id in the corpus | 50 / 222 | 44 / 222 | 1 / 222 |
| test chains (apo manifest) | 59 / 261 | | 1 / 261 |
| test residues | 15,574 / 73,974 (21 %) | | 468 |
| test positives | 1,123 / 3,996 (28 %) | | 15 |

So about a fifth of the test chains (and over a quarter of the positive residues) belong to
entries whose apo or holo structure the tokenizer saw as a MISATO MD trajectory, mostly
the ligand-bound holo complex. This is PDB-id identity only; the same protein under another
PDB id is not counted, so true protein-level overlap can only be higher. The probe's own
train/test split is CryptoBench's and is unaffected; the concern is that the frozen
tokenizer saw dynamics of test proteins, with the ligand bound. Over all folds: 241 of 1,107
entries hit MISATO.

## Open items (status 2026-10-22)

- Probe: decided (DECISIONS.md CB1: MLP primary, conv secondary), both ported.
- Not added (code_fix C13, released checkpoints only): same-data baselines `aminoaseed_ourdata`,
  `esm3style_ourdata` (weights in `output/sametrain_baselines/`, retrained on mdCATH + MISATO).
- Effective bits: no label swap (audit A M13); definition above.
- Leakage: the authors decided against a leakage stratum (DECISIONS.md CB5); the PDB-id table
  above is for the pre-fix corpus (audit B / D give the retrain corpus and sequence identity).
- Descriptor convention: follows the checkpoint (done).
- FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix;
  significance is protein level (paired cluster bootstrap over 30 % identity clusters,
  Holm per pre-registered family).
