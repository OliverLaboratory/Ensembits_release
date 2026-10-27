# data_prep/baselines: MISATO baseline caches

Builders of the per-protein token / feature caches that the baseline probes in
`downstream/baselines/` (and `downstream/probe_rmsf_*.py`) read. Each file has a
header with its paper reference and origin (`submission_exp/src/...`). Shared
helpers (real-or-ideal backbone, ProToken wrapper) are in `ensembits/baselines/`.
All paths are relative to `REPO_ROOT` (the research repo); outputs go to
`data/cached_descriptors/`.

| File | Writes | Needs |
|---|---|---|
| `../build_coords_ca8_misato.py` | `coords_ca8_misato.pkl` (8 frames) | `misato_real_bb/` |
| `../build_coords_ca5_misato.py` | `coords_ca5_misato.pkl` (first 5 frames) | `misato_real_bb/` |
| `build_mini3di_misato.py` | `mini3di_tokens_misato.npz`, `mini3di_features_misato.npz` (frame 0) | ca5 pickle, `misato_real_bb/`, `mini3di` |
| `build_mini3di_tokens_K_misato.py` | `mini3di_tokens_K8_misato.npz` (all 8 frames) | ca8 pickle, `misato_real_bb/`, `mini3di` |
| `build_protprofile_K_misato_foldseek.py` | `protprofile_K8_misato.npz` (`--K 8`) | `misato_real_bb/`, foldseek + mmseqs on PATH |
| `build_esm3struct_tokens.py` | `esm3struct_tokens_{misato,mdcath}.npz`, `esm3struct_codebook.npy` | ca8 pickle, real-bb caches, `esm`, HF login / `HF_TOKEN` |
| `build_aminoaseed_tokens.py` | `aminoaseed_tokens_{misato,mdcath}.npz`, `aminoaseed_codebook.npy` | ca8 pickle, real-bb caches, StructTokenBench source + checkpoint |
| `build_protoken_tokens.py` | `protoken_tokens_{misato,mdcath,bioemu}.npz` (+ `.provenance.json`; or shards in `protoken_shards/`), `protoken_codebook.npy` | ca8 pickle, MISATO pairs pickle (fallback), real-bb caches (mdCATH sequence: `mdcath_real_bb` with X -> H), ProToken source (TF + JAX) |
| `../build_misato_chain_breaks.py` | `misato_chain_breaks.json` (+ sidecar): kind of every MISATO backbone break (boundary / missing_gap / consecutive / unresolved) | `misato_real_bb/`, deposited PDB entries (`--deposited-dir`, `--download`) |
| `_chain_pieces.py` | (library, 2026-10-18, option A) the true chains each encoder is run on + gap-aware residue index: MISATO from `misato_chain_breaks.json`, mdCATH / BioEmu = whole protein | `misato_chain_breaks.json` |
| `_bioemu_inputs.py` | (library) BioEmu-CATH frames, real N/CA/C/O + Cβ, sequences for `--dataset(s) bioemu` | `bioemu_cath_real_bb/`, `bioemu_cath_extras/` (`../build_bioemu_extras.py`) |
| `merge_protoken_shards.py` | `protoken_tokens_{misato,mdcath}.npz` from the shards | `protoken_shards/` |
| `build_pooled_baseline_features.py` | `{esm3struct,aminoaseed,3di}_pooled_misato.npz` (rebuttal) | ca8 pickle; `3di` needs the K8 tokens; `aminoaseed` needs its codebook |
| `build_protoken_pooled.py` | `protoken_pooled_misato.npz` (or `protoken_pooled_misato_shard<k>of<n>.npz`) (rebuttal) | ca8 pickle, `misato_real_bb/`, ProToken source |
| `merge_protoken_shards.py --pooled` | `protoken_pooled_misato.npz` from the pooled shards | `protoken_pooled_misato_shard*of<n>.npz` |

The `mini3di_centroids.npy` (20 x 2) read by the probes is not built here (it is
mini3di's `Encoder._CENTROIDS`, see `ensembits/baselines/mini3di_baselines.py`).

## Run order

1. `python data_prep/build_misato_real_bb.py ...` (fixed parser, code_fix §4)
2. `python data_prep/build_coords_ca8_misato.py` and `python data_prep/build_coords_ca5_misato.py`
3. Single-frame and multi-frame caches, in any order:
   `build_mini3di_misato.py`, `build_mini3di_tokens_K_misato.py`,
   `build_protprofile_K_misato_foldseek.py --K 8`,
   `build_esm3struct_tokens.py`, `build_aminoaseed_tokens.py`,
   `build_protoken_tokens.py` (+ `merge_protoken_shards.py` when sharded; MISATO shards
   run one GPU per shard: `CUDA_VISIBLE_DEVICES=k ... --shard k --num-shards n --device ""`)
4. Pooled (rebuttal) features, after step 3: `build_pooled_baseline_features.py
   --baseline {esm3struct,aminoaseed,3di}` (`aminoaseed` reads the codebook of
   step 3, `3di` the K8 tokens), `build_protoken_pooled.py`
   (+ `merge_protoken_shards.py --pooled` when sharded; one GPU per shard via
   `CUDA_VISIBLE_DEVICES=k`)
5. Probes: `python downstream/baselines/run_misato_baselines.py`

BioEmu-CATH (phase 2, 2026-10-16; the baselines of the BioEmu RMSF column), after
`../build_bioemu_extras.py` (re-derived FPS picks, sequence, real Cβ):
`build_mini3di_misato.py --dataset bioemu` (frame 0 -> `mini3di_{features,tokens}_bioemu.npz`),
`build_mini3di_tokens_K_misato.py --dataset bioemu` (10 frames -> `mini3di_tokens_K10_bioemu.npz`),
`build_protprofile_K_misato_foldseek.py --K 10 --dataset bioemu` (`protprofile_K10_bioemu.npz`),
`build_aminoaseed_tokens.py --datasets bioemu`, `build_esm3struct_tokens.py --datasets bioemu`,
`build_protoken_tokens.py --datasets bioemu` (`<name>_tokens_bioemu.npz`, unsharded). The
`aa` and `random` baselines need no cache. Single-frame = frame 0 of `bb_10`, multi-frame =
all 10 frames. Then `downstream/probe_rmsf_baselines.py --dataset bioemu --feature <f>`.

Every builder that reads `misato_real_bb` (directly, through
`ensembits/baselines/_real_or_ideal_bb.py`, or through the coords pickles) refuses
a cache written by the old MISATO parser (code_fix C5); `--allow-old-parser`
skips that check, only to reproduce old results.

Since 2026-10-16 (code_fix P15) every builder here (and `../build_coords_ca5_misato.py`)
**exits 1 when its output already exists**, unless `--overwrite` (they used to skip
it silently, or overwrite it). Multi-dataset builders (`build_esm3struct_tokens.py`,
`build_aminoaseed_tokens.py`, `build_protoken_tokens.py`) check every requested
dataset's output: pass `--datasets misato` to rebuild only MISATO. The token
builders also **exit 1 on the first MISATO frame that falls back to ideal geometry**,
unless `--allow-ideal`: after the rebuild every MISATO frame of the coords pickles has
a real backbone in `misato_real_bb`. The merge script exits 1 when a shard is missing.

Chain splitting (authors' decisions, 2026-10-18, baseline fidelity audits, option A, table
v2; replaces code_fix P5): the baselines are cut **only at true chain boundaries**, matching
Ensembits, whose descriptor does not cut the protein at a gap (it only zeroes the
dihedrals across C–N > 2.5 Å). A MISATO `chain_index` (misato_real_bb, from the MISATO
`molecules_begin_atom_index`) is a MISATO *molecule*: often several polypeptide chains
(thrombin L + H, 3V2P A–E), and one deposited chain is often split into two molecules at a
missing loop (thrombin H147→H150). `../build_misato_chain_breaks.py` therefore places
every backbone break of every complex (C–N > 2.5 Å, and every chain_index change) in the
deposited mmCIF (`data/external/rcsb_mmcif/`, first model, polymer CA): full-length
5-residue exact flanks first, else a joint alignment of up to 15 residues per side
(identity >= 0.8, >= 4 identical; a short side only next to a long one), else segment
alignment; the kind follows from the SEQRES (label_seq_id) step: `missing_gap` (step > 1,
n_missing = step - 1), `consecutive` (step 1), `boundary` (different chains or copies; a
chain_index change that cannot be placed; or a same-chain pair whose Cα distance in the
DEPOSITED first model exceeds 3.80 Å * step + 0.5 Å, source "physical"; v2.1: MISATO
simulated each molecule alone, so frame-0 distances across a missing loop overstate the
gap; v2.2: a SEQRES step 1 with deposited Cα <= 6.0 Å is a strained bond, `consecutive`,
flag `strained_bond`), `unresolved` (inner break not placed). v2.3: candidates (chains,
copies) are chosen by geometry (Kabsch of up to 15 frame-0 Cα per side on the deposited
Cα; RMSD and runner-up recorded), 1-3-residue molecules are placed between their long
neighbours (or at the nearest same-type deposited Cα within 2 Å), and an unplaced
chain_index change with frame-0 C–N < 4 Å is kept whole (source "unplaced_bonded"); SEQRES
gaps MISATO closed with a bond keep step 1 (counted, `n_bonded_label_gaps`). Table of
2026-10-18 (rule v2.3; 16,972 complexes, 29,224 breaks): chain_index changes 16,542
missing_gap, 68 consecutive, 14 unplaced_bonded, 10,744 boundary, 401 boundary unplaced,
100 physical boundary; inner breaks 872 missing_gap, 2 consecutive, 418 boundary,
40 unresolved, 23 physical boundary; 1,543 complexes hold more copies of a chain than the
deposited asymmetric unit (`extra_copies`). Compared with the v1 table: pieces 45,302 ->
28,698 (< 5 residues 1,196 -> 558, < 10 residues 2,654 -> 1,293); 6,603 complexes change
their pieces and 36 only their numbering.
`_chain_pieces.py` cuts at `boundary` and `unresolved` breaks, keeps a chain whole across
`missing_gap` and `consecutive` breaks and numbers residues by SEQRES position. The
residue index goes to the encoders that take one (AminoAseed's residue_index input, ESM3
`encode(..., residue_index=)`, ProToken's PDB residue numbers); mini3di, Foldseek /
ProtProfileMD and the pooled 3Di use the pieces only. mdCATH-div and BioEmu-CATH are
single chains, encoded whole (no C–N > 2.5 Å in any frame of their real backbones, so
their caches are unchanged and their legacy 2.5 Å sidecars are still accepted,
`rmsf_common.LEGACY_SPLIT_NOOP`). Per-chain encoding is kept (no joint / multimer mode),
and AminoAseed keeps its projected codebook. Sidecars record `chain_split`
(`rmsf_common.split_record`: "true_chain_v2" + the chain-break table's sha256 for MISATO,
"whole protein (single chain)" otherwise) instead of `chain_break_A`;
`rmsf_common.check_cache_provenance` (both RMSF probes, new runs), `merge_protoken_shards.py`
and the pooled `3di` require it (refusing MISATO caches cut at 2.5 Å, made with the v1 table
or from another chain-break table); results already written are judged by the caches they
recorded. `split_chains_by_peptide_bond` remains only for callers outside this folder
(CryptoBench, fold pairs).

### Move aside for the chain-split change (2026-10-18)

First build the chain-break table (`../build_misato_chain_breaks.py --deposited-dir <dir>
--download`). The MISATO caches built before it (sidecar without `"chain_split":
"true_chain_v2"`) are refused by new probe runs. In `data/cached_descriptors/`, move away (with their
`.provenance.json`): `mini3di_tokens_misato.npz`, `mini3di_features_misato.npz`,
`mini3di_tokens_K8_misato.npz`, `protprofile_K8_misato.npz`, `aminoaseed_tokens_misato.npz`,
`esm3struct_tokens_misato.npz`, `protoken_tokens_misato.npz`, `protoken_shards/misato_shard*`,
`esm3struct_shards/misato_shard*`, `{esm3struct,aminoaseed,3di,protoken}_pooled_misato.npz`,
`protoken_pooled_misato_shard*`; in `output/`: `probe_rmsf_misato/baseline_*__test.json` and
the baseline JSONs of `{ec_conv1d,go_conv1d,binding}_misato/`. Rebuild in the run order
above (step 3, then the pooled step 4, then the probes); the mdCATH-div and BioEmu-CATH
caches stay.

### Move aside before the rebuild (code_fix E0, P15)

In `data/cached_descriptors/`, rename (e.g. suffix `_legacy_proline`) or move away:

- `coords_ca5_misato.pkl`
- `mini3di_tokens_misato.npz`, `mini3di_features_misato.npz`
- `mini3di_tokens_K8_misato.npz`
- `esm3struct_tokens_misato.npz`, `aminoaseed_tokens_misato.npz`, `protoken_tokens_misato.npz`
- `protprofile_K8_misato.npz`
- `esm3struct_pooled_misato.npz`, `aminoaseed_pooled_misato.npz`, `3di_pooled_misato.npz`,
  `protoken_pooled_misato.npz`
- `protoken_pooled_misato_shard*` (pooled ProToken shards)
- `protoken_shards/misato_shard*` (ProToken token shards)
- also `esm3struct_shards/misato_shard*`, if the ESM3struct tokens were built sharded

(`misato_real_bb`, `misato_real_bb_kabsch`, `coords_ca8_misato.pkl` and `misato_rmsf.npz`
are moved aside earlier in the E0 order, `camera_ready/code_fix.md` E0.) The codebook
files (`*_codebook.npy`) are rewritten by the builders and do not depend on MISATO.
Before the mdCATH-div caches: `../build_mdcath_real_bb.py --only-missing` (2026-10-17) gives
the 8 domains without a real backbone their file, so every mdCATH baseline cache covers all
2,442 domains (a missing domain now fails the build: it is in the RMSF residue set).
The mdCATH-div caches (`*_tokens_mdcath.npz`, `mini3di_*_mdcath.npz`,
`mini3di_tokens_K10_mdcath.npz`, `protprofile_K10_mdcath.npz`) MUST be rebuilt as well
(G15, 2026-10-17; this note used to say they need no rebuild): the RMSF probes refuse any
cache without a current provenance sidecar, and the rebuilt ones use His (X -> H) and no
ideal-geometry fallback inside the RMSF set (and, since 2026-10-18, record the
whole-protein chain split).

Environment: importing `ensembits` loads torch. ESM3struct needs `esm`;
AminoAseed needs `omegaconf` and the StructTokenBench source
(`ENSEMBITS_STB_SRC`); ProToken needs TensorFlow + JAX and its source tree
(`ENSEMBITS_PROTOKEN_DIR`); mini3di / ProtProfileMD need `mini3di`, foldseek and
mmseqs (`ENSEMBITS_FOLDSEEK`, `ENSEMBITS_MMSEQS`). See `ensembits/paths.py`.
