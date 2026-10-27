# downstream/analyses: token ANOVA and paper figure inputs

Ports of the research scripts behind the token ANOVA and the token, codebook, descriptor
and radar figures (code_fix P20, 2026-10-16). Every script takes the checkpoint from
`--model-dir` / `--variants` and computes descriptors with the checkpoint's own dihedral
convention and chain-break cutoff (`ensembits.featurize.load_model` reads them from
`config.json`; checkpoints trained before the fix load as `legacy_plus180` with no
cutoff). Data paths are relative to `REPO_ROOT` (`ensembits.paths`). Changes against the
research versions are marked `FIXED(...)` in each file.

`token_anova.py` defaults to `output/ensembits_bm_lam10_rv` (the chosen λ = 10,
decisions_20261023 R3); `--variants lambda_sweep` runs the _rv versions of
e0, lam0, lam0p05, lam0p25, lam0p5 and lam1 in one call (baselines computed once). The
other scripts default to `ensembits.paths.CKPT_DIR` (the λ = 10 checkpoint).

## Scripts and paper elements

| Script | Paper element | Reads | Writes |
|---|---|---|---|
| `token_anova.py` (+ `anova_stats.py`) | `appendix.tex` app:token_anova: Table tab:token_anova (s_1 row: default; s_1/s_2 row: `--feature s1_over_s2`), Table tab:token_anova_controls, Table tab:token_anova_baselines, the numbers in the text and the caption of fig:anova_test; main-text claim in `2_results.tex` ("We also perform an ANOVA test ...") | mdCATH-div `ca_10` (residue table) or the research s_1 cache (`--s1-source cache`, needs a sidecar), all-frames RMSF label, mdCATH-div descriptor cache of the checkpoint convention (or `--descriptor-source compute`), baseline caches (below, sidecars checked) | `--out` JSON (+ `.provenance.json`), `<stem>.residues.npz` (residue set, raw token ids of every row, targets), `<stem>.nulls.npz`, `<stem>.table.md`, figures (below), token caches `output/token_anova/token_cache/tokens_mdcath_flat__<variant>__<sha12>__<convention>__P<P>__<src>.npz`, residue table `output/token_anova/residue_table/`, optional `--export-token-cache` npz (raw ids) |
| `plot_anova_test.py` | `appendix.tex` Figure fig:anova_test (`img/ANOVAtest.pdf`; written to `output/token_anova/figures/<variant>/ANOVAtest.{pdf,png}`, the authors swap it in) | outputs of `token_anova.py` (called by it unless `--no-figures`) | the figure per Ensembits variant |
| `plot_codebook_tsne.py` | `appendix.tex` sec:codebook_tsne, Figure fig:codebook_tsne (`img/tSNE_codes.pdf`) and the numbers in that section (utilisation, top/mean, p99/p50, assignments, ensembles per corpus: printed) | training corpus dir (`--corpus-dir`, default `output/bioemu_cath_misato_e0/cached_desc`: `layout.json`, `splits.json`, `esm3desc_dihedral[_iupac]_K16_P10.npy`) | PNG + PDF (`--out`), counts JSON (`--counts-cache`) |
| `plot_skill_radar.py` | `2_results.tex` Figure fig:downstream_radar (`img/downstreams.pdf`) | result JSONs of the downstream battery (`output/ec_conv1d_misato`, `output/go_conv1d_misato`, `output/binding_misato`, recipe tag `--tag stab`), same files as `downstream/_compile_rebuttal_results.py` | `<out-dir>/<stem>.pdf/.png/.csv` |
| `plot_function_dotplot.py` (final_fix_list A3, 2026-10-19; replaces the radar) | `2_results.tex` Figure fig:downstream_radar → dot / forest plot of the function tasks (EC d1-d3 mAP, GO mAP, binding AUROC; sequence / structure panels; mean ± seed sd, EC / GO per-fold marks, random-feature line, single- vs multi-frame blocks, both λ candidates at P = 1 / P = full) | only through the checked loaders of `downstream/_compile_rebuttal_results.py` (split file, CV file, labels, function protocol incl. input standardisation, items / classes, 10 seeds); validation fields by default, `--split-metric test` refused without `--unseal` | `output/figures/function_dotplot_{val,test}.{pdf,png,json}` |
| `recon_by_p.py` (A9) | `appendix.tex` app:ablation-vq / app:ablation-sftd, recon columns | `output/ensembits_bm_<run>/` (config, best.pt, history), the training corpus descriptor cache + splits; GPU (data on the CPU, one batch at a time) | `output/ablations/recon_by_p.json` |
| `token_usage_by_p.py` (A9) | same tables: L1 distinct ids / perplexity at P = 1 and P = 10 (validation split; P = 10 checked against config.json per_level); `--source anova-cache` = the mdCATH-div ANOVA token caches | as recon_by_p (CPU) | `output/ablations/token_usage_by_p.json` |
| `fig_descriptor_extract.py` | descriptor figure (`camera_ready/fig_descriptor/`, illustrates `appendix.tex` app:descriptor-esm3) | mdCATH HDF5 (`data/mdcath/data`, `--h5-dir`) | `frames_P10.pdb`, `geom.json`, `desc.npy` in `--out-dir` (default `camera_ready/fig_descriptor/build`) |
| `fig_codebook_mine_tokens.py` | codebook row figure (`camera_ready/fig_codebook/`, token picks closest to their codeword; candidate panel for app:viz) | mdCATH HDF5 | `tok<n>.pdb`, `tokens.json` in `--out-dir` (default `camera_ready/fig_codebook/build`) |
| `_common.py` | (helpers) checkpoint hash, config fallbacks, sidecar check | | |

Rendering stays in the research repo and only reads the files written here (no
descriptors or tokens are computed there): `camera_ready/fig_descriptor/render_pymol.py`,
`render_p3_confs.py`, `sheet.py`, `camera_ready/fig_codebook/render_tokens.py` (PyMOL
environment).

### Paper elements NOT produced by this folder

| Element | Research script | Status |
|---|---|---|
| fig:token1699, fig:token1063 (`2_results.tex`), fig:5x5tokens (`appendix.tex` app:viz) | `experiments/shared_states_v2/render_token_knn_pymol.py` (exemplar selection, 5-frame overlays; reads the token column of `token_pca_residues_aligned.npz` and, for latent scoring, loads the model through the research `ensembits_featurize` with `_resolve_descriptor_mode`, i.e. the legacy +180° descriptor) | not ported. Feed it the new tokens with `token_anova.py --export-token-cache <npz>` and `--cache <npz>`; its latent scoring still uses the legacy descriptor and needs a port before it can be used with an iupac checkpoint. Token ids and exemplars in the captions change (require_fix #17). |
(fig:anova_test and tab:token_anova_controls are now produced by `plot_anova_test.py` and
`token_anova.py`, anova_audit A4 / A5, 2026-10-18. The direction-alignment test of
`test_token_dynamics_controls.py` is not ported: no paper element uses it.)

## Token ANOVA: statistics and residue set (anova_audit A1-A5 and fix round 2, 2026-10-18)

One residue set for every row: the requested mdCATH-div proteins minus the 53 residues
whose s_1 = 0 is a placeholder (fewer than 4 C-alpha within 10 A in frame 0; the count is
in the JSON). Every other gap (a protein missing inside a baseline cache, a domain without
a real backbone in compute mode, an ideal-geometry domain without `--allow-ideal`, a
letter outside the 20 amino acids, an unknown `--pids` entry, a missing RMSF value) is an
error. For e0: 308,240 residues, 2,442 proteins. No rare-token pooling and no count filter
(`--min-count` only adds a `diagnostic_pooled` block). The residue-set modes
`intersection` / `per_method` were removed.

Per row and target (s_1, or s_1/s_2 with `--feature`; all-frames RMSF):
* ANOVA: eta^2, omega^2 (M = distinct tokens used), F and its df. ProtProfileMD_K10
  (authors' decision: linear fit on its 20-D profile): OLS R^2 and adjusted R^2 (the omega^2
  counterpart). Permutation p = (b + 1)/(n + 1) under a within-protein circular shift (each
  protein's token sequence rotated by 5 to L - 5 residues; keeps the protein's token mix
  and the chain autocorrelation; n = `--n-perm` 1000; b, null mean / sd / max stored;
  written "p < 10^-3 (0/1,000)" when b = 0). The plain shuffle and the parametric F p are
  in the JSON only.
* Held-out within-protein R^2: target centred per protein; `--n-folds` 5 folds of proteins
  grouped by CATH topology (`data/external/mdcath_div/_manifest.json`,
  `representatives[pid].topology`), size-balanced by a deterministic greedy assignment
  (per-protein folds in `<stem>.residues.npz`); token means fitted on the other folds, an
  unseen token predicts 0; the prediction is centred per protein too (no label used;
  `r2_uncentred_prediction` without it); ProtProfileMD: ridge (alpha 1). 95% CIs: CATH-topology
  cluster bootstrap (B = `--n-boot` 1000). Delta per frame regime (authors' decision):
  Ensembits P = 1 vs the best single-frame tokenizer (3Di, AminoAseed, ESM3struct,
  ProToken), P = 10 vs the best 10-frame baseline (Vote_3Di, ProtProfileMD_K10); paired
  bootstrap. The amino acid row is shown, not a tokenizer baseline.
* chance rows: each Ensembits row's own tokens permuted over the residue set (same M and
  frequencies, seeded).
* Controls on the same residues: CATH class, protein-length quintile, residue-position
  quintile (as `test_token_dynamics_controls.py`), chain-end distance bins, protein identity.
* Gain over confounds (same folds): confounds = joint cell (chain-end distance bin x burial
  decile x C-alpha SS proxy, frame 0) + amino acid. Primary (`gain_over_confounds`,
  table.md): the method backfitted jointly with the confounds. Sensitivity only (JSON:
  `gain_over_confounds_residual_sensitivity`): the method fitted on the confound model's
  training residual.
* Provenance: code file sha256s, git HEAD, each checkpoint's sha and lambda
  (`consistency_mse_weight`); the residue table is keyed on the content of every `ca_10`.

Ensembits P = 1 reads frame 0 only (the frame of every single-frame baseline): frames [:P]
of the P = 10 descriptor cache (identical to the descriptor of `bb_10[:P]`; checked on 6
proteins, 100% equal tokens at P = 1 and P = 10).

## Baseline sources (token_anova.py)

Every cache must exist (a missing cache is an error) and carry a current provenance
sidecar (`rmsf_common.check_cache_provenance`: schema, dataset, file sha256, frames,
sequence source, codebook); `--allow-unverified-baselines` (recorded) skips the sidecar
check for side experiments only.

| Row | File (`data/cached_descriptors/`) | Builder (sidecar `builder`) |
|---|---|---|
| 3Di (mini3di, frame 0; chain-end residues take Foldseek's invalid state 2, count in the JSON) | `mini3di_tokens_mdcath.npz` | `data_prep/baselines/build_mini3di_misato.py --dataset mdcath` |
| Vote_3Di (paper rule: plurality with count >= 2, else the centroid nearest to the frames' mean centroid, lowest letter id on exact ties; `mini3di_baselines.vote_from_frame_tokens`) | `mini3di_tokens_K10_mdcath.npz` | `data_prep/baselines/build_mini3di_tokens_K_misato.py --dataset mdcath` |
| ProtProfileMD_K10 (Foldseek / MMseqs2 profile of the first 10 frames; linear fit: OLS R^2 / adjusted R^2, ridge held-out) | `protprofile_K10_mdcath.npz` | `data_prep/baselines/build_protprofile_K_misato_foldseek.py --dataset mdcath` |
| AminoAseed | `aminoaseed_tokens_mdcath.npz` (+ `aminoaseed_codebook.npy`) | `data_prep/baselines/build_aminoaseed_tokens.py` |
| ProToken | `protoken_tokens_mdcath.npz` (+ codebook) | `data_prep/baselines/build_protoken_tokens.py` (+ `merge_protoken_shards.py`) |
| ESM3struct | `esm3struct_tokens_mdcath.npz` (+ codebook) | `data_prep/baselines/build_esm3struct_tokens.py` |
| amino acid | `mdcath_real_bb/<pid>.npz['seq']` (X -> H) | `data_prep/build_mdcath_real_bb.py` (all 2,442 domains) |
| chance (one per Ensembits row; `random` in `--baselines`) | that row's tokens permuted over the residue set (seeded) | |
| s_1, s_2, burial, SS proxy | computed from `data/external/mdcath_div/<pid>.npz['ca_10']`, cached in `output/token_anova/residue_table/` (sidecar); `--s1-source cache` reads `experiments/results/shared_states/token_pca_residues_aligned.npz` (no sidecar: refused unless `--allow-unverified-s1-cache`; equal to the recompute, audit 2026-10-18) | `token_anova.py` |
| all-frames RMSF | `output/phase_b/cached_desc/rmsf_div_allframes.npy` (sidecar: label name, content hash, row order = manifest) | `data_prep/build_mdcath_div_rmsf_labels.py` |

## Run order after the retrain

1. Retrained checkpoint in `ckpt/<new>` and `CKPT_DIR` switched (P18).
2. mdCATH-div descriptor cache in the new convention:
   `python data_prep/build_esm3desc_dihedral_perdataset.py --dataset mdcath --dihedral-convention iupac`
   (or use `--descriptor-source compute` below).
3. Baseline mdCATH-div token caches: rebuild only if the baseline settings change
   (chain-break threshold P5, move-aside list P15).
4. ANOVA (about 4 min for one variant, about 11 min for the six lambda variants; tokens cached):
   `python downstream/analyses/token_anova.py --variants <chosen variant dir> --out output/token_anova/token_anova.json --export-token-cache output/token_anova/token_pca_residues_aligned__{variant}.npz`;
   all lambda variants: `--variants lambda_sweep`; `--feature s1_over_s2` for the
   motion-shape row. Figures in `output/token_anova/figures/<variant>/`.
5. t-SNE (needs the training corpus of step E0):
   `python downstream/analyses/plot_codebook_tsne.py --counts-cache output/codebook_tsne/counts_<variant>.json --out output/codebook_tsne/tSNE_codes.png`
6. Descriptor and codebook figures: `fig_descriptor_extract.py --domain 1ctfA00 --query 42`
   (the residue of the current figure, Leu42) and `fig_codebook_mine_tokens.py`, then the
   PyMOL renderers in the research repo.
7. After the downstream battery and the table compilation:
   `python downstream/analyses/plot_function_dotplot.py` (validation; the radar,
   `plot_skill_radar.py`, is superseded by this dot plot, final_fix_list A3); the test version
   only after λ is chosen: `--split-metric test --unseal`.
   Ablation inputs (A9): `recon_by_p.py --device cuda:0`, `token_usage_by_p.py` (CPU),
   `token_anova.py --variants output/ensembits_bm_vq2048_rv output/ensembits_bm_nosftd_rv --out
   output/token_anova/token_anova_rv.json`; the compiler reads them for the VQ / SFTD tables.
8. Copy the figures into `ensembits_neurips_2026/img/` and update the numbers in the text.
