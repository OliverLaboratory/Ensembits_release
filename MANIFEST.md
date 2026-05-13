# MANIFEST — external data, weights, and binaries

This repo is **code + a small shipped checkpoint**. The data and external
dependencies listed below are required to reproduce the cells in
[`all_baselines_summary.md`](all_baselines_summary.md). Anything marked
**\[ship\]** must be downloaded into the indicated path before
`reproduce.sh` will run.

---

## One-shot data download

The fastest path: a single dereferenced zip that mirrors the in-repo
layout. Unzipping it at the repo root populates `data/` and
`ckpt/combined_esm3/` with real files (not symlinks):

```bash
# Fetch the data bundle (~7.5 GB)
curl -L -o ensembits_repro_data.zip \
    "https://zenodo.org/records/20152240/files/ensembits_repro_data.zip?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ1YzkxYjI4LTBlY2EtNGM4Mi04ZTE2LTUyYjBlYWJjMWIxMiIsImRhdGEiOnt9LCJyYW5kb20iOiIzYTEzYTBhYTljMTUwZTk4NmI4OTBhYWRkNWI4OWJhYSJ9.qfijHJglR9dlIHWReDW9t9Zo5cVUXI3fAasblenQQ8x8xLoRq96suHLZVKh-vD6_GLN8k3AJR6FSavftn6Jlkg"

# Unzip at the repo root
unzip ensembits_repro_data.zip          # → data/  +  ckpt/combined_esm3/
```

> 🗄️ **Data zip:** ~7.5 GB; DOI: [10.5281/zenodo.20152240](https://doi.org/10.5281/zenodo.20152240). Record is **Restricted-access** during peer review; the tokenized URL above is the reviewer-facing share link. After acceptance we will publish a new Zenodo version with open access + non-anonymized authors (the concept DOI stays the same).

If you're building the zip yourself from a local rebuild of the
canonical caches, run [`scripts/pack_data.sh`](scripts/pack_data.sh) —
it dereferences every symlink under `data/` and `ckpt/` and produces a
single redistributable `ensembits_repro_data.zip` (~9 GB, ~10 min).

The rest of this document is the per-file inventory for users who want
to fetch individual pieces or rebuild from the raw sources.

---

## Data caches (paper-canonical)

| Path | Size | Origin | Status |
|---|---|---|---|
| `data/mdcath_real_bb/<pid>.npz` | ~142 MB | Built once by `scripts/build_real_bb.py --dataset mdcath` from the 3.1 TB raw mdCATH h5 dir; per-domain N/CA/C/O/Cβ + FPS frame indices | **\[ship\]** via Zenodo (preferred over rebuilding from raw mdCATH) |
| `data/misato_real_bb/<pid>.npz` | ~7.8 GB | Built once by `scripts/build_real_bb.py --dataset misato` from `MD.hdf5`; 16,972 PDBs | **\[ship\]** via Zenodo |
| `data/labels/{rmsf_mdcath.npy, rmsf_misato.npz, aa_labels_mdcath.npy}` | ~6 MB | Per-residue RMSF + AA index labels | **\[ship\]** |
| `data/labels/misato_{binding_site,affinity,ec,go}.{npz,json}` | ~12 MB | misato downstream labels | **\[ship\]** |
| `data/splits/{mdcath_div_splits.json, misato_splits.json}` | ~1 MB | CATH-H / sequence / structure / random split files | **\[ship\]** |
| `data/local_s1_cache.npz` | ~12 MB | 308k-residue locally-Kabsch-aligned s₁ cache (mdCATH-div) for the ANOVA probe | **\[ship\]** |
| `data/proteingym/pg_dms_scores.json`, `pg_esm2_scores.json` | ~13 MB | ProteinGym v1.2 substitution scores + ESM2-650M zero-shot pseudo-LL baseline | **\[ship\]** (or fetch via ProteinGym repo) |

## Raw inputs (only needed if rebuilding the real-bb caches)

| Path | Size | Source |
|---|---|---|
| `<mdcath-h5-dir>/mdcath_dataset_<dom>.h5` | 3.1 TB | mdCATH HuggingFace release |
| `<misato-md-hdf5>/MD.hdf5` | 124 GB | MISATO Zenodo release |

## Shipped checkpoint

| Path | Size | Description |
|---|---|---|
| `ckpt/combined_esm3/best.pt` | ~14 MB | Shipped tokenizer (ours/combined/ESM3) state dict |
| `ckpt/combined_esm3/config.json` | <1 KB | descriptor=esm3desc_K16, codebook=[2048,128,128], P=10 |
| `ckpt/combined_esm3/stats.npz` | ~2 KB | per-channel mean/std over training cache |

## Optional: ProteinGym ESMFold cache

| Path | Size | Build |
|---|---|---|
| `data/proteingym_esmfold_atom14/<assay>/<hash>.npy` | ~12 GB | `scripts/build_pg_atom14.py --save-atom14` (HF `facebook/esmfold_v1`) |

## External baseline dependencies (only needed to reproduce specific baseline rows)

| Baseline | External requirement |
|---|---|
| `aminoaseed` | StructTokenBench source tree (https://github.com/KatarinaYuan/StructTokenBench) + `codebook_512x1024-1e+19-linear-fixed-last.ckpt` |
| `esm3struct` | `esm` package (already a core dep); HuggingFace `EvolutionaryScale/esm3-sm-open-v1` (gated; set `HF_TOKEN`) |
| `protoken` | ProToken repo (https://github.com/issacAzazel/ProToken) + `ProToken_Code_Book.pkl` + `data_process/preprocess.py`; install `pip install -e .[protoken]` for TF + JAX |
| `protprofile_K{5,8,10}` | `foldseek` v11.79cd10b static binary + `mmseqs` v17.b804f-static binary on PATH |
| `mini3di_*`, `3di_tokens`, `vote_3di` | `mini3di==0.2.1` (already a core dep) |
| ProteinGym α-blend | `pg_esm2_scores.json` ships with ProteinGym; or `pip install -e .[esm2]` to recompute |

## Hardware

- 1× GPU with ≥ 24 GB VRAM (any of A100/H100/H200/RTX 4090)
- ProToken CPU fallback works but is ~50× slower than TF-GPU

## Licenses and attribution

This repository (code, the shipped tokenizer at `ckpt/combined_esm3/`,
and the docs) is released under the **MIT License** — see
[`LICENSE`](LICENSE).

The data bundle distributed via Zenodo redistributes derivative artifacts
of several upstream resources, each governed by its own license. The
table below lists every redistributed component, its upstream license,
and the canonical citation. All redistribution is permitted under each
upstream license; the bundle inherits **CC-BY 4.0** so that downstream
users are required to cite the upstream sources alongside this work.

| Redistributed component | In-zip path | Upstream license | Cite |
|---|---|---|---|
| mdCATH real-backbone subset | `data/mdcath_real_bb/` | **CC-BY 4.0** | Mirarchi et al. 2024, *Sci. Data* 11 |
| MISATO real-backbone subset | `data/misato_real_bb/` | **CC-BY 4.0** | Siebenmorgen et al. 2024, *Nat. Comp. Sci.* 4(5) |
| ProteinGym DMS labels | `data/pg/dms_scores.json` | **CC-BY 4.0** | Notin et al. 2023, *NeurIPS Datasets & Benchmarks* |
| ESM2-650M pseudo-LL scores | `data/pg/esm2_scores.json` | **MIT** (ESM-2) | Lin et al. 2023, *Science* 379(6637) |
| ESMFold-derived ProteinGym tokens (all 5 tokenizers' wt/mut caches) | `data/pg/{wt,mut}_tokens/*.npz` | ESMFold: **MIT**; per-tokenizer outputs governed by the tokenizer's own license | Lin et al. 2023 + per-tokenizer cite below |
| `aminoaseed` token caches | `data/tokens/aminoaseed_*.npz` | **MIT** (StructTokenBench) | Yuan et al. 2025, arXiv:2503.00089 |
| `protoken` token caches | `data/tokens/protoken_*.npz` | **Apache-2.0** (ProToken) | Lin et al. 2023, *bioRxiv* 2023.11.27.568722 |
| `esm3struct` token caches | `data/tokens/esm3struct_*.npz` | ESM3 Community License (EvolutionaryScale) — outputs permitted for research + redistribution | Hayes et al. 2025, *Science* 387(6736) |
| `3di_tokens`, `vote_3di`, `mini3di_K8`, `protprofile_K{5,8,10}` | `data/tokens/{3di_tokens,vote_3di,mini3di_K8,protprofile_K*}_*.npz` | **GPL-3.0** (Foldseek) + **MIT** (mini3di) | van Kempen et al. 2022, *bioRxiv* 2022.02.07.479398 |
| Shipped Ensembits tokenizer | `ckpt/combined_esm3/`, `data/tokens/ours_*.npz`, `data/codebooks/ours_combined_esm3.npy` | **MIT** (this repository) | this submission |

If you use the data bundle, please cite **all** upstream resources whose
derivatives you consume, in addition to this work. The canonical .bib
entries for each upstream are reproduced in the paper's references list.

### Things not redistributed (gated upstream models)

The bundle does **not** contain:

- ESM3 structure-encoder weights (`EvolutionaryScale/esm3-sm-open-v1`) —
  gated on HuggingFace under EvolutionaryScale's Community License
  Agreement; free for research after CLA acceptance.
- ESMFold weights (`facebook/esmfold_v1`) — gated on HuggingFace; free
  with a HuggingFace account.
- ESM2-650M model weights — only the pre-computed pseudo-LL scores are
  redistributed (sufficient for the PG α-blend); fetching weights
  themselves is automatic via the `esm` package if you want to recompute.
- ProToken / AminoAseed / StructTokenBench / Foldseek / mmseqs binaries
  — fetch instructions in the "External baseline dependencies" table
  above. Each is under a permissive license but the upstream is the
  authoritative source.

Re-running any of the above against new sequences/structures requires
fetching the corresponding upstream weights and accepting their license
terms.
