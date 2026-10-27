"""Where the code lives and where the data lives.

The camera-ready codebase (this folder) holds only code, the reported checkpoint
and documentation. The large inputs and outputs (MD trajectories, real-backbone
caches, descriptor caches, token caches, downstream JSONs) stay in the original
research repository. Every script resolves data paths through this module, so the
codebase can be moved without editing paths in each file.

    CODE_ROOT  : this codebase (camera_ready/ensembits_code)
    REPO_ROOT  : the folder that holds data/ and output/. Default: two levels above
                 this codebase (camera_ready/ensembits_code -> the research repo);
                 override with the environment variable ENSEMBITS_REPO_ROOT.
    EXTERNAL   : REPO_ROOT/data/external, third-party inputs (each overridable):
                 MDCATH_DIV_SRC (ENSEMBITS_MDCATH_DIV_SRC), PROTEINGYM_ZERO_SHOT_DIR
                 (ENSEMBITS_PROTEINGYM_DIR), STRUCTTOKENBENCH_SRC (ENSEMBITS_STB_SRC).
    PROTOKEN_SRC : the ProToken-1.0 `full/` source tree used by the ProToken baseline,
                 default REPO_ROOT/data/protoken/full (ENSEMBITS_PROTOKEN_DIR).
    FOLDSEEK_BIN, MMSEQS_BIN : the foldseek / mmseqs executables of the ProtProfileMD
                 baseline; found on PATH (ENSEMBITS_FOLDSEEK / ENSEMBITS_MMSEQS override).
    TMALIGN_BIN : the TM-align executable of the fold-matched pair test
                 (downstream/fold_pairs); ENSEMBITS_TMALIGN overrides, else TMalign on
                 PATH, else USalign on PATH (same TM-score / alignment output), else
                 the bare name "TMalign".
    MISATO_AFFINITY, MISATO_LIGAND_MACCS : affinity labels and ligand MACCS fingerprints of
                 the MISATO affinity probe (2026-10-19; ENSEMBITS_MISATO_AFFINITY /
                 ENSEMBITS_MISATO_MACCS override)
    MISATO_SPLITS : the MISATO probe split file, data/misato_splits_v2.json (benchmark_audit
                 D3; data_prep/build_misato_splits_v2.py); MISATO_SPLITS_LEGACY = the old
                 data/misato_splits.json (read only by that builder)
    MISATO_FUNCTION_CV : the EC / GO cross-validation file, data/misato_function_cv_v1.json
                 (distinct proteins, grouped stratified 5-fold CV; 2026-10-19,
                 data_prep/build_misato_function_cv.py; ENSEMBITS_MISATO_FUNCTION_CV overrides)
    CKPT_DIR   : the checkpoint reported in the paper (BioEmu-CATH + MISATO); for now
                 the legacy checkpoint, switched to the retrained one after E0
    LEGACY_CKPT_DIR : the pre-fix (legacy +180 deg) checkpoint, used by checks/

Paper: not described in the paper (infrastructure only).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]

# Original scripts computed ROOT = <file>.parents[2] inside the research repo and
# then used ROOT / "data" / ..., ROOT / "output" / ... . Pointing ROOT here keeps
# every data path exactly as it was when the paper's numbers were produced.
REPO_ROOT = Path(os.environ.get("ENSEMBITS_REPO_ROOT", CODE_ROOT.parents[1])).resolve()

# Third-party inputs, addressed relative to REPO_ROOT (no machine-specific paths).
EXTERNAL = REPO_ROOT / "data" / "external"
MDCATH_DIV_SRC = Path(os.environ.get("ENSEMBITS_MDCATH_DIV_SRC", EXTERNAL / "mdcath_div"))
# mdCATH-div RMSF labels (code_fix P3, 2026-10-16): the converged-superposition label
# built by data_prep/build_mdcath_div_rmsf_labels.py; the legacy label (3 rounds from the
# mean of the raw frames) is kept for reproducing earlier numbers.
MDCATH_DIV_RMSF = REPO_ROOT / "output" / "phase_b" / "cached_desc" / "rmsf_div_converged.npy"
MDCATH_DIV_RMSF_LEGACY = REPO_ROOT / "output" / "phase_b" / "cached_desc" / "rmsf_div.npy"
PROTEINGYM_ZERO_SHOT_DIR = Path(os.environ.get("ENSEMBITS_PROTEINGYM_DIR",
                                               EXTERNAL / "proteingym_zero_shot_scores"))
STRUCTTOKENBENCH_SRC = Path(os.environ.get("ENSEMBITS_STB_SRC",
                                           EXTERNAL / "StructTokenBench" / "src"))
# ProToken-1.0 source tree (data_process/, ProToken_Code_Book.pkl, model files),
# used by data_prep/baselines/build_protoken_*.py.
PROTOKEN_SRC = Path(os.environ.get("ENSEMBITS_PROTOKEN_DIR",
                                   REPO_ROOT / "data" / "protoken" / "full"))
# foldseek / mmseqs executables (data_prep/baselines/build_protprofile_K_misato_foldseek.py).
# Resolved on PATH; the bare name is kept as a last resort so the error shows up
# when the binary is called, not at import time.
FOLDSEEK_BIN = os.environ.get("ENSEMBITS_FOLDSEEK") or shutil.which("foldseek") or "foldseek"
MMSEQS_BIN = os.environ.get("ENSEMBITS_MMSEQS") or shutil.which("mmseqs") or "mmseqs"
# MISATO probe splits (EC, GO, binding site, affinity, MISATO RMSF). Authors' decision D3
#   (2026-10-18, camera_ready/benchmark_audit.md D3): every MISATO probe reads MISATO_SPLITS
#   (v2: MMseqs2 30 % sequence clusters / CATH H, data_prep/build_misato_splits_v2.py) and
#   records its path + sha256 in its result JSON; the compilers refuse results built on
#   another split file. MISATO_SPLITS_LEGACY is the old file (UniProt / first CATH H
#   grouping, leaky); it is read only by the v2 builder (universe, random, ligand_disjoint,
#   leakage comparison). ENSEMBITS_MISATO_SPLITS overrides (tests only).
MISATO_SPLITS = Path(os.environ.get("ENSEMBITS_MISATO_SPLITS",
                                    REPO_ROOT / "data" / "misato_splits_v2.json"))
MISATO_SPLITS_LEGACY = REPO_ROOT / "data" / "misato_splits.json"
# MISATO function cross-validation (authors' decision 2026-10-19, option A): the EC and GO
#   probes run on distinct proteins (one representative complex per UniProt accession set)
#   with grouped, label-stratified 5-fold cross-validation in a sequence and a structure
#   mode (data_prep/build_misato_function_cv.py; loader downstream/function_cv.py). Binding
#   site, affinity and RMSF keep MISATO_SPLITS. ENSEMBITS_MISATO_FUNCTION_CV overrides (tests).
MISATO_FUNCTION_CV = Path(os.environ.get("ENSEMBITS_MISATO_FUNCTION_CV",
                                         REPO_ROOT / "data" / "misato_function_cv_v2.json"))
# MISATO EC labels. Authors' decision D8 (2026-10-18, EC audit): the EC probes read
#   MISATO_EC_LABELS (v2: polyprotein chains labelled by the mature products they cover,
#   data_prep/build_misato_ec_labels.py) and record its path + sha256 in every result JSON;
#   the compilers refuse EC results built on another label file. MISATO_EC_LABELS_LEGACY is
#   the old per-PDB SIFTS union (symbolic link to the original file; copied by the builder
#   to data/annotations/source/misato_ec.json). ENSEMBITS_MISATO_EC_LABELS overrides (tests).
MISATO_EC_LABELS = Path(os.environ.get("ENSEMBITS_MISATO_EC_LABELS",
                                       REPO_ROOT / "data" / "annotations" / "misato_ec_v3.json"))
MISATO_EC_LABELS_LEGACY = REPO_ROOT / "data" / "annotations" / "misato_ec.json"
# MISATO affinity labels (-log Kd/Ki) and ligand MACCS fingerprints (function_reaudit pre-run
#   fixes, 2026-10-19): built by data_prep/build_misato_affinity_labels.py and
#   data_prep/build_misato_ligand_maccs.py (ported from the research repo), each with a
#   provenance sidecar; the affinity probes record both sha256 in every result JSON
#   (downstream/function_common.py). ENSEMBITS_MISATO_AFFINITY / ENSEMBITS_MISATO_MACCS
#   override (tests only).
MISATO_AFFINITY = Path(os.environ.get("ENSEMBITS_MISATO_AFFINITY",
                                      REPO_ROOT / "data" / "annotations" / "misato_affinity_v2.json"))
MISATO_LIGAND_MACCS = Path(os.environ.get(
    "ENSEMBITS_MISATO_MACCS", REPO_ROOT / "data" / "cached_descriptors" / "misato_ligand_maccs_v2.npz"))
# TM-align executable (downstream/fold_pairs/: mining, verification, token comparison).
# The original scripts hard-coded a conda-env path to TMalign. USalign prints the same
# "TM-score=" lines and alignment block, so the parser in fold_dynamics_miner.tmalign
# reads either; the bare name is the last resort (fails when called, not at import).
TMALIGN_BIN = (os.environ.get("ENSEMBITS_TMALIGN") or shutil.which("TMalign")
               or shutil.which("USalign") or "TMalign")

# The checkpoint whose numbers are reported in the submission (copied from
# REPO_ROOT/output/dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO).
# FIXED(code_fix P18, 2026-10-16): old: only CKPT_DIR, used both as "the reported model"
#   and by the checks that are about this legacy model. New: LEGACY_CKPT_DIR names the
#   legacy (pre-fix) checkpoint explicitly; checks/ about it use LEGACY_CKPT_DIR, so
#   switching CKPT_DIR after the retrain does not change what they test.
LEGACY_CKPT_DIR = CODE_ROOT / "ckpt" / "bioemu_cath_misato"
LEGACY_CKPT_NAME = "dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO"
# FIXED(recon, 2026-10-24): CKPT_DIR switched from the legacy checkpoint to the
#   retrained λ = 10 run (ensembits_bm_lam10_rv; decisions_20261023 Round 3).
#   LEGACY_CKPT_DIR is kept for checks/ that verify the pre-fix model.
CKPT_DIR = CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10"
CKPT_NAME = "ensembits_bm_lam10_rv"
