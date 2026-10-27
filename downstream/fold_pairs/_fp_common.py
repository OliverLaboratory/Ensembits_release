"""Shared file layout and BioEmu-CATH helpers of the fold-matched pair test.

What it does
------------
Gives every script in `downstream/fold_pairs/` one place for

  * the input / output locations (all under REPO_ROOT, outputs under
    REPO_ROOT/output/fold_pairs/);
  * the mapping system -> domain and system -> BioEmu-CATH directory. A *system* is
    one simulation directory, e.g. `cath2_1o82A00`; its *pid* (domain) is the part
    after the first underscore, `1o82A00`. 16 domains were simulated in both deposits
    (`ONE_cath1/cath1_<pid>` and `MSR_cath2/cath2_<pid>`), so a pid can have two
    systems. Same convention as `data_prep/build_bioemu_real_bb.py`;
  * the amino-acid sequence of a system read from its `topology.pdb` (the same rule
    as `seq()` in the original verifier: one letter per CA atom line);
  * the backbone residue filter of `data_prep/build_bioemu_real_bb.py` (residues
    with all of N, CA, C, O, in residue order), so the medoid backbone used for
    tokenization has exactly the residues of the standard `bb_10` cache.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Infrastructure only.

Origin
------
New file written for this codebase. The constants replace the hard-coded relative
paths of `submission_exp/src/rebuttal_analyses/{bioemu_extract_ca, verify_fold_dynamics_pair,
bioemu_token_tokenize, bioemu_token_compare}.py` (`data/reviewer_3_special_corpus/...`,
`output/rebuttal_analyses/...`); `backbone_indices` repeats the inline residue filter of
`data_prep/build_bioemu_real_bb.py::_process` (that file has no function for it).

Used by the reported model: yes (path handling of the fold-pair scripts).

Inputs / outputs
----------------
None by itself.

How to run
----------
Imported by the scripts in this folder; not run directly.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

# ---- inputs ---------------------------------------------------------------
# BioEmu-CATH MD release (Zenodo 10.5281/zenodo.15629740), two deposits.
BIOEMU_ROOT = ROOT / "data" / "reviewer_3_special_corpus" / "bioemu_cath" / "extracted"
# Deposit order of the original Ca extraction (MSR_cath2 first); with system keys the
# order no longer decides which simulation is kept.
DEPOSITS = ("MSR_cath2", "ONE_cath1")
PREFIX_TO_DEPOSIT = {"cath2": "MSR_cath2", "cath1": "ONE_cath1"}
# Standard 10-frame real-backbone cache (data_prep/build_bioemu_real_bb.py), system-keyed.
REAL_BB = ROOT / "data" / "cached_descriptors" / "bioemu_cath_real_bb"

# ---- outputs (everything this test writes) --------------------------------
OUT = ROOT / "output" / "fold_pairs"
CA_CACHE = OUT / "bioemu_ca"                     # <system>.npz  ca (F,L,3), runlens, pid, system
SCORED = OUT / "scored_pairs.json"               # every length-compatible pair, scored
MEDOID_DIR = OUT / "medoids"                     # <system>.npz  medoid real backbone (L,4,3), seq
TOKENS_DIR = OUT / "tokens"                      # <population>/<tokenizer>.npz, manifest.json
REAL_BB_FALLBACK = OUT / "real_bb_fallback"      # bb_10 computed here when REAL_BB lacks a system
COMPARISON = OUT / "token_comparison.json"

SELECTIONS = ("divergent", "similar")


def candidates_path(select: str) -> Path:
    """Mining output for one selection rule (top-N pairs sent to verification)."""
    return OUT / f"candidates_{select}.json"


def verification_path(mode: str) -> Path:
    """Verification output for one mode."""
    return OUT / f"verification_{mode}.json"


def pid_of(system: str) -> str:
    """cath2_1o82A00 -> 1o82A00 (CATH domain). A bare domain id is returned unchanged."""
    return system.split("_", 1)[-1]


def sysdir_of(system: str) -> Path | None:
    """BioEmu-CATH directory of a system (`<BIOEMU_ROOT>/<deposit>/<system>`)."""
    dep = PREFIX_TO_DEPOSIT.get(system.split("_", 1)[0])
    if dep is not None:
        p = BIOEMU_ROOT / dep / system
        return p if p.is_dir() else None
    for dep in DEPOSITS:                       # unknown prefix: search both deposits
        p = BIOEMU_ROOT / dep / system
        if p.is_dir():
            return p
    return None


def list_system_dirs() -> list[str]:
    """All system directories of both deposits (sorted within each deposit)."""
    out = []
    for dep in DEPOSITS:
        out += sorted(glob.glob(str(BIOEMU_ROOT / dep / "cath*_*")))
    return [s for s in out if os.path.isdir(s)]


AA3TO1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
          'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
          'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def topology_sequence(system: str) -> str:
    """One-letter sequence of a system, one letter per CA atom of `topology.pdb`.

    Same parsing as the original verifier's BioEmu branch
    (`verify_fold_dynamics_pair.seq`): residue name of every ATOM line whose atom
    name is CA; unknown residue names become 'X'.
    """
    S = sysdir_of(system)
    if S is None or not (S / "topology.pdb").exists():
        raise FileNotFoundError(f"no topology.pdb for system {system} under {BIOEMU_ROOT}")
    rn = [l[17:20].strip() for l in open(S / "topology.pdb")
          if l.startswith('ATOM') and l[12:16].strip() == 'CA']
    return ''.join(AA3TO1.get(r, 'X') for r in rn)


def backbone_indices(top):
    """Atom indices of N, CA, C, O for the residues that have all four (mdtraj topology).

    Same rule as `data_prep/build_bioemu_real_bb.py::_process` (residues missing any of
    the four atoms are skipped). Returns (n_idx, ca_idx, c_idx, o_idx, res_ids).
    """
    import numpy as np
    by_res: dict = {}
    for atom in top.atoms:
        if atom.name in ("N", "CA", "C", "O"):
            by_res.setdefault(atom.residue.index, {})[atom.name] = atom.index
    res_ids = sorted(r for r, d in by_res.items() if all(n in d for n in ("N", "CA", "C", "O")))
    n_idx = np.array([by_res[r]["N"] for r in res_ids], dtype=int)
    ca_idx = np.array([by_res[r]["CA"] for r in res_ids], dtype=int)
    c_idx = np.array([by_res[r]["C"] for r in res_ids], dtype=int)
    o_idx = np.array([by_res[r]["O"] for r in res_ids], dtype=int)
    return n_idx, ca_idx, c_idx, o_idx, res_ids
