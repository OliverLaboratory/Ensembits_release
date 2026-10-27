"""Binding-site labels for MISATO: the PDBbind pocket mapped onto MISATO residues by sequence and checked by geometry.

What it does
------------
Label definition (unchanged from the paper's source, ProteinShake's MISATO dataset): a residue
is positive if it belongs to the PDBbind v2020 refined-set pocket of the complex
(`<pid>_pocket.pdb`; amino-acid residues with a carbon CA atom). What changes is the mapping
of pocket residues onto the MISATO residue list. ProteinShake (`misato_protein_ligand_
interface.py::_load_binding_site_mask`) set label = 1 where a pocket CA's AUTHOR residue
number equalled MISATO's sequential residue counter (1..L over all chains, with skips),
ignoring chain IDs and insertion codes; the labels sat on the wrong residues for most
complexes (benchmark audit; authors' decision D4: "align them by using the MISATO paper and
resources").

MISATO itself stores no residue numbers, chain IDs or insertion codes (MD.hdf5 per complex:
atoms_number, atoms_type, atoms_residue = residue TYPE index, atoms_element,
molecules_begin_atom_index, trajectory_coordinates, frames_*; the official
`src/data/processing/h5_to_pdb.py` renumbers residues sequentially without chains;
github.com/t7morgen/misato-dataset, Siebenmorgen et al., Nat. Comput. Sci. 2024), so there is
no official map back to PDB numbering. MISATO's starting structures are the PDBbind
complexes, so the map is rebuilt from the PDBbind protein file:

  1. MISATO residues: `build_misato_real_bb.parse_complex` on MD.hdf5 (same residue list as
     `misato_real_bb` and every token cache; sequence and length are checked against it).
     Heavy atoms of each residue in frame 0. The MISATO ligand is the LAST molecule
     (MISATO convention, `src/data/components/datasets.py`: cutoff =
     molecules_begin_atom_index[-1]) plus any other molecule made only of 'MOL' atoms.
  2. PDBbind residues: `<pid>_protein.pdb`, residues with a carbon CA (alt-loc blank / A),
     keyed by (chain, resSeq, iCode), in file order; one-letter code (HID/HIE/HIP -> H,
     CYX -> C, MSE -> M, other non-standard -> X).
  3. Sequence alignment: one global alignment of the MISATO sequence against the PDBbind
     sequence in file order (all chains concatenated; match 2, mismatch -1, gap open -5,
     extend -0.5, free end gaps; Biopython PairwiseAligner). MISATO kept PDBbind's chain
     order and only drops residues, but it sometimes holds fewer homo-oligomer copies than
     the PDBbind file (1LAN: 1 of 4, 3LP4: 2 of 3), where free end gaps put a chain on an
     arbitrary copy.
  4. Copy choice (`choose_copies`): every PDBbind chain is aligned locally to every copy of
     it in the MISATO residue list; chains with several MISATO copies, or sharing one, are
     assigned jointly to MISATO copies (Hungarian algorithm, one copy per chain, swaps and
     extra MISATO copies allowed) to maximise the number of residues within 8 Å of the
     MISATO ligand in frame 0 whose chain residue is within 8 Å of the crystal ligand
     (`<pid>_ligand.mol2`); another copy needs >= 3 agreeing residues and the global
     alignment's copy gets a 3-residue bonus when it has ligand support. Chains whose
     copy cannot be told apart this way are recorded as `ambiguous` (symmetric site) or
     `no_contact`.
  5. Geometric check (MISATO frame 0 Cα vs PDBbind Cα): a global Kabsch superposition on the
     identical aligned pairs, trimmed iteratively (pairs within max(2 Å, 3 x median)); per
     MISATO molecule the median deviation and its own-fit RMSD (a molecule with median
     deviation > 4 Å but own-fit RMSD <= 3 Å is `displaced`, a periodic image or rigid
     drift; recorded only); per positive the deviation after a local
     superposition on the +-7 aligned neighbours (> 3 Å flagged, not removed).
  6. Labels: a pocket residue is transferred to the MISATO residue aligned to it when the
     residue types agree. Unaligned pocket residues are listed with a reason
     (not_in_protein_pdb, type_mismatch, nonstandard_residue, chain_absent, chain_terminus,
     internal_gap). Complexes with no pocket file, no amino-acid pocket residue, a failed
     alignment, or fewer than half of their pocket residues aligned are NOT written to the
     label file (listed in the report and sidecar), so probes skip them instead of seeing an
     all-negative protein.

Validation numbers (report and sidecar; information, not used for the labels): in MISATO
frame 0, the minimum heavy-atom distance of every residue to the MISATO ligand; Jaccard of
the labels with the residues within 10 Å (new and old labels); the 5 Å contact set and its
overlap with the aligned pocket; the pocket-frame shift of the MISATO ligand from the
crystal ligand (`<pid>_ligand.mol2`, after superposing the aligned pocket Cα); and a
cross-check of the residue map against the author ids that
`data/cached_descriptors/misato_chain_breaks.json` (mmCIF-based) gives at every break.
Complexes whose ligand is an amino acid or a peptide (MISATO stores it as a protein molecule,
so its residues are in the residue list and the token input) are listed with their residue
indices; they are labelled 0 (PDBbind pockets hold receptor residues only).

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{Binding Site} (per-residue labels of the
binding-site task, Table \\label{tab:binding_structure}).

Origin
------
New (2026-10-18, benchmark audit D4). Replaces `submission_exp/src/build_misato_binding_
labels.py` (research repo), which copied the `binding_site` field of the ProteinShake Avro
into `data/cached_descriptors/misato_binding_site.npz`.

Used by the reported model: yes (binding-site labels of the Ensembits and baseline probes,
read through `ensembits.misato_binding_labels`).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/external/pdbbind_v2020_refined/refined-set/<pid>/<pid>_{pocket,protein}.pdb,
          <pid>_ligand.mol2 and INDEX_refined_data.2020 (PDBbind v2020 refined set, the
          archive ProteinShake downloads: sandbox.zenodo.org/records/327402,
          PDBbind_v2020_refined.tar.gz md5 e9412f4b5b065a4eab6958b3bb1c49a2; --pdbbind-dir)
          data/misato/MD.hdf5, data/cached_descriptors/misato_real_bb/<PID>.npz
          data/cached_descriptors/misato_binding_site.npz   (old labels: complex set and
                                                             old-vs-new comparison only)
          data/cached_descriptors/misato_chain_breaks.json  (cross-check only)
  writes  data/annotations/misato_binding_site_v2.npz  ({PID: int8[L]}, + .provenance.json)
          data/annotations/misato_binding_site_v2.report.json  (per complex: status,
              positives with their PDBbind ids, unaligned pocket residues and reasons,
              geometry flags, ligand facts, validation numbers)
  Refuses existing outputs unless --overwrite.

How to run
----------
    python data_prep/build_misato_binding_site_labels.py --workers 48 [--overwrite]
    python data_prep/build_misato_binding_site_labels.py --pids 1A4W,3V2P \\
        --out <scratch>/x.npz --report <scratch>/x.json        # partial run (flagged partial)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # build_misato_real_bb
from ensembits.paths import REPO_ROOT as ROOT, EXTERNAL     # noqa: E402
from ensembits.misato_binding_labels import (BINDING_LABELS_SCHEMA,  # noqa: E402
                                             MISATO_BINDING_SITE, MISATO_BINDING_SITE_LEGACY)

PDBBIND_DIR = Path(os.environ.get("ENSEMBITS_PDBBIND_DIR", EXTERNAL / "pdbbind_v2020_refined"))
H5_PATH = ROOT / "data" / "misato" / "MD.hdf5"
REAL_BB_DIR = ROOT / "data" / "cached_descriptors" / "misato_real_bb"
CHAIN_BREAKS = ROOT / "data" / "cached_descriptors" / "misato_chain_breaks.json"
REPORT = Path(str(MISATO_BINDING_SITE).replace(".npz", ".report.json"))

RULE_VERSION = "2.1"   # 2.1 (2026-10-19): joint copy assignment (swaps, extra MISATO copies)
ALIGN = dict(match=2.0, mismatch=-1.0, gap_open=-5.0, gap_extend=-0.5)
CORE_ITERS = 4                 # trimmed Kabsch iterations
CORE_MIN_A = 2.0               # keep pairs within max(2 Å, 3 x median deviation)
DISPLACED_MEDIAN_A = 4.0       # molecule median deviation above this = displaced ...
MOL_FIT_RMSD_A = 3.0           # ... when its own-fit RMSD is at most this (rigid move)
LOCAL_HALF_WINDOW = 7          # local superposition on +-7 aligned neighbours
LOCAL_FLAG_A = 3.0             # positives with a larger local deviation are flagged
MIN_POCKET_ALIGNED = 0.5       # fewer aligned pocket residues -> complex not labelled
COPY_NEAR_A = 8.0              # homo-oligomer copy choice: ligand neighbourhood cutoff
COPY_MIN_SCORE = 0.3           # ... a copy is chosen / verified with Jaccard >= 0.3
COPY_MARGIN = 0.2              # ... and >= 0.2 above the other copies
COPY_MIN_GAIN = 3              # a block leaves its aligned chain only for >= 3 more agreeing residues
CONTACT_A = (5.0, 8.0, 10.0)        # validation distance sets (MISATO frame 0, heavy atoms)

# Information for the authors (no label change), from the D4 re-audit (2026-10-19).
KNOWN_NOTES = {
    "several_ligand_copies": {
        "pids": ["1HSL", "2PYY", "2XXX", "4IO2", "3ERY"],
        "note": "the PDBbind entry holds several copies of the ligand (one per receptor copy); "
                "MISATO keeps one ligand molecule (the last molecule) and PDBbind's pocket "
                "surrounds the crystal ligand of <pid>_ligand.mol2; the joint copy assignment "
                "(rule 2.1) puts the pocket on the receptor copy next to the MISATO ligand"},
    "known_oddities": {
        "pids": ["4J44", "4J45", "4J46", "4J47", "5EH8"],
        "note": "4J44-4J47: MISATO's last molecule (the ligand by MISATO convention) is a "
                "76-residue protein molecule, not PDBbind's 4-mer peptide, so distances to "
                "the 'MISATO ligand' do not describe the pocket; 5EH8: the MISATO frame-0 "
                "ligand lies ~34 A from its crystal pose in the pocket frame; labels kept "
                "(PDBbind pocket, correct residue map)"},
}
ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "MSE": "M"}
STANDARD = set(ONE) - {"MSE"}
RULE = ("label = PDBbind v2020 refined-set pocket residue (<pid>_pocket.pdb, residues with a "
        "carbon CA); MISATO residues (parse_complex, = misato_real_bb order) are aligned to "
        "<pid>_protein.pdb residues (carbon CA, alt-loc blank/A, file order, all chains "
        "concatenated) by one global alignment with free end gaps (match 2, mismatch -1, open "
        "-5, extend -0.5); a pocket residue is transferred when it is aligned to a MISATO "
        "residue of the same type, after assigning PDBbind chains with several MISATO "
        "copies (local alignment) jointly to those copies (Hungarian, swaps and extra "
        "MISATO copies allowed) by the number of residues within 8 A of the MISATO frame-0 "
        "ligand whose chain residue is within 8 A of the crystal ligand (other copies need "
        ">= 3 such residues; the global-alignment copy +3 when it has any); complexes with "
        "no pocket file / no amino-acid pocket "
        "residue / failed alignment / < 50 % of pocket residues aligned are not labelled; "
        "geometry (frame-0 Ca vs PDBbind Ca, trimmed global Kabsch, per-molecule fit, local "
        "+-7 superposition) is recorded as flags only")


# ---------------------------------------------------------------- parsing
def parse_pdb_residues(lines) -> list:
    """Residues of PDB ATOM/HETATM lines in file order (waters skipped; alt-loc blank / A).

    -> [{key: (chain, resseq, icode), resname, ca: xyz | None (carbon CA only), heavy:
    (n, 3), record}]."""
    res, order = {}, []
    for ln in lines:
        if ln[:6] not in ("ATOM  ", "HETATM") or ln[17:20].strip() in ("HOH", "WAT"):
            continue
        if ln[16] not in (" ", "A"):
            continue
        key = (ln[21], int(ln[22:26]), ln[26].strip())
        name = ln[12:16].strip()
        el = (ln[76:78].strip() or name.lstrip("0123456789")[:1]).upper()
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        r = res.get(key)
        if r is None:
            r = res[key] = {"key": key, "resname": ln[17:20].strip(), "ca": None, "heavy": [],
                            "record": ln[:6].strip()}
            order.append(key)
        if el not in ("H", "D"):
            r["heavy"].append(xyz)
        if name == "CA" and el == "C" and r["ca"] is None:
            r["ca"] = xyz
    out = []
    for k in order:
        r = res[k]
        r["heavy"] = np.asarray(r["heavy"], dtype=np.float64).reshape(-1, 3)
        out.append(r)
    return out


def amino_acid_residues(residues) -> list:
    """Residues with a carbon CA atom (amino acids; ions named CA are excluded)."""
    return [r for r in residues if r["ca"] is not None]


def parse_mol2_heavy(lines) -> np.ndarray:
    """Heavy-atom coordinates of a mol2 file (SYBYL type H / H.* skipped)."""
    xs, on = [], False
    for ln in lines:
        if ln.startswith("@<TRIPOS>ATOM"):
            on = True; continue
        if ln.startswith("@<TRIPOS>"):
            on = False; continue
        t = ln.split()
        if on and len(t) >= 6 and t[5].split(".")[0].upper() != "H":
            xs.append((float(t[2]), float(t[3]), float(t[4])))
    return np.asarray(xs, dtype=np.float64).reshape(-1, 3)


def read_index(path: Path) -> dict:
    """PDBbind INDEX file -> {PID: ligand name}."""
    out = {}
    if not path.exists():
        return out
    for ln in open(path):
        if ln.startswith("#") or "//" not in ln:
            continue
        pre, post = ln.split("//", 1)
        name = post[post.rfind("(") + 1: post.rfind(")")] if "(" in post else ""
        out[pre.split()[0].upper()] = name
    return out


# ---------------------------------------------------------------- alignment and geometry
def _aligner():
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = ALIGN["match"], ALIGN["mismatch"]
    al.open_gap_score, al.extend_gap_score = ALIGN["gap_open"], ALIGN["gap_extend"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:                                                 # Biopython >= 1.86 names
            al.end_insertion_score = 0.0
            al.end_deletion_score = 0.0
        except AttributeError:                               # older Biopython
            al.target_end_gap_score = 0.0
            al.query_end_gap_score = 0.0
    return al


def align_pairs(mseq: str, pseq: str) -> np.ndarray:
    """Aligned (MISATO index, PDBbind index) pairs of one global alignment, (n, 2) int."""
    if not mseq or not pseq:
        return np.zeros((0, 2), dtype=np.int64)
    a = _aligner().align(mseq, pseq)[0]
    pairs = [(s1 + t, s2 + t) for (s1, e1), (s2, _) in zip(*a.aligned) for t in range(e1 - s1)]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t with P @ R.T + t ~ Q (least squares)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - pc @ R.T


def trimmed_superpose(M: np.ndarray, Q: np.ndarray, use: np.ndarray):
    """Kabsch of M onto Q on the pairs `use`, re-fitted CORE_ITERS times on the pairs within
    max(CORE_MIN_A, 3 x median deviation). -> (R, t, deviation of every pair)."""
    keep = use.copy()
    R, t = np.eye(3), np.zeros(3)
    dev = np.full(len(M), np.inf)
    for _ in range(CORE_ITERS):
        if keep.sum() < 3:
            break
        R, t = kabsch(M[keep], Q[keep])
        dev = np.linalg.norm(M @ R.T + t - Q, axis=1)
        keep = use & (dev < max(CORE_MIN_A, 3.0 * float(np.median(dev[use]))))
    return R, t, dev


def local_deviation(M: np.ndarray, Q: np.ndarray, q: int, half: int = LOCAL_HALF_WINDOW):
    """Deviation of pair q after superposing its +-half aligned neighbours (None if < 5)."""
    lo, hi = max(0, q - half), min(len(M), q + half + 1)
    if hi - lo < 5:
        return None
    R, t = kabsch(M[lo:hi], Q[lo:hi])
    return float(np.linalg.norm(M[q] @ R.T + t - Q[q]))


def min_dist(A: np.ndarray, B: np.ndarray) -> float:
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    return float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1).min()))


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    """Rank AUROC of binary `y` against `score` (None without both classes)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return None if u == 0 else float((a & b).sum() / u)


def transfer_pocket(pocket_keys, prot, pairs: np.ndarray, mseq: str, L: int):
    """Pocket keys -> (labels int8[L], positives [(i, key, resname)], unaligned
    [(key, resname, reason)]) through the alignment pairs (MISATO i, PDBbind j)."""
    key2j = {r["key"]: j for j, r in enumerate(prot)}
    p2m = {int(j): int(i) for i, j in pairs}
    aligned_chains = {prot[int(j)]["key"][0] for _, j in pairs}
    chain_span = {}
    for _, j in pairs:
        c = prot[int(j)]["key"][0]
        lo, hi = chain_span.get(c, (int(j), int(j)))
        chain_span[c] = (min(lo, int(j)), max(hi, int(j)))
    labels = np.zeros(L, dtype=np.int8)
    pos, un = [], []
    for key, rn in pocket_keys:
        j = key2j.get(key)
        if j is None:
            un.append((key, rn, "not_in_protein_pdb")); continue
        i = p2m.get(j)
        if i is None:
            if rn not in STANDARD:
                why = "nonstandard_residue"
            elif key[0] not in aligned_chains:
                why = "chain_absent"
            else:
                lo, hi = chain_span[key[0]]
                why = "chain_terminus" if (j < lo or j > hi) else "internal_gap"
            un.append((key, rn, why)); continue
        if ONE.get(rn, "X") != mseq[i]:
            un.append((key, rn, "type_mismatch")); continue
        labels[i] = 1
        pos.append((i, key, rn))
    return labels, pos, un


def _local_aligner():
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner()
    al.mode = "local"
    al.match_score, al.mismatch_score = ALIGN["match"], ALIGN["mismatch"]
    al.open_gap_score, al.extend_gap_score = ALIGN["gap_open"], ALIGN["gap_extend"]
    return al


def chain_occurrences(cseq: str, mseq: str, max_occ: int = 48) -> list:
    """Every copy of one PDBbind chain sequence in the MISATO residue list.

    Repeated local alignment of `cseq` against the MISATO sequence, masking the span of each
    copy found; a copy needs >= 90 % identical aligned pairs and >= max(5, half the chain)
    aligned residues. -> [array (n, 2) of (MISATO index, chain-local index)]."""
    al = _local_aligner()
    masked = list(mseq)
    out = []
    for _ in range(max_occ):
        try:
            a = al.align("".join(masked), cseq)[0]
        except (IndexError, ValueError):
            break
        prs = [(s1 + t, s2 + t) for (s1, e1), (s2, _) in zip(*a.aligned) for t in range(e1 - s1)]
        if not prs:
            break
        ident = sum(masked[i] == cseq[j] for i, j in prs)
        if ident < 0.9 * len(prs) or len(prs) < max(5, 0.5 * len(cseq)):
            break
        out.append(np.asarray(prs, dtype=np.int64))
        for i in range(prs[0][0], prs[-1][0] + 1):
            masked[i] = "#"
    return out


def choose_copies(pairs: np.ndarray, prot, chain_index, dmis: np.ndarray, dcry: np.ndarray,
                  mseq: str, exclude=(), near: float = COPY_NEAR_A) -> tuple:
    """Assign PDBbind chains to the MISATO copies that share their ligand neighbourhood.

    The order-preserving global alignment cannot tell homo-oligomer copies apart: it puts a
    MISATO chain on an arbitrary PDBbind copy when MISATO holds fewer copies (1LAN: 1 of 4),
    keeps the copy order when the ligand sits on the other copy (1XFF: A <-> B), and leaves
    extra MISATO copies unaligned when MISATO holds more copies than the PDBbind file (6IM4:
    two MISATO copies of chain A). Here every PDBbind chain is aligned locally to every copy
    of it in the MISATO residue list (`chain_occurrences`). A chain is contested when it has
    several MISATO copies or shares one with another chain; contested chains and their MISATO
    copies are assigned jointly (Hungarian algorithm, one copy per chain, swaps allowed) to
    maximise the number of residues within `near` Å of the MISATO ligand (frame 0) whose chain
    residue is within `near` Å of the crystal ligand; another copy counts only with >=
    COPY_MIN_GAIN agreeing residues, and the alignment's own copy gets a bonus of
    COPY_MIN_GAIN when it has any (plus a tie-break), so a change needs a real gain. Chains that keep their copy keep
    the global-alignment pairs; moved chains take the local-alignment pairs. MISATO copies
    overlapping a non-contested chain's residues are not used, nor are the MISATO residues in
    `exclude` (the ligand molecule when the ligand is a peptide or protein: 4J44's ligand is a
    76-residue molecule with the receptor's sequence).
    FIXED(re-audit of D4, 2026-10-19): the first version moved one segment at a time onto an
    unused PDBbind copy and undid colliding moves, so swaps and extra MISATO copies were never
    resolved (25 complexes with the pocket on the wrong copy).

    Every contested chain is classed by its Jaccard (MISATO-ligand vs crystal-ligand
    neighbourhood) on the chosen copy against its other copies: `supported` (>=
    COPY_MIN_SCORE and >= COPY_MARGIN above the others), `ambiguous` (another copy within
    COPY_MARGIN, e.g. a ligand on the two-fold axis of HIV protease), `no_contact`
    (< COPY_MIN_SCORE).
    -> (new pairs, {"chains_contested", "reassigned": [...], "supported" / "ambiguous" /
    "no_contact": [(i0, i1)]} with MISATO index ranges)."""
    from scipy.optimize import linear_sum_assignment
    info = {"chains_contested": 0, "reassigned": [], "supported": [], "ambiguous": [],
            "no_contact": []}
    chains = list(dict.fromkeys(r["key"][0] for r in prot))
    cj = {c: np.array([j for j, r in enumerate(prot) if r["key"][0] == c]) for c in chains}
    cseq = {c: "".join(ONE.get(prot[j]["resname"], "X") for j in cj[c]) for c in chains}
    chain_of_j = np.array([r["key"][0] for r in prot])
    orig = {c: set(int(i) for i, j in pairs if chain_of_j[j] == c) for c in chains}
    excl = set(int(i) for i in exclude)
    mseq_x = "".join("#" if i in excl else x for i, x in enumerate(mseq))
    occ_cache, occ = {}, {}
    for c in chains:
        if cseq[c] not in occ_cache:
            occ_cache[cseq[c]] = chain_occurrences(cseq[c], mseq_x) if len(cseq[c]) >= 5 else []
        occ[c] = [np.column_stack([o[:, 0], cj[c][o[:, 1]]]) for o in occ_cache[cseq[c]]]

    # distinct MISATO copies over all chains (same copy = >= 50 % shared MISATO residues)
    copies = []                                       # list of MISATO residue sets
    def copy_id(iset):
        for k, cs in enumerate(copies):
            if len(iset & cs) >= 0.5 * min(len(iset), len(cs)):
                return k
        copies.append(set(iset)); return len(copies) - 1
    opt = {c: {} for c in chains}                     # chain -> {copy id: pairs}
    for c in chains:
        for o in occ[c]:
            opt[c].setdefault(copy_id(set(o[:, 0].tolist())), o)
    users = {}
    for c in chains:
        for k in opt[c]:
            users.setdefault(k, set()).add(c)
    contested = [c for c in chains if len(opt[c]) >= 2 or any(len(users[k]) > 1 for k in opt[c])]
    if not contested:
        return pairs, info
    fixed_i = set().union(*[orig[c] for c in chains if c not in contested])
    ks = sorted({k for c in contested for k in opt[c]
                 if len(copies[k] & fixed_i) <= 0.1 * len(copies[k])})
    info["chains_contested"] = len(contested)

    def own_copy(c):
        if not orig[c]:
            return None
        best = max(opt[c], key=lambda k: len(copies[k] & orig[c]), default=None)
        return best if best is not None and len(copies[best] & orig[c]) >= 0.5 * len(orig[c]) else None

    def near_sets(o):
        return dmis[o[:, 0]] < near, dcry[o[:, 1]] < near
    S = np.full((len(contested), max(len(ks), 1)), -1e9)
    for a, c in enumerate(contested):
        oc = own_copy(c)
        for b, k in enumerate(ks):
            if k in opt[c]:
                A, Bn = near_sets(opt[c][k])
                inter = float((A & Bn).sum())
                if k == oc:          # own copy: bonus when it has ligand support, tie-break
                    S[a, b] = inter + (COPY_MIN_GAIN if inter > 0 else 0) + 0.01
                else:                # another copy counts only with >= COPY_MIN_GAIN residues
                    S[a, b] = inter if inter >= COPY_MIN_GAIN else 0.0
    rows, cols = linear_sum_assignment(-S)
    choice = {contested[r]: ks[cc] for r, cc in zip(rows, cols) if S[r, cc] > -1e8}

    new = [(int(i), int(j)) for i, j in pairs if chain_of_j[j] not in contested]
    for c in contested:
        oc = own_copy(c)
        k = choice.get(c)
        if k is None:
            if orig[c]:
                info["reassigned"].append({"chain": c, "from": [min(orig[c]), max(orig[c]) + 1],
                                           "to": None})
            continue
        if k == oc:
            new += [(int(i), int(j)) for i, j in pairs if chain_of_j[j] == c]
        else:
            o = opt[c][k]
            new += [(int(i), int(j)) for i, j in o]
            info["reassigned"].append({
                "chain": c, "from": [min(orig[c]), max(orig[c]) + 1] if orig[c] else None,
                "to": [int(o[0, 0]), int(o[-1, 0]) + 1]})
        jac = {}
        for kk, o in opt[c].items():
            A, Bn = near_sets(o)
            u = int((A | Bn).sum())
            jac[kk] = float((A & Bn).sum() / u) if u else 0.0
        others = max([v for kk, v in jac.items() if kk != k], default=0.0)
        cls = ("no_contact" if jac[k] < COPY_MIN_SCORE else
               "ambiguous" if jac[k] - others < COPY_MARGIN else "supported")
        o = opt[c][k]
        info[cls].append((int(o[0, 0]), int(o[-1, 0]) + 1))
    # one MISATO residue, one PDBbind residue
    new = sorted(set(new))
    seen_i, seen_j, out = set(), set(), []
    for i, j in new:
        if i in seen_i or j in seen_j:
            continue
        seen_i.add(i); seen_j.add(j); out.append((i, j))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2), info


# ---------------------------------------------------------------- one complex
def _misato(pid: str, h5, rb_dir: Path):
    import build_misato_real_bb as B
    g = h5[pid]
    pc = B.parse_complex(g)
    if pc is None:
        raise ValueError("no protein residue in MD.hdf5")
    an = g["atoms_number"][:]; ar = g["atoms_residue"][:]
    mb = [int(x) for x in g["molecules_begin_atom_index"][:]]
    X = np.asarray(g["trajectory_coordinates"][0], dtype=np.float64)
    N = len(an); ends_mol = mb[1:] + [N]
    seq = pc["seq"]; L = len(seq)
    with np.load(rb_dir / f"{pid}.npz", allow_pickle=False) as z:
        if str(z["seq"]) != seq or z["ca_8"].shape[1] != L:
            raise ValueError("MD.hdf5 parse differs from misato_real_bb (rebuild it)")
    n = pc["n"]; ci = pc["chain_index"]
    res_end = []
    for i in range(L):
        mi = int(np.searchsorted(mb, n[i], "right")) - 1
        e = ends_mol[mi]
        if i + 1 < L and ci[i + 1] == ci[i]:
            e = int(n[i + 1])
        res_end.append(e)
    heavy = [np.flatnonzero(an[a:b] > 1) + a for a, b in zip(n, res_end)]
    lig_mols = sorted({len(mb) - 1} | {k for k in range(len(mb)) if (ar[mb[k]:ends_mol[k]] == 0).all()})
    lig = np.concatenate([np.arange(mb[k], ends_mol[k]) for k in lig_mols])
    lig_heavy = lig[an[lig] > 1]
    last_is_mol = bool((ar[mb[-1]:] == 0).all())
    lig_res = np.flatnonzero(n >= mb[-1]).tolist()          # ligand residues inside the list
    return dict(seq=seq, L=L, X=X, ca=X[pc["ca"]], heavy=heavy, chain_index=ci,
                lig_xyz=X[lig_heavy], lig_mols=lig_mols, n_molecules=len(mb),
                ligand_amino_acid=not last_is_mol, ligand_residue_index=lig_res)


def label_complex(pid: str, h5, rb_dir: Path, pdbbind_dir: Path, breaks: dict | None,
                  old: np.ndarray | None, index_names: dict) -> tuple:
    """-> (labels int8[L] or None, report record)."""
    rec = {"status": "ok"}
    m = _misato(pid, h5, rb_dir)
    L, mseq = m["L"], m["seq"]
    rec.update(L=L, n_molecules=m["n_molecules"], ligand={
        "molecules": m["lig_mols"], "n_heavy": int(len(m["lig_xyz"])),
        "amino_acid_or_peptide": m["ligand_amino_acid"],
        "residue_index": m["ligand_residue_index"] if m["ligand_amino_acid"] else [],
        "pdbbind_index_name": index_names.get(pid)})
    d = pdbbind_dir / "refined-set" / pid.lower()
    pocket_f, prot_f = d / f"{pid.lower()}_pocket.pdb", d / f"{pid.lower()}_protein.pdb"
    if not pocket_f.exists():
        rec["status"] = "no_pocket_file"; return None, rec
    if not prot_f.exists():
        rec["status"] = "no_protein_file"; return None, rec
    pocket = amino_acid_residues(parse_pdb_residues(open(pocket_f)))
    prot = amino_acid_residues(parse_pdb_residues(open(prot_f)))
    pocket_keys = [(r["key"], r["resname"]) for r in pocket]
    rec["n_pocket"] = len(pocket_keys)
    if not pocket_keys:
        rec["status"] = "empty_pocket"; return None, rec
    pseq = "".join(ONE.get(r["resname"], "X") for r in prot)
    pairs = align_pairs(mseq, pseq)
    rec["n_pdbbind_residues"] = len(prot)
    rec["n_aligned"] = int(len(pairs))
    ident = np.array([mseq[i] == pseq[j] for i, j in pairs], dtype=bool)
    rec["identity"] = float(ident.mean()) if len(pairs) else 0.0
    if len(pairs) < 3 or ident.sum() < 3:
        rec["status"] = "alignment_failed"; return None, rec

    # distances to the ligand: MISATO residues (frame 0, MISATO ligand) and PDBbind residues
    # (crystal ligand), both heavy atoms
    X, LX = m["X"], m["lig_xyz"]
    dmin = np.array([min_dist(X[h], LX) for h in m["heavy"]])
    lig_f = d / f"{pid.lower()}_ligand.mol2"
    cx = parse_mol2_heavy(open(lig_f)) if lig_f.exists() else np.zeros((0, 3))
    dcry = np.array([min_dist(r["heavy"], cx) for r in prot]) if len(cx) else None

    # homo-oligomer copies: the order-preserving alignment cannot tell copies apart (free end
    # gaps, MISATO may hold fewer copies than PDBbind); choose per segment the copy whose
    # ligand neighbourhood matches (choose_copies)
    copy_rec = {"chains_contested": 0, "reassigned": [], "supported": [],
                "ambiguous": [], "no_contact": []}
    if dcry is not None:
        pairs, copy_rec = choose_copies(pairs, prot, m["chain_index"], dmin, dcry, mseq,
                                        exclude=m["ligand_residue_index"] if m["ligand_amino_acid"] else ())
        ident = np.array([mseq[i] == pseq[j] for i, j in pairs], dtype=bool)
    rec["copies"] = copy_rec

    labels, pos, un = transfer_pocket(pocket_keys, prot, pairs, mseq, L)
    rec["unaligned"] = [[k[0], k[1], k[2], rn, why] for k, rn, why in un]
    rec["n_pos"] = int(labels.sum())
    rec["positives"] = [[i, k[0], k[1], k[2], rn] for i, k, rn in pos]
    pos_idx = set(int(i) for i, _, _ in pos)
    for cls in ("ambiguous", "no_contact"):
        rec[f"n_pos_copy_{cls}"] = int(sum(1 for s, e in copy_rec.get(cls, [])
                                           for i in range(s, e) if i in pos_idx))

    # geometry: global trimmed superposition, per-molecule checks, local deviation
    M = m["ca"][pairs[:, 0]]
    Q = np.array([prot[j]["ca"] for j in pairs[:, 1]], dtype=np.float64)
    R, t, dev = trimmed_superpose(M, Q, ident)
    rec["global_dev_median"] = float(np.median(dev[ident]))
    ci = m["chain_index"][pairs[:, 0]]
    displaced = []
    for c in np.unique(ci):
        sel = (ci == c) & ident
        if sel.sum() < 3:
            continue
        med = float(np.median(dev[sel]))
        if med > DISPLACED_MEDIAN_A:
            R2, t2 = kabsch(M[sel], Q[sel])
            own = float(np.sqrt(((M[sel] @ R2.T + t2 - Q[sel]) ** 2).sum(1).mean()))
            displaced.append({"chain_index": int(c), "median_dev": round(med, 2),
                              "own_fit_rmsd": round(own, 2), "rigid": own <= MOL_FIT_RMSD_A})
    rec["displaced_molecules"] = displaced
    q_of = {int(i): q for q, (i, _) in enumerate(pairs)}
    loc = [local_deviation(M, Q, q_of[i]) for i, _, _ in pos]
    rec["n_pos_local_flag"] = int(sum(1 for x in loc if x is not None and x > LOCAL_FLAG_A))
    rec["pos_local_dev_max"] = max([x for x in loc if x is not None], default=None)
    rec["pos_global_dev"] = [round(float(dev[q_of[i]]), 2) for i, _, _ in pos]

    # validation in MISATO frame 0 (ligand = MISATO ligand)
    lig_res = set(m["ligand_residue_index"]) if m["ligand_amino_acid"] else set()
    rec_mask = np.array([i not in lig_res for i in range(L)])   # receptor residues only
    y = labels.astype(bool)
    near = {c: (dmin < c) & rec_mask for c in CONTACT_A}
    val = {"dmin_pos": [round(float(x), 2) for x in dmin[y]],
           "n_within_10": int(near[10.0].sum()), "n_within_5": int(near[5.0].sum()),
           "n_within_5_in_pocket": int((near[5.0] & y).sum()),
           "jaccard_10_new": jaccard(y, near[10.0]), "jaccard_8_new": jaccard(y, near[8.0]),
           "jaccard_5_new": jaccard(y, near[5.0]),
           # per-complex AUROC of the labels against -distance to the MISATO ligand (receptor)
           "auroc_dist_new": auroc(y[rec_mask], -dmin[rec_mask])}
    if old is not None:
        o = np.asarray(old).astype(bool)
        val["old_len_ok"] = bool(len(o) == L)
        if len(o) == L:
            val["jaccard_10_old"] = jaccard(o, near[10.0])
            val["jaccard_8_old"] = jaccard(o, near[8.0])
            val["auroc_dist_old"] = auroc(o[rec_mask], -dmin[rec_mask])
            val["n_pos_old"] = int(o.sum())
            val["dmin_pos_old"] = [round(float(x), 2) for x in dmin[o]]
    rec["validation"] = val
    if lig_res:
        rec["ligand"]["residues_labelled_positive"] = int(sum(labels[i] for i in lig_res))
        rec["ligand"]["residues_aligned_to_pdbbind_protein"] = int(
            sum(1 for i, _ in pairs if int(i) in lig_res))

    # ligand shift relative to the pocket (pocket-frame superposition on aligned positives)
    if len(pos) >= 3 and len(cx) and len(LX):
        P_ = m["ca"][[i for i, _, _ in pos]]
        key2r = {r["key"]: r for r in prot}
        Q_ = np.array([key2r[k]["ca"] for _, k, _ in pos])
        R3, t3 = kabsch(P_, Q_)
        if len(cx):
            rec["ligand"]["pocket_frame_centroid_shift"] = round(float(np.linalg.norm(
                (LX @ R3.T + t3).mean(0) - cx.mean(0))), 2)
            dcr = [min_dist(X[m["heavy"][i]] @ R3.T + t3, cx) for i, _, _ in pos]
            val["dmin_pos_to_crystal_ligand"] = [round(x, 2) for x in dcr]

    # cross-check with the mmCIF author ids of misato_chain_breaks.json
    if breaks is not None and pid in breaks:
        m2p = {int(i): int(j) for i, j in pairs}
        cnt, devs = Counter(), {}
        alt_ca = {(r["key"][0], f"{r['key'][1]}{r['key'][2]}"): np.asarray(r["ca"]) for r in prot}
        for b in breaks[pid].get("breaks", []):
            for side, i in (("left", b["pos"] - 1), ("right", b["pos"])):
                auth = b.get(f"{side}_auth")
                if auth is None:
                    continue
                j = m2p.get(int(i))
                if j is None:
                    cnt["unmapped"] += 1; continue
                ch, rs, ic = prot[j]["key"]
                same_num = f"{rs}{ic}" == str(auth)
                same_ch = ch == b.get(f"{side}_chain")
                kind = ("agree" if same_num and same_ch else
                        "other_copy_same_number" if same_num else "disagree")
                cnt[kind] += 1
                if kind != "agree":      # global deviation there: our map vs the table's residue
                    alt = alt_ca.get((b.get(f"{side}_chain"), str(auth)))
                    x = m["ca"][int(i)] @ R.T + t
                    devs.setdefault(kind, []).append(
                        [round(float(dev[q_of[int(i)]]), 2),
                         None if alt is None else round(float(np.linalg.norm(x - alt)), 2)])
        rec["chain_breaks_crosscheck"] = dict(cnt)
        rec["chain_breaks_crosscheck_dev"] = devs

    n_pk = len(pocket_keys)
    if rec["n_pos"] < MIN_POCKET_ALIGNED * n_pk:
        rec["status"] = "pocket_mostly_unaligned"
        return None, rec
    return labels, rec


def _job(args):
    pid, rb_dir, pdbbind_dir, h5_path, old_path, breaks_path, index_names = args
    import h5py
    try:
        with h5py.File(h5_path, "r") as h5:
            old = None
            if old_path is not None and Path(old_path).exists():
                with np.load(old_path, allow_pickle=False) as z:
                    old = z[pid] if pid in z.files else None
            breaks = _BREAKS_CACHE.get(breaks_path)
            if breaks is None and breaks_path is not None and Path(breaks_path).exists():
                breaks = json.loads(Path(breaks_path).read_text())["complexes"]
                _BREAKS_CACHE[breaks_path] = breaks
            lab, rec = label_complex(pid, h5, Path(rb_dir), Path(pdbbind_dir), breaks, old,
                                     index_names)
        return pid, lab, rec, None
    except Exception as e:                          # noqa: BLE001
        return pid, None, None, f"{type(e).__name__}: {e}"


_BREAKS_CACHE: dict = {}


# ---------------------------------------------------------------- summary
def _q(x, qs=(0.5, 0.9, 0.99)):
    x = np.asarray(x, dtype=float)
    return None if x.size == 0 else {f"q{int(q * 100)}": round(float(np.quantile(x, q)), 2)
                                     for q in qs}


def summary(records: dict) -> dict:
    """Old-vs-new validation and coverage counts over all complexes."""
    st = Counter(r["status"] for r in records.values())
    ok = {p: r for p, r in records.items() if r["status"] == "ok"}
    allv = {p: r for p, r in records.items() if "validation" in r}
    un = Counter(u[4] for r in records.values() for u in r.get("unaligned", []))
    n_pk = sum(r.get("n_pocket", 0) for r in records.values())
    dpos = [x for r in ok.values() for x in r["validation"]["dmin_pos"]]
    dold = [x for r in allv.values() for x in r["validation"].get("dmin_pos_old", [])]
    dcr = [x for r in ok.values() for x in r["validation"].get("dmin_pos_to_crystal_ligand", [])]
    j_new = [r["validation"]["jaccard_10_new"] for r in ok.values()
             if r["validation"]["jaccard_10_new"] is not None]
    j_old = [r["validation"].get("jaccard_10_old") for r in allv.values()
             if r["validation"].get("jaccard_10_old") is not None]
    j8n = [r["validation"]["jaccard_8_new"] for r in ok.values()
           if r["validation"]["jaccard_8_new"] is not None]
    j8o = [r["validation"].get("jaccard_8_old") for r in allv.values()
           if r["validation"].get("jaccard_8_old") is not None]
    au_n = [r["validation"]["auroc_dist_new"] for r in ok.values()
            if r["validation"].get("auroc_dist_new") is not None]
    au_o = [r["validation"]["auroc_dist_old"] for r in allv.values()
            if r["validation"].get("auroc_dist_old") is not None]
    n5 = sum(r["validation"]["n_within_5"] for r in ok.values())
    n5in = sum(r["validation"]["n_within_5_in_pocket"] for r in ok.values())
    j5 = [r["validation"]["jaccard_5_new"] for r in ok.values()
          if r["validation"]["jaccard_5_new"] is not None]
    pep = {p: r for p, r in records.items() if r.get("ligand", {}).get("amino_acid_or_peptide")}
    shift = [r["ligand"]["pocket_frame_centroid_shift"] for r in ok.values()
             if "pocket_frame_centroid_shift" in r.get("ligand", {})]
    xc, xdev = Counter(), {}
    for r in records.values():
        xc.update(r.get("chain_breaks_crosscheck", {}))
        for k, v in r.get("chain_breaks_crosscheck_dev", {}).items():
            xdev.setdefault(k, []).extend(v)
    return {
        "n_complexes": len(records), "status": dict(st),
        "not_labelled": {s: sorted(p for p, r in records.items() if r["status"] == s)
                         for s in st if s != "ok"},
        "n_pocket_residues": n_pk,
        "n_pocket_residues_aligned": sum(r.get("n_pos", 0) for r in records.values()),
        "unaligned_pocket_residues_by_reason": dict(un),
        "n_positives_labelled": sum(r["n_pos"] for r in ok.values()),
        "old_labels": {"n_complexes_all_negative": sum(
            1 for r in allv.values() if r["validation"].get("n_pos_old") == 0),
            "jaccard_10_median": round(float(np.median(j_old)), 3) if j_old else None,
            "jaccard_8_median": round(float(np.median(j8o)), 3) if j8o else None,
            "auroc_dist_median": round(float(np.median(au_o)), 3) if au_o else None,
            "n_auroc_dist_lt_0.5": sum(1 for x in au_o if x < 0.5),
            "frac_positives_within_10A": round(float(np.mean(np.array(dold) < 10)), 4) if dold else None,
            "dmin_positives": _q(dold)},
        "new_labels": {"jaccard_10_median": round(float(np.median(j_new)), 3) if j_new else None,
                       "jaccard_10_mean": round(float(np.mean(j_new)), 3) if j_new else None,
                       "jaccard_8_median": round(float(np.median(j8n)), 3) if j8n else None,
                       "auroc_dist_median": round(float(np.median(au_n)), 3) if au_n else None,
                       "n_auroc_dist_lt_0.5": sum(1 for x in au_n if x < 0.5),
                       "frac_positives_within_10A": round(float(np.mean(np.array(dpos) < 10)), 4) if dpos else None,
                       "frac_positives_within_8A": round(float(np.mean(np.array(dpos) < 8)), 4) if dpos else None,
                       "dmin_positives": _q(dpos),
                       "dmin_positives_to_crystal_ligand": _q(dcr),
                       "frac_positives_within_10A_of_crystal_ligand": round(float(np.mean(np.array(dcr) < 10)), 4) if dcr else None},
        "contacts_5A": {"n_residues": n5, "n_in_pocket": n5in,
                        "frac_in_pocket": round(n5in / max(n5, 1), 4),
                        "jaccard_with_pocket_median": round(float(np.median(j5)), 3) if j5 else None},
        "geometry_flags": {
            "complexes_with_displaced_molecule": sum(1 for r in ok.values() if r.get("displaced_molecules")),
            "complexes_with_positives_in_copy_ambiguous_segment": sorted(
                p for p, r in ok.items() if r.get("n_pos_copy_ambiguous")),
            "positives_in_copy_ambiguous_segment": sum(r.get("n_pos_copy_ambiguous", 0)
                                                       for r in ok.values()),
            "complexes_with_positives_in_copy_no_contact_segment": sorted(
                p for p, r in ok.items() if r.get("n_pos_copy_no_contact")),
            "positives_in_copy_no_contact_segment": sum(r.get("n_pos_copy_no_contact", 0)
                                                        for r in ok.values()),
            "complexes_with_copy_reassigned": sorted(
                p for p, r in records.items() if (r.get("copies") or {}).get("reassigned")),
            "chains_reassigned": sum(len((r.get("copies") or {}).get("reassigned", []))
                                       for r in records.values()),
            "positives_local_dev_gt_3A": sum(r.get("n_pos_local_flag", 0) for r in ok.values()),
            "global_dev_median_q": _q([r["global_dev_median"] for r in ok.values()]),
            "ligand_pocket_frame_shift_gt_5A": sorted(p for p, r in ok.items()
                                                      if r.get("ligand", {}).get("pocket_frame_centroid_shift", 0) > 5)},
        "ligand_shift_q": _q(shift),
        "amino_acid_or_peptide_ligands": {
            "n": len(pep),
            "n_single_residue": sum(1 for r in pep.values() if len(r["ligand"]["residue_index"]) == 1),
            "n_two_or_more_residues": sum(1 for r in pep.values() if len(r["ligand"]["residue_index"]) >= 2),
            "n_with_residues_in_residue_list": sum(1 for r in pep.values() if r["ligand"]["residue_index"]),
            "n_ligand_residues_labelled_positive": sum(r["ligand"].get("residues_labelled_positive", 0) for r in pep.values()),
            "n_ligand_residues_aligned_to_pdbbind_protein": sum(r["ligand"].get("residues_aligned_to_pdbbind_protein", 0) for r in pep.values()),
            "pids": sorted(pep)},
        "chain_breaks_crosscheck": dict(xc),
        # [ours, the table's residue]: global deviation (A) of the MISATO residue after the
        # trimmed superposition, and how often ours is the closer one
        "chain_breaks_crosscheck_global_dev_A": {
            k: {"ours": _q([a for a, _ in v]),
                "table": _q([b for _, b in v if b is not None]),
                "n_ours_closer": sum(1 for a, b in v if b is not None and a < b),
                "n_table_closer": sum(1 for a, b in v if b is not None and b < a),
                "n_table_residue_absent": sum(1 for _, b in v if b is None)}
            for k, v in xdev.items()},
    }


# ---------------------------------------------------------------- main
def _atomic_savez(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdbbind-dir", type=Path, default=PDBBIND_DIR)
    ap.add_argument("--h5", type=Path, default=H5_PATH)
    ap.add_argument("--real-bb", type=Path, default=REAL_BB_DIR)
    ap.add_argument("--legacy", type=Path, default=MISATO_BINDING_SITE_LEGACY,
                    help="old label file: complex set and old-vs-new comparison")
    ap.add_argument("--chain-breaks", type=Path, default=CHAIN_BREAKS)
    ap.add_argument("--out", type=Path, default=MISATO_BINDING_SITE)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--pids", default=None, help="comma-separated subset (partial run)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.out
    report = args.report or Path(str(out).replace(".npz", ".report.json"))
    if not args.overwrite and (out.exists() or report.exists()):
        print(f"[refuse] {out} or {report} exists; pass --overwrite"); return 1
    if args.pids:
        pids = [p.strip().upper() for p in args.pids.split(",") if p.strip()]
    else:
        with np.load(args.legacy, allow_pickle=False) as z:
            pids = sorted(z.files)
    index_f = args.pdbbind_dir / "INDEX_refined_data.2020"
    index_names = read_index(index_f)
    t0 = time.time()
    labels, records, failed = {}, {}, {}
    jobs = [(p, str(args.real_bb), str(args.pdbbind_dir), str(args.h5),
             str(args.legacy) if args.legacy else None,
             str(args.chain_breaks) if args.chain_breaks else None,
             {p: index_names.get(p)} if p in index_names else {}) for p in pids]
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (pid, lab, rec, err) in enumerate(ex.map(_job, jobs, chunksize=8)):
            if err:
                failed[pid] = err; continue
            records[pid] = rec
            if lab is not None:
                labels[pid] = lab
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    if failed:
        print(f"[error] {len(failed)} complexes failed, nothing written: "
              f"{dict(list(failed.items())[:5])}"); return 1
    s = summary(records)
    # complex set of the INDEX file vs ours (ProteinShake meant to drop '-mer' ligands but its
    # test ran on the line with its newline, so it never did; MISATO lacks some entries)
    if not args.pids:
        import h5py
        with h5py.File(args.h5, "r") as h5:
            missing = sorted(set(index_names) - set(pids))
            s["pdbbind_refined_index"] = {
                "n": len(index_names), "n_in_label_set": len(set(index_names) & set(pids)),
                "n_not_in_set": len(missing),
                "n_not_in_set_but_in_md_h5": sum(1 for p in missing if p in h5),
                "n_mer_ligands_in_set": sum(1 for p in pids if index_names.get(p, "").endswith("-mer"))}
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(out, labels)
    _atomic_text(report, json.dumps({"schema": BINDING_LABELS_SCHEMA, "rule_version": RULE_VERSION,
                                     "rule": RULE, "complexes": records}, separators=(",", ":")))
    from ensembits.data_versions import write_provenance
    from ensembits.featurize import file_sha256
    pdb_files = sorted(f for p in pids for f in (args.pdbbind_dir / "refined-set" / p.lower()).glob("*")
                       if f.is_file())
    h = hashlib.sha256()
    for f in pdb_files:
        h.update(f.name.encode()); h.update(bytes.fromhex(file_sha256(f)))
    rb_h = hashlib.sha256()
    for p in pids:
        f = args.real_bb / f"{p}.npz"
        if f.exists():
            rb_h.update(p.encode()); rb_h.update(bytes.fromhex(file_sha256(f)))
    write_provenance(out, {
        "builder": "data_prep/build_misato_binding_site_labels.py", "schema": BINDING_LABELS_SCHEMA,
        "rule_version": RULE_VERSION, "rule": RULE, "align": ALIGN,
        # B.13 fix: added copy-choice constants to the provenance sidecar
        "thresholds": {"core_min_A": CORE_MIN_A, "displaced_median_A": DISPLACED_MEDIAN_A,
                       "mol_fit_rmsd_A": MOL_FIT_RMSD_A, "local_half_window": LOCAL_HALF_WINDOW,
                       "local_flag_A": LOCAL_FLAG_A, "min_pocket_aligned": MIN_POCKET_ALIGNED,
                       "copy_near_A": COPY_NEAR_A, "copy_min_score": COPY_MIN_SCORE,
                       "copy_margin": COPY_MARGIN, "copy_min_gain": COPY_MIN_GAIN},
        "output": str(out), "sha256": file_sha256(out), "n_labelled": len(labels),
        "report": str(report), "report_sha256": file_sha256(report),
        "inputs": {"pdbbind_dir": str(args.pdbbind_dir),
                   "pdbbind_source": "https://sandbox.zenodo.org/records/327402/files/"
                                     "PDBbind_v2020_refined.tar.gz (md5 e9412f4b5b065a4eab6958b3bb1c49a2)",
                   "pdbbind_index_sha256": file_sha256(index_f) if index_f.exists() else None,
                   "pdbbind_files_sha256": h.hexdigest(), "n_pdbbind_files": len(pdb_files),
                   "md_h5": str(args.h5), "misato_real_bb": str(args.real_bb),
                   "misato_real_bb_sha256": rb_h.hexdigest(),
                   "legacy_labels": str(args.legacy),
                   "legacy_labels_sha256": file_sha256(args.legacy) if args.legacy and args.legacy.exists() else None,
                   "chain_breaks": str(args.chain_breaks),
                   "chain_breaks_sha256": file_sha256(args.chain_breaks) if args.chain_breaks and args.chain_breaks.exists() else None},
        "notes": KNOWN_NOTES,
        "partial": bool(args.pids), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": s})
    def _short(v):     # lists of pids -> their count, for the console
        if isinstance(v, dict):
            return {k: _short(x) for k, x in v.items()}
        return f"{len(v)} pids" if isinstance(v, list) else v
    print(json.dumps(_short(s), indent=1))
    print(f"wrote {out} ({len(labels)} complexes) + .provenance.json, {report} "
          f"in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
