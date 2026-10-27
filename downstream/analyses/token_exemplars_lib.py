"""Geometry, loaders and PyMOL helpers of the token exemplar figures.

What it does
------------
Building blocks of `render_token_exemplars.py`, kept separate so they can be tested:

  * `load_system(row, settings)`: the real backbone the descriptor was built from
    (`bb_10`, N/CA/C/O of the descriptor's ten frames), sequence, chain index and the
    trajectory frame of every descriptor frame (MISATO: `chosen_indices_10`, indices into
    the 100 frames of MD.hdf5; BioEmu-CATH: run file and frame of each FPS pick, from
    `data/cached_descriptors/bioemu_cath_extras`).
  * `segments(bb0, chain_index)`: chain pieces (a new piece at a chain change or where
    |C_{i-1} - N_i| > 2.5 A, the descriptor's chain-break rule); used for the chain-end
    filter, the 3-mers and the PDB chain ids (no cartoon is drawn across a break).
  * `knn_edges_all_frames(bb)`: the descriptor's own neighbour lists
    (`ensembits.geometry.knn_edges`, K = 16 by C-alpha distance, the residue itself first,
    no sequence-gap filter) in every frame.
  * `canonical_knn(edges, ca, r, k)`: the "canonical" neighbours of residue r: the k
    residues that occur most often in r's per-frame lists over the P frames, ties broken
    by the mean C-alpha distance to r (r itself always first: it is slot 1 of every list).
  * `threemer_residues(nbrs, seg)`: union of (j - 1, j, j + 1) over the canonical
    neighbours, clipped to j's chain piece.
  * `kabsch` / `align_to_first(ca, align_idx)`: rigid transforms that superpose every
    frame on the first using the C-alpha of `align_idx` (all residues minus the 3-mers).
  * `anchor_frame(n, ca, c)`: the anchor's backbone frame (`ensembits.geometry`'s
    Gram-Schmidt frame of the descriptor), used to put every panel in the anchor's own
    coordinates so the panels share an orientation.
  * All-atom loaders: `load_misato_allatom(pid, frames)` (MISATO MD.hdf5, the residue
    parser of `data_prep/build_misato_real_bb.py`, so residue i is residue i of the
    descriptor) and `load_bioemu_allatom(system, run_frames)` (BioEmu-CATH topology.pdb +
    run xtc files via mdtraj, the residues kept by `build_bioemu_real_bb.py`).
    Both return every protein atom of the requested frames, a residue index per atom,
    atom names (MISATO stores no names: only N, CA, C, O, CB are named, the other atoms
    are kept for the rigid transform but not written), residue names and the C-alpha
    atom of every residue.
  * `write_multistate_pdb(...)`: one MODEL per frame, a chain id per chain piece, TER
    records between pieces.
  * `seq_identity(a, b)`: identical positions of a local BLOSUM62 alignment divided by the
    length of the shorter sequence (the "distinct protein" test of the exemplar walk).

Paper
-----
`appendix.tex` \\section{Token visualizations} (\\label{app:viz}): "the sixteen canonical
k-nearest neighbours (... the residues most frequently chosen as descriptor neighbours
across the P = 10 ensemble frames, matching the K = 16 kNN that the ESM3-style
descriptor uses at training time; no sequence-gap filter ...) ... the local 3-mer
(j-1, j, j+1) around each ... Kabsch-align all P frames using all C-alpha atoms of the
protein with the 3-mer residues excluded, applying the same rigid-body transform to the
all-atom coordinates from the source trajectory". K = 16 includes the residue itself
(`appendix.tex` \\label{app:descriptor-esm3}: "The list keeps r itself as its first
entry"), so the sixteen canonical neighbours are the anchor plus 15 other residues.

Origin
------
New file (2026-10-19, final_fix_list A11). Re-implements, with the paper's settings, the
helpers of the research scripts `experiments/shared_states_v2/render_token_knn_pymol.py`
(per_frame_knn / canonical_knn / alignment_atoms / write_multistate_pdb_allatom /
build_session_pml_allatom) and `experiments/shared_states_v2/_misato_allatom.py`. The
research defaults that contradicted the paper are not kept: k = 3 neighbours with a
sequence gap of 3 on 3Di virtual centres (here: the descriptor's K = 16 C-alpha kNN, no
gap), alignment on a 10 A anchor ball (here: all C-alpha minus the 3-mers), one chain id
with continuous numbering for all MISATO chains (here: a chain id per chain piece), and
PyMOL's dashed cartoon gaps (here: cartoon_gap_cutoff 0).

Used by the reported model: yes (figures fig:token1699, fig:token1063, fig:5x5tokens).

Inputs / outputs
----------------
reads (via the callers) data/cached_descriptors/{bioemu_cath_real_bb, bioemu_cath_extras,
misato_real_bb_kabsch}/<system>.npz, data/misato/MD.hdf5,
data/reviewer_3_special_corpus/bioemu_cath/extracted/<deposit>/<system>/.

How to run
----------
imported by `render_token_exemplars.py`; tests in tests/test_token_exemplars.py.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT  # noqa: E402

MISATO_H5 = Path(os.environ.get("ENSEMBITS_MISATO_H5", REPO_ROOT / "data" / "misato" / "MD.hdf5"))
BIOEMU_RAW_ROOT = Path(os.environ.get(
    "ENSEMBITS_BIOEMU_RAW", REPO_ROOT / "data" / "reviewer_3_special_corpus" / "bioemu_cath" / "extracted"))
BIOEMU_EXTRAS_DIR = REPO_ROOT / "data" / "cached_descriptors" / "bioemu_cath_extras"
CHAIN_BREAK_A = 2.5        # descriptor rule: |C_{i-1} - N_i| > 2.5 A is a break
K_DESCRIPTOR = 16          # descriptor kNN slots, the residue itself included
NAMED_ATOMS = ("N", "CA", "C", "O", "CB")
CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


# ----------------------------------------------------------------------------- systems

@dataclass
class System:
    corpus: str                 # "bioemu_cath" | "misato"
    system: str                 # layout system name (cath2_1a1wA00 / 9ICD)
    pid: str                    # CATH domain / PDB id
    bb: np.ndarray              # (10, L, 4, 3) N, CA, C, O of the descriptor frames
    seq: str
    chain_index: np.ndarray     # (L,)
    traj_frames: dict = field(default_factory=dict)   # descriptor frame -> trajectory frame

    @property
    def L(self) -> int:
        return int(self.bb.shape[1])


def bioemu_deposit(system: str) -> str:
    """cath1_<pid> -> ONE_cath1, cath2_<pid> -> MSR_cath2 (build_bioemu_real_bb layout)."""
    pre = system.split("_", 1)[0]
    return {"cath1": "ONE_cath1", "cath2": "MSR_cath2"}[pre]


def load_system(row: dict, settings: dict) -> System:
    """Real backbone, sequence, chains and trajectory frames of one corpus layout row."""
    if row["corpus"] == "misato":
        z = np.load(Path(settings["misato_bb"]) / f"{row['pid']}.npz", allow_pickle=False)
        frames = {"md_frame": np.asarray(z["chosen_indices_10"], np.int64)}
        return System("misato", row["system"], row["pid"], np.asarray(z["bb_10"], np.float32),
                      str(z["seq"]), np.asarray(z["chain_index"], np.int64), frames)
    if row["corpus"] == "bioemu_cath":
        z = np.load(Path(settings["bioemu_bb"]) / f"{row['system']}.npz", allow_pickle=False)
        bb = np.asarray(z["bb_10"], np.float32)
        ex = np.load(BIOEMU_EXTRAS_DIR / f"{row['system']}.npz", allow_pickle=False)
        frames = {k: np.asarray(ex[k]) for k in ("pick_run", "pick_frame", "pick_concat",
                                                 "run_files", "run_n_frames", "res_index")}
        if len(frames["res_index"]) != bb.shape[1]:
            raise RuntimeError(f"{row['system']}: extras res_index {len(frames['res_index'])} "
                               f"!= bb_10 L {bb.shape[1]}")
        return System("bioemu_cath", row["system"], row["pid"], bb, str(ex["seq"]),
                      np.zeros(bb.shape[1], np.int64), frames)
    raise ValueError(row["corpus"])


def segments(bb0: np.ndarray, chain_index: np.ndarray, cutoff: float = CHAIN_BREAK_A) -> np.ndarray:
    """(L,) chain-piece id: a new piece at a chain change or where |C_{i-1} - N_i| > cutoff."""
    L = bb0.shape[0]
    brk = np.zeros(L, bool)
    if L > 1:
        cn = np.linalg.norm(bb0[1:, 0] - bb0[:-1, 2], axis=-1)
        brk[1:] = (cn > cutoff) | (chain_index[1:] != chain_index[:-1])
    return np.cumsum(brk).astype(np.int64)


def dist_to_segment_end(seg: np.ndarray) -> np.ndarray:
    """(L,) number of residues between i and the nearer end of its chain piece."""
    L = len(seg)
    idx = np.arange(L)
    out = np.zeros(L, np.int64)
    for s in np.unique(seg):
        m = np.where(seg == s)[0]
        out[m] = np.minimum(idx[m] - m[0], m[-1] - idx[m])
    return out


# ----------------------------------------------------------------------------- kNN

def knn_edges_all_frames(bb: np.ndarray, k: int = K_DESCRIPTOR) -> np.ndarray:
    """(P, L, k) descriptor neighbour lists of every frame (ensembits.geometry.knn_edges)."""
    import torch
    from ensembits import geometry as G
    coords = torch.as_tensor(np.ascontiguousarray(bb[:, :, :3, :]), dtype=torch.float32)
    _, _, mask = G.backbone_frames(coords)
    return G.knn_edges(coords, mask, k).numpy()


def canonical_knn(edges: np.ndarray, ca: np.ndarray, r: int, k: int = K_DESCRIPTOR):
    """Canonical neighbours of residue r over the P frames.

    edges (P, L, K) per-frame neighbour lists; ca (P, L, 3). Returns (nbrs, counts,
    mean_dist): the k residues most often in r's lists, ties by mean C-alpha distance,
    then index. nbrs[0] == r (r is slot 1 of every list).
    """
    flat = edges[:, r, :].reshape(-1)
    uniq, cnt = np.unique(flat, return_counts=True)
    md = np.linalg.norm(ca[:, uniq, :] - ca[:, r:r + 1, :], axis=-1).mean(0)
    order = np.lexsort((uniq, md, -cnt))[:k]
    return uniq[order].astype(np.int64), cnt[order].astype(np.int64), md[order].astype(np.float64)


def threemer_residues(nbrs, seg: np.ndarray) -> list[int]:
    """Union of (j-1, j, j+1) over nbrs, restricted to j's chain piece."""
    L = len(seg)
    out = set()
    for j in nbrs:
        j = int(j)
        for i in (j - 1, j, j + 1):
            if 0 <= i < L and seg[i] == seg[j]:
                out.add(i)
    return sorted(out)


# ----------------------------------------------------------------------------- alignment

def kabsch(mobile: np.ndarray, target: np.ndarray):
    """(R, m_c, t_c) with (mobile - m_c) @ R.T + t_c ~= target (proper rotation)."""
    mobile = np.asarray(mobile, np.float64); target = np.asarray(target, np.float64)
    m_c = mobile.mean(0); t_c = target.mean(0)
    H = (mobile - m_c).T @ (target - t_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, m_c, t_c


def align_to_first(ca: np.ndarray, align_idx) -> list:
    """Per frame (R, m_c, t_c) superposing its align_idx C-alpha on those of frame 0."""
    align_idx = np.asarray(align_idx, np.int64)
    if len(align_idx) < 3:
        raise ValueError("alignment needs at least 3 C-alpha")
    ref = ca[0, align_idx]
    out = [(np.eye(3), np.zeros(3), np.zeros(3))]
    for p in range(1, ca.shape[0]):
        out.append(kabsch(ca[p, align_idx], ref))
    return out


def apply_transform(x: np.ndarray, tr) -> np.ndarray:
    R, m_c, t_c = tr
    return (np.asarray(x, np.float64) - m_c) @ R.T + t_c


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum(-1).mean()))


def anchor_frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray):
    """(rot (3, 3) with the frame axes as columns, origin) of the descriptor's backbone frame
    (ensembits.geometry: origin CA, x along CA - C, xy plane through N). Local coordinates
    of a point x are (x - origin) @ rot."""
    import torch
    from ensembits import geometry as G
    t = lambda v: torch.as_tensor(np.asarray(v, np.float64)[None], dtype=torch.float64)  # noqa: E731
    rot, trans = G._frame_from_atoms(t(n), t(ca), t(c))
    return rot[0].double().numpy(), trans[0].double().numpy()


# ----------------------------------------------------------------------------- sequences

_ALIGNER = None


def seq_identity(a: str, b: str) -> float:
    """Identical positions of the best local BLOSUM62 alignment (gap open -10, extend -0.5)
    divided by the length of the shorter sequence."""
    global _ALIGNER
    if not a or not b:
        return 0.0
    if _ALIGNER is None:
        from Bio import Align
        from Bio.Align import substitution_matrices
        al = Align.PairwiseAligner()
        al.mode = "local"
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
        al.open_gap_score = -10.0
        al.extend_gap_score = -0.5
        _ALIGNER = al
    clean = lambda s: "".join(ch if ch in ONE_TO_THREE else "X" for ch in s)  # noqa: E731
    aln = _ALIGNER.align(clean(a), clean(b))[0]
    ident = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        sa = clean(a)[s1:e1]; sb = clean(b)[s2:e2]
        ident += sum(x == y for x, y in zip(sa, sb))
    return ident / min(len(a), len(b))


def chain_sequence(sysm: System, r: int) -> str:
    """Sequence of the chain (chain_index) that holds residue r."""
    m = sysm.chain_index == sysm.chain_index[r]
    return "".join(np.array(list(sysm.seq))[m])


# ----------------------------------------------------------------------------- all-atom

@dataclass
class AllAtom:
    coords: np.ndarray          # (F, A, 3) float64, Angstrom
    atom_res: np.ndarray        # (A,) residue index 0..L-1
    atom_name: list             # (A,) PDB atom name, "" = unnamed (not written)
    resname: list               # (L,) three-letter residue names
    ca_atom: np.ndarray         # (L,) atom index of every residue's C-alpha
    frames: list                # trajectory frame ids loaded (MISATO: MD frame; BioEmu: (run file, frame))

    def ca(self) -> np.ndarray:
        return self.coords[:, self.ca_atom, :]


def _misato_parse(g):
    """parse_complex of data_prep/build_misato_real_bb.py plus per-residue atom ranges."""
    sys.path.insert(0, str(CODE_ROOT / "data_prep"))
    import build_misato_real_bb as BMR
    parsed = BMR.parse_complex(g)
    if parsed is None:
        raise RuntimeError("no protein residues")
    an = g["atoms_number"][:]
    at = g["atoms_type"][:]
    mb = list(g["molecules_begin_atom_index"][:])
    ends = mb[1:] + [len(an)]
    # protein molecule of every residue: the molecule whose atom range holds its N
    mol_of = np.searchsorted(np.asarray(mb), parsed["n"], side="right") - 1
    mol_end = np.asarray(ends)[mol_of]
    L = len(parsed["n"])
    # residue i spans [N_i, N_{i+1}) inside its molecule, else [N_i, molecule end)
    nxt = np.empty(L, np.int64)
    nxt[:-1] = np.where(mol_of[1:] == mol_of[:-1], parsed["n"][1:], mol_end[:-1])
    nxt[-1] = mol_end[-1]
    return parsed, parsed["n"], nxt, an, at


def load_misato_allatom(pid: str, frames, h5_path: Path = MISATO_H5) -> AllAtom:
    """Every protein atom of the complex `pid` in MD frames `frames` (indices into the 100
    stored frames), residues as parsed by build_misato_real_bb (= the descriptor's)."""
    import h5py
    frames = [int(f) for f in frames]
    with h5py.File(h5_path, "r", swmr=True) as h:
        if pid not in h:
            raise KeyError(f"{pid} not in {h5_path}")
        g = h[pid]
        parsed, start, stop, an, at = _misato_parse(g)
        order = np.argsort(frames)
        srt = [frames[i] for i in order]
        uniq, inv = np.unique(srt, return_inverse=True)
        xyz_u = g["trajectory_coordinates"][list(uniq)]           # h5py needs increasing indices
        xyz_s = xyz_u[inv]
        xyz = np.empty_like(xyz_s)
        xyz[order] = xyz_s
    L = len(start)
    atom_idx, atom_res, names = [], [], []
    named = {}
    for key, nm in (("n", "N"), ("ca", "CA"), ("c", "C"), ("o", "O")):
        for i, a in enumerate(parsed[key]):
            named[int(a)] = nm
    for i in np.where(parsed["cb_present"])[0]:
        named[int(parsed["cb"][i])] = "CB"
    ca_atom = np.zeros(L, np.int64)
    for i in range(L):
        for a in range(int(start[i]), int(stop[i])):
            if an[a] == 1:                       # hydrogens are not needed
                continue
            if a == parsed["ca"][i]:
                ca_atom[i] = len(atom_idx)
            atom_idx.append(a); atom_res.append(i); names.append(named.get(a, ""))
    resname = [ONE_TO_THREE.get(ch, "UNK") for ch in parsed["seq"]]
    return AllAtom(np.asarray(xyz[:, atom_idx, :], np.float64), np.asarray(atom_res, np.int64),
                   names, resname, ca_atom, frames)


def load_bioemu_allatom(system: str, run_frames, res_index, raw_root: Path = BIOEMU_RAW_ROOT) -> AllAtom:
    """Heavy atoms of the BioEmu-CATH residues `res_index` (topology residue indices, the
    residues kept by build_bioemu_real_bb) in the frames run_frames = [(run file, frame)].
    Coordinates nm -> Angstrom as in the corpus builder."""
    import mdtraj as md
    sysdir = Path(raw_root) / bioemu_deposit(system) / system
    top = md.load_topology(str(sysdir / "topology.pdb"))
    res_index = [int(r) for r in res_index]
    pos = {r: i for i, r in enumerate(res_index)}
    atoms = [a for a in top.atoms if a.residue.index in pos
             and (a.element is None or a.element.symbol != "H")]
    sel = np.array([a.index for a in atoms], np.int64)
    xyz = []
    for run_file, fr in run_frames:
        t = md.load_frame(str(sysdir / "trajs" / str(run_file)), int(fr), top=top)
        xyz.append(t.xyz[0][sel] * 10.0)
    atom_res = np.array([pos[a.residue.index] for a in atoms], np.int64)
    names = [a.name if a.name in NAMED_ATOMS else "" for a in atoms]
    ca_atom = np.full(len(res_index), -1, np.int64)
    for k, a in enumerate(atoms):
        if a.name == "CA":
            ca_atom[atom_res[k]] = k
    if (ca_atom < 0).any():
        raise RuntimeError(f"{system}: residue without CA")
    sys.path.insert(0, str(CODE_ROOT / "downstream"))
    from rmsf_common import three_to_one
    residues = list(top.residues)
    resname = [ONE_TO_THREE.get(three_to_one(residues[r].name), "UNK") for r in res_index]
    return AllAtom(np.asarray(np.stack(xyz), np.float64), atom_res, names, resname, ca_atom,
                   [(str(a), int(b)) for a, b in run_frames])


def descriptor_frame_positions(n_frames: int, P: int = 10) -> list[int]:
    """Positions of n_frames frames evenly spaced over the P descriptor frames (storage order)."""
    return sorted({int(round(x)) for x in np.linspace(0, P - 1, n_frames)})


def trajectory_frames(sysm: System, mode: str, n_frames: int):
    """Frames to load for the render, and the descriptor positions they correspond to.

    mode "descriptor": n_frames of the descriptor's ten frames (positions evenly spaced over
      the stored order: MISATO time order, BioEmu FPS order); position 0 is the first
      descriptor frame (the alignment reference).
    mode "even": n_frames evenly spaced over the whole trajectory (MISATO: the 100 stored
      frames; BioEmu: all runs concatenated); positions are None.
    Returns (frames, positions): MISATO frames are MD frame indices, BioEmu frames are
    (run file, frame within run).
    """
    if mode == "descriptor":
        pos = descriptor_frame_positions(n_frames)
        if sysm.corpus == "misato":
            return [int(sysm.traj_frames["md_frame"][p]) for p in pos], pos
        tf = sysm.traj_frames
        return [(str(tf["run_files"][int(tf["pick_run"][p])]), int(tf["pick_frame"][p])) for p in pos], pos
    if mode == "even":
        if sysm.corpus == "misato":
            return [int(x) for x in np.linspace(0, 99, n_frames).astype(int)], None
        tf = sysm.traj_frames
        run_n = np.asarray(tf["run_n_frames"], np.int64)
        T = int(run_n.sum())
        conc = np.linspace(0, T - 1, n_frames).astype(int)
        ends = np.cumsum(run_n); starts = np.concatenate([[0], ends[:-1]])
        run = np.searchsorted(ends, conc, side="right")
        return [(str(tf["run_files"][r]), int(c - starts[r])) for r, c in zip(run, conc)], None
    raise ValueError(mode)


def load_allatom(sysm: System, frames) -> AllAtom:
    if sysm.corpus == "misato":
        return load_misato_allatom(sysm.pid, frames)
    return load_bioemu_allatom(sysm.system, frames, sysm.traj_frames["res_index"])


def allatom_available(corpus: str, system: str, pid: str, misato_keys: set | None) -> bool:
    if corpus == "misato":
        return misato_keys is None or pid in misato_keys
    d = BIOEMU_RAW_ROOT / bioemu_deposit(system) / system
    return (d / "topology.pdb").exists() and (BIOEMU_EXTRAS_DIR / f"{system}.npz").exists()


# ----------------------------------------------------------------------------- PDB

def write_multistate_pdb(path: Path, coords: np.ndarray, atom_res: np.ndarray, atom_name,
                         resname, seg: np.ndarray, keep_res=None):
    """Multi-model PDB of the named atoms of residues keep_res (all if None).

    coords (F, A, 3). Chain id = chain piece (cycled over 62 letters), residue number =
    residue index + 1, a TER after every piece. Returns {residue index: (chain, resSeq)}.
    """
    keep = set(range(len(resname))) if keep_res is None else set(int(r) for r in keep_res)
    rows = [a for a in range(len(atom_res)) if atom_name[a] and int(atom_res[a]) in keep]
    rows.sort(key=lambda a: (int(atom_res[a]), NAMED_ATOMS.index(atom_name[a])))
    ids = {}
    out = []
    for m in range(coords.shape[0]):
        out.append(f"MODEL     {m + 1:4d}")
        serial = 1
        prev_seg = None
        for a in rows:
            i = int(atom_res[a])
            s = int(seg[i])
            if prev_seg is not None and s != prev_seg:
                out.append("TER")
            prev_seg = s
            ch = CHAIN_LETTERS[s % len(CHAIN_LETTERS)]
            resseq = (i + 1) % 10000
            ids[i] = (ch, resseq)
            nm = atom_name[a]
            name4 = nm if len(nm) == 4 else (" " + nm).ljust(4)
            x, y, z = coords[m, a]
            # columns: 1-6 record, 7-11 serial, 13-16 name, 18-20 resname, 22 chain,
            # 23-26 resSeq, 31-54 x y z, 55-66 occupancy / B, 77-78 element
            out.append(f"ATOM  {serial % 100000:5d} {name4} {resname[i]:>3s} {ch}{resseq:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {nm[0]:>2s}")
            serial += 1
        out.append("TER")
        out.append("ENDMDL")
    out.append("END")
    Path(path).write_text("\n".join(out) + "\n")
    return ids
