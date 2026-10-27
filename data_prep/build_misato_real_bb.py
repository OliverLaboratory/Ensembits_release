"""Find the protein backbone atoms in MISATO's MD.hdf5 and cache FPS-selected frames.

What it does
------------
MISATO stores 16,972 protein-ligand complexes, 100 saved MD frames each, in one
HDF5 file with atomic numbers and AMBER atom types but no atom names. This
script

  1. walks the molecules of each complex, keeps the protein chains (first atom
     is a backbone N with a protein AMBER type), splits them into residues at
     the peptide-bond pattern and finds every residue's N, CA, C, O and CB
     (`find_residues`). CA is the atom with AMBER type CX (MISATO's own atom
     type map); CB is the carbon bonded to CA other than C;
  2. concatenates all protein chains of the complex into one residue list and
     records the chain index of every residue;
  3. picks K frames out of the 100 by greedy farthest point sampling seeded at
     frame 0 on Cα RMSD after optimal superposition (`--fps kabsch`, default; the
     rule for both caches since 2026-10-15). `--fps frobenius` (raw Cα
     coordinates, no superposition) reproduces the caches of the current
     checkpoint and its downstream results;
  4. writes the backbone of those frames and the sequence (from MISATO's
     residue names).

Two caches are built with it (the output directory follows from --K since
2026-10-16, code_fix P14; --out-dir overrides it):
  * downstream inputs (8 frames; code_fix C10):
        --K 8   -> data/cached_descriptors/misato_real_bb
  * tokenizer training corpus (10 frames; code_fix T1.6):
        --K 10  -> data/cached_descriptors/misato_real_bb_kabsch
  (any other K, or --fps frobenius with K = 10: misato_real_bb_K<K>_<fps>)

Paper
-----
  * `appendix.tex`, \\subsection{Dataset} (\\label{app:dataset}),
    \\paragraph{MISATO.}: "select 10 of the 100 saved frames by FPS on the
    Cα coordinates, starting from the first frame". The current checkpoint used
    raw-coordinate FPS (`fps_select`, K10 script); the retrain uses Kabsch FPS
    (`--fps kabsch`), so the text must add "after superposition" (audit T10).
  * `2_results.tex`, \\subsection{Datasets}: MISATO, 16,972 complexes.

Origin
------
Copied from `submission_exp/src/build_misato_real_bb.py`; the Kabsch option
replaces `submission_exp/src/rebuild_misato_real_bb_kabsch.py` (same output
layout). Changed on 2026-10-15 (code_fix §4 / T1.1, T1.6): CA by atom type,
CB by bond to CA, sequence from residue names, `--fps`, `--out-dir`,
`chain_index` and `parser_version` fields. Earlier caches were written by the
old parser (proline CA wrong) and are refused as output directory.
Changed on 2026-10-16 (code_fix P14): default output directory from K; `K`
and `fps` stored in every file; every existing file in the output directory
must match parser_version, K and fps (it is refused otherwise).

Used by the reported model: the current checkpoint was trained on the old
parser's output (`build_misato_real_bb_K10.py`, bb_10). The retrained model
uses the `--K 10 --fps kabsch` cache written here.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/misato/MD.hdf5
  writes  <out-dir>/<pid>.npz (fields below)

How to run
----------
    python data_prep/build_misato_real_bb.py --K 8  [--workers 16]   # -> misato_real_bb
    python data_prep/build_misato_real_bb.py --K 10 [--workers 16]   # -> misato_real_bb_kabsch
    (--limit N, --pids-file pids.txt restrict the complexes.) Move caches built
    by the old parser aside first; the script refuses to mix them.

Original description
--------------------
Mirrors `build_mdcath_real_bb.py` but with misato's MD.hdf5 layout.
Each PDB group in `data/misato/MD.hdf5` has:
    atoms_number          (n_atoms,)        — atomic number per atom
    atoms_type            (n_atoms,)        — AMBER ff14SB atom-type integer
    atoms_residue         (n_atoms,)        — residue-type integer (NOT residue index)
    molecules_begin_atom_index (n_mol+1,)   — start atom of each molecule (chains + ligands)
    trajectory_coordinates (100, n_atoms, 3) float64 — MD frames

Residue boundaries are detected via the peptide-bond pattern
    C(atom_type=3) → O(an=8) → [optional OXT(an=8)] → N(an=7),
where the post-N heavy atom must be a non-sp2 C (else ASN ND2 / GLN NE2
trigger false starts). Within each residue:
    N  : the boundary N
    CA : the atom with AMBER type CX (type code 14)    [was: first C after N]
    C  : last an=6 with atom_type=3 in the residue
    O  : first an=8 after C (skipping H)
    CB : the carbon nearest to CA in frame 0, other than C; None ⇒ glycine
                                                       [was: first C between CA and C]

Output (one file per pid; K = 8 or 10):
    <out-dir>/<pid>.npz
        bb_K           : (K, L, 4, 3) float32  — [N, CA, C, O]
        cb_K           : (K, L, 3)   float32   — Cβ (NaN for GLY)
        ca_K           : (K, L, 3)   float32   — = bb_K[:, :, 1]
        seq            :  one-letter sequence (L,), from MISATO residue names
        res_type_int   : (L,) int32  — K = 8 layout: raw atoms_residue code;
                                       Kabsch layout: index in ACDEFGHIKLMNPQRSTVWY
                                       (20 = unknown), as the files it replaces
        aa_index       : (L,) int32  — index in ACDEFGHIKLMNPQRSTVWY (both layouts)
        chain_index    : (L,) int32  — protein chain of each residue (0, 1, ...)
        chosen_indices : (8,) int64  — K = 8 layout (frame indices in [0, 100))
        chosen_indices_10 : (10,) int64 — K = 10 layout
        parser_version : "cx_2026-10-15"
        K              : () int64   — frames requested (--K)          [since 2026-10-16]
        fps            : "kabsch" or "frobenius" (--fps)                [since 2026-10-16]

Resumable: skips PDBs whose output already exists, after checking that every
existing file carries this parser_version, K and fps (code_fix P14).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo

from ensembits.kabsch_fps import kabsch_fps_select   # noqa: E402

H5_PATH = ROOT / 'data' / 'misato' / 'MD.hdf5'
# Default output: the downstream 8-frame cache. The training corpus cache is
# data/cached_descriptors/misato_real_bb_kabsch (--K 10 --fps kabsch).
OUT_DIR = ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'


# FIXED(code_fix P14, 2026-10-16): the default output directory follows from K (and
#   fps). Old: one default (misato_real_bb) for every K, so a K = 10 run without
#   --out-dir resumed into the K = 8 cache by file name. New: K = 8 -> misato_real_bb,
#   K = 10 -> misato_real_bb_kabsch (Kabsch FPS), anything else a name of its own.
def default_out_dir(K: int, fps: str = 'kabsch') -> Path:
    """Default cache directory for (K, fps)."""
    base = ROOT / 'data' / 'cached_descriptors'
    if K == 8 and fps == 'kabsch':
        return base / 'misato_real_bb'
    if K == 10 and fps == 'kabsch':
        return base / 'misato_real_bb_kabsch'
    return base / f'misato_real_bb_K{K}_{fps}'

# Version tag written into every file by this parser (code_fix §4 fix); defined in
# ensembits/data_versions.py so every reader checks the same value.
from ensembits.data_versions import MISATO_PARSER_VERSION as PARSER_VERSION  # noqa: E402

# MISATO stores no atom names, only integer AMBER atom types and residue codes.
# MISATO's own maps (github.com/t7morgen/misato-dataset,
# src/data/processing/Maps/atoms_type_map.pickle and atoms_residue_map.pickle)
# decode them. Type 14 is "CX", the ff14SB type of the protein Cα (every amino
# acid's CA has type CX in atoms_name_map_for_pdb.pickle).
CX_TYPE = 14
MISATO_RESIDUE_NAMES = {
    0: 'MOL', 1: 'ACE', 2: 'ALA', 3: 'ARG', 4: 'ASN', 5: 'ASP', 6: 'CYS', 7: 'CYX',
    8: 'GLN', 9: 'GLU', 10: 'GLY', 11: 'HIE', 12: 'ILE', 13: 'LEU', 14: 'LYS',
    15: 'MET', 16: 'PHE', 17: 'PRO', 18: 'SER', 19: 'THR', 20: 'TRP', 21: 'TYR',
    22: 'VAL',
}
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'CYX': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIE': 'H', 'ILE': 'I', 'LEU': 'L',
    'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T',
    'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
AA20 = 'ACDEFGHIKLMNPQRSTVWY'
CB_BOND_MAX = 2.0   # Å; guard only (bonded CA–CB 1.38–1.69 Å, non-bonded >= ~2.4 Å)

# Side-chain heavy-atom signature (n_C, n_N, n_O, n_S) → 1-letter AA.
# Ambiguous signatures are resolved at runtime (PRO/VAL via N–CD bond,
# LEU/ILE via H count on CB).
HEAVY_SIG_TO_AA = {
    (0, 0, 0, 0): 'G',
    (1, 0, 0, 0): 'A',
    (1, 0, 1, 0): 'S',
    (1, 0, 0, 1): 'C',
    (2, 0, 1, 0): 'T',
    (3, 0, 0, 1): 'M',
    (2, 1, 1, 0): 'N',
    (2, 0, 2, 0): 'D',
    (4, 1, 0, 0): 'K',
    (3, 0, 2, 0): 'E',
    (3, 1, 1, 0): 'Q',
    (4, 2, 0, 0): 'H',
    (7, 0, 0, 0): 'F',
    (4, 3, 0, 0): 'R',
    (7, 0, 1, 0): 'Y',
    (9, 1, 0, 0): 'W',
}


# AMBER ff14SB backbone-N atom types observed in misato proteins.
# Anything else as the first atom of a molecule means it's a ligand/cofactor.
PROTEIN_N_TYPES = {24, 25, 26, 27, 28}


def find_residues(an, at, s, e, xyz0):
    """Return list of dicts {N, CA, C, O, CB} for residues in atoms[s:e),
    or [] / None if the segment is not a well-formed protein chain.

    xyz0: (n_atoms, 3) coordinates of one frame, used only to find CB (the
    carbon bonded to CA).
    """
    # Pass 1: find residue start atoms. A new residue starts at an N that
    # follows a carbonyl C (AMBER type 3) and its O (plus an optional OXT),
    # and is itself followed by a non-carbonyl C (its CA).
    starts = [s] if an[s] == 7 else []
    i = s
    while i < e - 1:
        if an[i] == 6 and at[i] == 3:
            j = i + 1
            while j < e and an[j] == 1:
                j += 1
            if j >= e or an[j] != 8:
                i += 1; continue
            k = j + 1
            while k < e and an[k] in (1, 8):
                k += 1
            if k >= e or an[k] != 7:
                i += 1; continue
            m = k + 1
            while m < e and an[m] == 1:
                m += 1
            if m < e and an[m] == 6 and at[m] != 3:
                starts.append(k); i = k; continue
        i += 1
    starts.append(e)

    # Pass 2: inside each residue pick N (start), CA (atom type CX), C (last
    # carbonyl carbon), O (first oxygen after C), CB (carbon bonded to CA; None
    # for glycine). Any malformed residue rejects the whole chain.
    out = []
    for r in range(len(starts) - 1):
        rs, re_ = starts[r], starts[r + 1]
        ni = rs
        # FIXED(code_fix §4 / T1.1, 2026-10-15): CA is the atom of AMBER type CX.
        #   The old rule "first carbon after N" picked Cδ for every proline (AMBER
        #   order N, CD, CG, CB, CA, C, O): 4.7–5% of MISATO residues had Cδ in the
        #   CA slot ("CA"–C 3.5–3.9 Å) and Cγ in the CB slot. For the other 19
        #   amino acids the old and new choice are the same atom.
        cx = [k for k in range(rs, re_) if at[k] == CX_TYPE]
        if len(cx) != 1:
            return None
        ca = cx[0]
        c_cands = [k for k in range(ca + 1, re_) if an[k] == 6 and at[k] == 3]
        if not c_cands:
            return None
        ci = c_cands[-1]
        oi = ci + 1
        while oi < re_ and an[oi] == 1:
            oi += 1
        if oi >= re_ or an[oi] != 8:
            return None
        # CB: the carbon nearest to CA other than the carbonyl C (glycine has no
        # other carbon). Bonded C–C is 1.38–1.69 Å in MISATO frames; the next
        # carbons (1–3 neighbours such as CG, or proline's CD) are >= ~2.4 Å away.
        cb_cands = [k for k in range(rs, re_) if an[k] == 6 and k not in (ca, ci)]
        cb = None
        if cb_cands:
            dist = [np.linalg.norm(xyz0[k] - xyz0[ca]) for k in cb_cands]
            j = int(np.argmin(dist))
            if dist[j] > CB_BOND_MAX:
                return None
            cb = cb_cands[j]
        out.append({'N': ni, 'CA': ca, 'C': ci, 'O': oi, 'CB': cb})
    return out


def aa_from_resname(ar, res) -> str:
    """One-letter code from MISATO's residue code at the CA atom ('X' if unknown)."""
    return THREE_TO_ONE.get(MISATO_RESIDUE_NAMES.get(int(ar[res['CA']]), ''), 'X')


def aa_from_residue(an, frame0, res):
    """Derive 1-letter AA code from heavy-atom side-chain signature.

    Old parser only (assumes its CA = first carbon after N). Replaced by
    `aa_from_resname`; kept so the old sequences can be reproduced.
    """
    ca, ci, ni = res['CA'], res['C'], res['N']
    if res['CB'] is None:
        return 'G'
    sc = [k for k in range(ca + 1, ci) if an[k] != 1]
    n_C = sum(1 for k in sc if an[k] == 6)
    n_N = sum(1 for k in sc if an[k] == 7)
    n_O = sum(1 for k in sc if an[k] == 8)
    n_S = sum(1 for k in sc if an[k] == 16)
    sig = (n_C, n_N, n_O, n_S)
    code = HEAVY_SIG_TO_AA.get(sig)
    if code is not None:
        return code
    if sig == (3, 0, 0, 0):
        # PRO vs VAL — PRO's CD is bonded to N (~1.45 Å); VAL has no such bond.
        last_sc = sc[-1] if sc else None
        if last_sc is not None:
            d = float(np.linalg.norm(frame0[last_sc] - frame0[ni]))
            if d < 2.0:
                return 'P'
        return 'V'
    if sig == (4, 0, 0, 0):
        # LEU vs ILE — LEU's CB has 2 H neighbors, ILE's CB has 1.
        cb = res['CB']
        h_count = 0
        k = cb + 1
        while k < ci and an[k] == 1:
            h_count += 1; k += 1
        if h_count >= 2:
            return 'L'
        if h_count == 1:
            return 'I'
        return 'X'
    return 'X'


def fps_select(ca_traj: np.ndarray, K: int) -> list[int]:
    """Greedy farthest-point sampling in CA Frobenius distance.
    Always seeds from frame 0 for determinism."""
    # Paper: appendix.tex \paragraph{MISATO.} — "select 10 of the 100 saved
    # frames by FPS on the Cα coordinates, starting from the first frame".
    # Distance = Frobenius norm of the raw CA coordinate difference, with NO
    # superposition (unlike the Kabsch-aligned RMSD used for BioEmu-CATH).
    # Stops early if a frame would repeat; returns indices in time order.
    # Used only with --fps frobenius (old caches); the default is Kabsch-aligned
    # FPS for both caches (code_fix T1.6, C10).
    N = ca_traj.shape[0]
    selected = [0]
    dists = np.linalg.norm(ca_traj - ca_traj[0:1], axis=(1, 2))
    for _ in range(K - 1):
        nxt = int(np.argmax(dists))
        if nxt in selected:
            break
        selected.append(nxt)
        new_d = np.linalg.norm(ca_traj - ca_traj[nxt:nxt + 1], axis=(1, 2))
        dists = np.minimum(dists, new_d)
    return sorted(selected)


def parse_complex(g) -> dict | None:
    """Parse one MD.hdf5 group into per-residue atom indices.

    Returns None if the complex has no protein residue, else a dict with the
    index arrays n/ca/c/o/cb (cb = 0 where absent), cb_present, chain_index,
    the sequence and the raw residue codes, all of length L.
    """
    an = g['atoms_number'][:]
    at = g['atoms_type'][:]
    ar = g['atoms_residue'][:]
    mb = list(g['molecules_begin_atom_index'][:])
    chain_ends = mb[1:] + [len(an)]
    xyz0 = g['trajectory_coordinates'][0]                      # frame 0, for CB bonds

    residues_per_chain: list[list[dict]] = []
    for mi in range(len(mb)):
        s = mb[mi]; e = chain_ends[mi]
        # Keep protein chains only: they start with a backbone N (amide N or
        # N-terminal NH3+); ligand N atoms have other AMBER types.
        if an[s] != 7 or int(at[s]) not in PROTEIN_N_TYPES:
            continue
        res = find_residues(an, at, s, e, xyz0)
        if not res:
            # Empty or malformed; skipped (e.g. a short peptide cofactor).
            continue
        residues_per_chain.append(res)
    # All protein chains are concatenated into one residue list; chain_index
    # records the chain. The descriptor treats φ/ψ across a chain junction as
    # undefined (C–N distance > 2.5 Å; code_fix §6).
    all_res = [r for chain in residues_per_chain for r in chain]
    if not all_res:
        return None
    chain_index = np.concatenate([np.full(len(c), i, np.int32)
                                  for i, c in enumerate(residues_per_chain)])
    seq = ''.join(aa_from_resname(ar, r) for r in all_res)
    return {
        'n': np.array([r['N'] for r in all_res], dtype=np.int64),
        'ca': np.array([r['CA'] for r in all_res], dtype=np.int64),
        'c': np.array([r['C'] for r in all_res], dtype=np.int64),
        'o': np.array([r['O'] for r in all_res], dtype=np.int64),
        'cb_present': np.array([r['CB'] is not None for r in all_res]),
        'cb': np.array([r['CB'] if r['CB'] is not None else 0 for r in all_res],
                       dtype=np.int64),
        'chain_index': chain_index,
        'seq': seq,
        'res_code': ar[np.array([r['N'] for r in all_res], dtype=np.int64)].astype(np.int32),
    }


def existing_file_mismatch(path: Path, K: int, fps: str) -> str | None:
    """Why an existing <pid>.npz cannot be kept for this run (None = it matches).

    Checks parser_version, K and fps (code_fix P14). Files written before 2026-10-16
    carry no K / fps and are refused.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            got = {k: (z[k].item() if k in z.files else None)
                   for k in ('parser_version', 'K', 'fps')}
    except Exception as exc:                       # unreadable / truncated file
        return f'unreadable ({type(exc).__name__})'
    if got['parser_version'] is None or str(got['parser_version']) != PARSER_VERSION:
        return f'parser_version {got["parser_version"]!r} != {PARSER_VERSION!r}'
    if got['K'] is None or int(got['K']) != int(K):
        return f'K {got["K"]!r} != {K}'
    if got['fps'] is None or str(got['fps']) != fps:
        return f'fps {got["fps"]!r} != {fps!r}'
    return None


def _process_pid(pid: str, K: int = 8, fps: str = 'kabsch',
                 out_dir: Path = OUT_DIR) -> tuple[str, str | None]:
    """Parse one complex, FPS-select K frames and write <pid>.npz; returns (pid, status)."""
    out_path = Path(out_dir) / f'{pid}.npz'
    if out_path.exists():
        # FIXED(code_fix P14, 2026-10-16): old: any existing file was skipped by name.
        #   New: it is kept only if parser_version, K and fps match this run; otherwise
        #   the status 'mismatch: ...' makes main() exit 1.
        why = existing_file_mismatch(out_path, K, fps)
        return pid, ('skip' if why is None else f'mismatch: {why}')
    try:
        with h5py.File(H5_PATH, 'r', swmr=True) as f:
            if pid not in f:
                return pid, 'no_pid'
            g = f[pid]
            parsed = parse_complex(g)
            if parsed is None:
                return pid, 'no_residues'
            L = len(parsed['ca'])

            # Trajectory: all 100 frames, FPS on CA, then gather the atoms.
            coords = g['trajectory_coordinates'][:]            # (100, n_atoms, 3) float64
            ca_traj = coords[:, parsed['ca'], :].astype(np.float32)   # (100, L, 3)
            if fps == 'kabsch':
                # Paper: appendix.tex app:dataset — FPS on Cα RMSD (code_fix T1.6).
                chosen = kabsch_fps_select(ca_traj, K=K, seed_idx=0)
            else:
                chosen = fps_select(ca_traj, K)
            chosen_arr = np.array(chosen, dtype=np.int64)

            sel = coords[chosen_arr].astype(np.float32)        # (K, n_atoms, 3)
            bb_K = np.empty((len(chosen), L, 4, 3), dtype=np.float32)
            bb_K[:, :, 0] = sel[:, parsed['n']]
            bb_K[:, :, 1] = sel[:, parsed['ca']]
            bb_K[:, :, 2] = sel[:, parsed['c']]
            bb_K[:, :, 3] = sel[:, parsed['o']]
            cb_K = np.full((len(chosen), L, 3), np.nan, dtype=np.float32)
            present = parsed['cb_present']
            if present.any():
                cb_K[:, present, :] = sel[:, parsed['cb']][:, present, :]

        seq = parsed['seq']
        aa_index = np.array([AA20.index(c) if c in AA20 else 20 for c in seq], dtype=np.int32)
        # Field names follow the two layouts this script replaces: the K = 8
        # downstream cache and the Kabsch K = 10 cache.
        fields = {f'bb_{K}': bb_K, f'cb_{K}': cb_K, f'ca_{K}': bb_K[:, :, 1],
                  'seq': np.array(seq), 'aa_index': aa_index,
                  'chain_index': parsed['chain_index'],
                  'parser_version': np.array(PARSER_VERSION),
                  # FIXED(code_fix P14, 2026-10-16): K and the FPS rule are stored in
                  #   every file (old: neither), so a resume can verify them.
                  'K': np.array(K, dtype=np.int64), 'fps': np.array(fps)}
        if K == 8:
            fields['chosen_indices'] = chosen_arr
            fields['res_type_int'] = parsed['res_code']
        else:
            fields[f'chosen_indices_{K}'] = chosen_arr
            fields['res_type_int'] = aa_index
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # FIXED(preretrain_audit_r2 Q14, 2026-10-16): old = written in place, so a
        #   killed worker left a truncated <pid>.npz that made check_out_dir refuse the
        #   directory. New = temporary name (not matched by *.npz), then rename.
        tmp = Path(out_dir) / f'.{pid}.npz.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **fields)
        os.replace(tmp, out_path)
        return pid, 'ok'
    except Exception as e:
        return pid, f'err_{type(e).__name__}_{str(e)[:60]}'


def check_out_dir(out_dir: Path, K: int = 8, fps: str = 'kabsch') -> None:
    """Refuse an output directory that holds files this run would wrongly keep.

    The script skips existing files (resumable), so writing into a directory of
    old files would silently keep the wrong proline atoms, or frames of another
    K / FPS rule.
    """
    # FIXED(code_fix P14, 2026-10-16): old: only the first 50 files, parser_version only.
    #   New: EVERY existing file, parser_version AND K AND fps.
    bad = []
    for p in sorted(Path(out_dir).glob('*.npz')):
        why = existing_file_mismatch(p, K, fps)
        if why is not None:
            bad.append(f'{p.name}: {why}')
    if bad:
        raise SystemExit(
            f'{out_dir} holds {len(bad)} file(s) that do not match this run '
            f'(parser {PARSER_VERSION}, K={K}, fps={fps}), e.g.\n  ' + '\n  '.join(bad[:10]) +
            f'\nMove the directory aside (e.g. to {Path(out_dir).name}_legacy_proline) '
            f'or choose another --out-dir, and run again.')


def main():
    # CLI: worker count, frames per complex, FPS rule, output directory and an
    # optional pid subset.
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--K', type=int, default=8)
    # FIXED(code_fix C10, 2026-10-15): Kabsch FPS is the default for both caches
    #   (the downstream 8-frame cache used raw-coordinate FPS before).
    ap.add_argument('--fps', choices=('frobenius', 'kabsch'), default='kabsch',
                    help='frame selection: Cα RMSD after superposition (default) or raw '
                         'Cα coordinates (old caches, to reproduce earlier results)')
    # FIXED(code_fix P14, 2026-10-16): default from K (was always misato_real_bb).
    ap.add_argument('--out-dir', default=None,
                    help='output directory (relative paths are taken from REPO_ROOT); '
                         'default: misato_real_bb for --K 8, misato_real_bb_kabsch for '
                         '--K 10 (Kabsch FPS), misato_real_bb_K<K>_<fps> otherwise')
    ap.add_argument('--limit', type=int, default=0,
                    help='process at most N pids (0 = all)')
    ap.add_argument('--pids-file', type=str, default='',
                    help='restrict to pids listed (one per line) — '
                         'default: all groups in MD.hdf5')
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.K, args.fps)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    check_out_dir(out_dir, args.K, args.fps)
    if args.pids_file:
        with open(args.pids_file) as f:
            pids = [ln.strip() for ln in f if ln.strip()]
    else:
        with h5py.File(H5_PATH, 'r') as f:
            pids = sorted(f.keys())
    if args.limit:
        pids = pids[:args.limit]
    print(f'misato pids to process: {len(pids)}, K={args.K}, fps={args.fps}, '
          f'out={out_dir}, workers={args.workers}', flush=True)

    # Process complexes in parallel (each worker opens the HDF5 read-only) and
    # tally the status codes ('ok', 'skip', 'no_pid', 'no_residues', 'err_...').
    t0 = time.time()
    counts: dict[str, int] = {}
    status_of: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_pid, p, args.K, args.fps, out_dir): p for p in pids}
        for i, fut in enumerate(as_completed(futures)):
            pid, status = fut.result()
            status_of[pid] = status
            key = (status if status in ('ok', 'skip', 'no_pid', 'no_residues')
                   else 'mismatch' if status.startswith('mismatch') else status[:8])
            counts[key] = counts.get(key, 0) + 1
            if status not in ('ok', 'skip'):
                print(f'  {pid}: {status}', flush=True)
            if (i + 1) % 200 == 0:
                done = counts.get('ok', 0) + counts.get('skip', 0)
                print(f'  [{i + 1}/{len(pids)}] {time.time() - t0:.0f}s  '
                      f'ok+skip={done}', flush=True)

    print(f'\nDONE in {time.time() - t0:.0f}s — {counts}', flush=True)
    # FIXED(preretrain_audit_r2 Q13, 2026-10-16): old = failures only printed, exit 0,
    #   and the corpus builder could not see complexes without a file. New = the status
    #   of every complex goes to <out_dir>/_status.json (read by
    #   build_bioemu_misato_descriptor.py) and any err_* exits 1.
    if not args.limit and not args.pids_file:
        (out_dir / '_status.json').write_text(json.dumps(
            {'parser_version': PARSER_VERSION, 'K': args.K, 'fps': args.fps,
             'n_pids': len(pids), 'counts': counts, 'status': dict(sorted(status_of.items()))},
            indent=1))
    n_err = sum(v for k, v in counts.items() if k.startswith('err_'))
    if n_err:
        print(f'{n_err} complex(es) failed (err_*); see above', flush=True)
        return 1
    # FIXED(code_fix P14, 2026-10-16): an existing file that does not match this run
    #   (written after check_out_dir, e.g. by a concurrent run) fails the build.
    if counts.get('mismatch', 0):
        print(f'{counts["mismatch"]} existing file(s) do not match parser/K/fps; see above',
              flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
