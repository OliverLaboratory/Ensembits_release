"""Kinds of the backbone breaks of every MISATO complex (true chain boundary or missing residues).

What it does
------------
A MISATO `chain_index` (misato_real_bb, from the MISATO `molecules_begin_atom_index`) is a
MISATO MOLECULE, not a polypeptide chain: one molecule often holds several deposited
chains written one after the other (thrombin light + heavy chain, 3V2P chains A–E), and
MISATO also splits ONE deposited chain into two molecules at a missing loop (thrombin
H147→H150 in 1A2C, 1D3Q B183→B191, 4XAR A117→A140, ...). For every complex of
`misato_real_bb` (frame 0 of `bb_8`) this builder lists every backbone break, i.e. every
residue pair with C(i)–N(i+1) > 2.5 Å (the Ensembits chain-break distance) plus every
change of `chain_index`, places it in the deposited PDB entry and gives it a kind:

  * `boundary`     the two sides lie in different deposited chains (or in two copies of
                   one chain); also every chain_index change that cannot be placed
                   (source "chain_index_unplaced");
  * `missing_gap`  both sides lie in one deposited chain and the SEQRES positions
                   (label_seq_id) skip residues: n_missing = label step − 1;
  * `consecutive`  both sides lie in one deposited chain with label_seq_id step 1 (nothing
                   missing; a stretched peptide bond);
  * `unresolved`   an inner break that could not be placed (poor alignment, or no file).

Every break, inner or at a chain_index change, is placed with the same rule (v2,
2026-10-18; v1 treated every chain_index change as a boundary and took n_missing from
auth_seq_id, which ignores insertion codes and numbering jumps):

  1. Deposited residues: first model, CA atoms of polymer residues (label_seq_id given),
     alt-loc blank / "A", one per (auth_asym_id, auth_seq_id, pdbx_PDB_ins_code) in file
     order; one letter per residue (20 standard names, HID / HIE / HIP / CYX / MSE; other
     names "X"); per residue the author id (with insertion code) and the SEQRES position
     label_seq_id.
  2. Exact flanks: the K = 5 MISATO residues left and right of the break must both be
     full length (a flank cut by the sequence end never places a break alone) and are
     searched as exact substrings in each deposited chain in file order; in the first
     chain where the left flank occurs before the right flank, the closest pair places the
     break on its two residues.
  3. Alignment fallback (when 2 does not place the break): (a) up to JOINT_WINDOW (15)
     MISATO residues on each side of the break (not crossing the neighbouring breaks) are
     aligned together (global, free end gaps in the deposited chain) to every deposited
     chain; each side must pass on its own: identity >= ALIGN_MIN_IDENTITY (0.8) and >=
     ALIGN_MIN_IDENTICAL (4) identical residues, or, for a side shorter than 4 residues,
     >= 2 residues with at most one mismatch while the other side is long; the best chain
     (the chain of the segment assignment below among equal scores) places the break on the
     aligned residues nearest to it (at most 2 away; the step is corrected by the offset);
     (b) otherwise every segment between two breaks is
     aligned alone (same thresholds) and assigned to a chain in order (the chain of the
     previous segment when the new one continues it, else an unused copy in file order,
     else a used copy again: the complex is flagged `extra_copies`, MISATO holding more
     copies of a chain than the deposited asymmetric unit, e.g. 1DX6, 1QTI); segments in
     different chains make the break a boundary. The same assignment names the chain of
     each side of every boundary (homo-oligomers: the copy actually assigned).
  4. Same chain and label step > 0 → missing_gap (step > 1) or consecutive (step 1);
     different chains, or a later copy of the same chain → boundary; nothing placed →
     unresolved (inner) / boundary "chain_index_unplaced" (chain_index change).
  5. Physical check (AminoAseed audit check4; v2.1 on the deposited coordinates): the two
     placed residues must be one polypeptide in the DEPOSITED entry, i.e. their Cα distance
     in the deposited first model (alt-loc blank / "A") <= 3.80 * label step + 0.5 Å
     (MISATO simulated each molecule alone, so at a missing loop the two ends were free
     termini and drift apart in frame 0); otherwise (e.g. thrombin entries whose cleaved
     light and heavy chains share one author chain) the break is a `boundary` with source
     "physical" — except (v2.2) a SEQRES step 1 with deposited Cα <= 6.0 Å, which is one
     covalent chain with a strained or poorly modelled peptide bond: `consecutive`, flag
     `strained_bond`. Every placed same-chain break records `deposited_ca`, `frame0_ca`,
     `ca_limit`, `excess` and `physical_basis`; a placed residue without a deposited Cα
     falls back to frame 0 (`physical_basis` "frame0"), where a step-1 pair with C–N > 4 Å
     steps by round(Cα / 3.8) (flag `renumbered_deletion`).

v2.3 (ESM3 / ProToken re-checks of v2.2): (a) whenever a break has more than one
candidate placement (chains, homo-oligomer copies, repeats), the candidates are compared
by geometry: up to GEOM_WINDOW (15) MISATO frame-0 Cα per side are superposed (Kabsch) on
each candidate pair's deposited Cα and the lowest RMSD wins (<= 4 Å; ties within 0.25 Å
keep the sequence placement); the RMSD, the number of pairs and the runner-up are
recorded (`geom_rmsd`, `geom_n_pairs`, `geom_runner_up`; `placed_by` "geometry" and the
sequence result under `sequence_placement` when geometry changed it). (b) Molecules of 1-3
residues are placed between their long neighbours when those lie in one chain (in order,
exact matches between the two anchors), else at the nearest deposited Cα of the same type
within 2 Å after superposing the adjacent long segment. (c) A chain_index change that
stays unplaced but whose frame-0 C–N is < 4 Å (a bond) is kept whole (`consecutive`,
source "unplaced_bonded", step 1). (d) Where MISATO closed a SEQRES gap with a bond (no
break, e.g. thrombin H147→H150 in 2ANK / 2CF9 / 4UFD, 4JN2 B, 6EO9 H), the simulated
topology is kept (step 1); such pairs are only counted (`n_bonded_label_gaps`).

PDB-format entries have no label_seq_id: they are used only as a fallback, with n_missing
from auth_seq_id (`n_missing_source` "auth_seq_id (PDB fallback)").

The segment assignment of step 3 also names the chain of every side (`left_chain`,
`right_chain`), so homo-oligomers record the copy actually assigned (3V2P: A|B, B|C, ...),
and its agreement with step 2 is counted in the sidecar. Each break records the author ids
(with insertion code) and the label_seq_id of both sides and how it was placed.

The deposited entries are gzipped mmCIF files in `--deposited-dir` (default
`data/external/rcsb_mmcif/`, see rmsf_common.RCSB_MMCIF_DIR); `--download` fetches the
missing ones from RCSB (https://files.rcsb.org/download/<PDBID>.cif.gz, at most 8 in
parallel, 3 retries) and records each file's sha256 and download date in
`<dir>/_manifest.json`; the table's sidecar records the sha256 of every file used.

Consumers: `data_prep/baselines/_chain_pieces.py` cuts MISATO proteins at `boundary` and
`unresolved` breaks and keeps a chain whole across `missing_gap` and `consecutive` breaks,
numbering residues by SEQRES position (authors' decision of 2026-10-18, option A).

Paper
-----
`appendix.tex` \\label{app:baselines} (the baselines are applied per chain).

Origin
------
New (2026-10-18). v1 = the auditor's scratch script `classify_breaks.py` (ProToken
fidelity audit) made a builder; v2 (same day, ESM3 / AminoAseed / 3Di fidelity audits) =
chain_index changes classified too, SEQRES (label_seq_id) step, full-length flanks only,
alignment fallback, physical check, assigned chain per side, extra-copy flag, mmCIF in
data/external/rcsb_mmcif with a manifest.

Used by the reported model: no (baseline caches of the MISATO RMSF / downstream rows).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/misato_real_bb/<PID>.npz   (bb_8 frame 0, seq, chain_index)
          data/external/rcsb_mmcif/<PDBID>.cif.gz             (--deposited-dir; --download)
  writes  data/external/rcsb_mmcif/<PDBID>.cif.gz, _manifest.json   (--download only)
          data/cached_descriptors/misato_chain_breaks.json   (+ .provenance.json; refuses
          an existing file unless --overwrite)

    {"schema": "misato-chain-breaks-v2", "rule": ..., "numbering": ...,
     "complexes": {PID: {"L": L, "deposited": file | null, "extra_copies": bool,
                   "breaks": [{"pos": i, "cn": Å, "ca": Å, "kind": ..., "source":
                   "deposited" | "chain_index" | "chain_index_unplaced" | "physical" |
                   "no_deposited_file", "origin": "inner" | "chain_index",
                   "placed_by": "flank" | "alignment" | null, "n_missing": int | null,
                   "n_missing_source": "label_seq_id" | "ca_distance" | "auth_seq_id (PDB
                   fallback)" | null, "label_step": int | null,
                   "left_chain", "right_chain", "left_auth", "right_auth", "left_label",
                   "right_label", ("physical", "ca_limit", "excess", "renumbered_deletion",
                   "segment_agrees", "why")}, ...]}}}

  `pos` is the index of the first residue after the break (0 < pos < L).

How to run
----------
    python data_prep/build_misato_chain_breaks.py --download --workers 16      # all complexes
    python data_prep/build_misato_chain_breaks.py --deposited-dir <dir> --out <json> \\
        --pids 1A2C,3V2P [--overwrite]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))   # rmsf_common
from rmsf_common import MISATO_CHAIN_BREAKS, MISATO_REAL_BB_DIR, RCSB_MMCIF_DIR  # noqa: E402

SCHEMA = 'misato-chain-breaks-v2'   # v2.1: physical check on deposited coordinates; v2.2: strained bonds
RULE_VERSION = '2.3'
CN_BREAK_A = 2.5          # C(i)-N(i+1) distance of a break (Ensembits chain-break distance)
FLANK = 5                 # residues of MISATO sequence matched on each side of a break
ALIGN_MIN_IDENTITY = 0.8  # segment alignment: identical / segment length
ALIGN_MIN_IDENTICAL = 4   # segment alignment: identical residues
CA_STEP_MAX_A = 3.80      # physical check: DEPOSITED Cα distance <= 3.80 * step + 0.5 Å
CA_SLACK_A = 0.5
CN_STRETCH_A = 4.0      # a step-1 pair with C–N above this is a renumbered deletion
STRAINED_BOND_MAX_A = 6.0   # v2.2: step 1 with deposited Cα <= 6.0 Å = one chain, strained bond
THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU',
         'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
         'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
ONE = {v: k for k, v in THREE.items()}
ONE.update({'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'CYX': 'C', 'MSE': 'M'})
RULE = ('breaks = C(i)-N(i+1) > 2.5 A in frame 0 of misato_real_bb bb_8, plus every '
        'chain_index change; every break is placed in the deposited mmCIF (first model, CA '
        'of polymer residues, alt-loc blank/A): (1) exact match of full-length 5-residue '
        'MISATO flanks (first author chain in file order, closest flank pair), else (2) '
        'global alignment with free end gaps of each MISATO segment between breaks to each '
        'deposited chain (identity >= 0.8, >= 4 identical), segments assigned in order '
        '(continuation, unused copy, reused copy = extra_copies); same chain with SEQRES '
        '(label_seq_id) step > 1 -> missing_gap (n_missing = step - 1), step 1 -> '
        'consecutive, different chains or copies -> boundary, not placed -> unresolved '
        '(chain_index change: boundary, source chain_index_unplaced); physical check on the '
        'deposited first-model Ca distance of the placed residues <= 3.80 * step + 0.5 A, '
        'else boundary (source physical), except step 1 with deposited Ca <= 6.0 A -> '
        'consecutive (strained_bond); frame-0 Ca only when a deposited Ca is missing '
        '(then a step-1 pair with C-N > 4 A steps by round(Ca / 3.8), renumbered_deletion)')
NUMBERING = ('pieces cut at boundary and unresolved breaks; inside a piece the residue '
             'index starts at 0 and advances by the SEQRES (label_seq_id) step = 1 + '
             'n_missing at a missing_gap (round(Ca distance / 3.8) if n_missing is unknown), '
             'by 1 otherwise')


# ---------------------------------------------------------------- deposited entries
def _open_text(path: Path):
    return gzip.open(path, 'rt') if path.suffix == '.gz' else open(path)


def deposited_file(dep_dir: Path, pid: str) -> Path | None:
    """<PID>.cif[.gz] (preferred) or <PID>.pdb[.gz] in dep_dir (upper or lower case)."""
    for ext in ('.cif.gz', '.cif', '.pdb.gz', '.pdb', '.ent.gz', '.ent'):
        for stem in (pid.upper(), pid.lower()):
            p = Path(dep_dir) / f'{stem}{ext}'
            if p.exists():
                return p
    return None


def _chains_pdb(fh) -> dict:
    """PDB fallback: no label_seq_id (None)."""
    chains: dict = {}
    seen = set()
    for line in fh:
        if line.startswith('ENDMDL'):
            break
        if (line.startswith('ATOM') or line.startswith('HETATM')) and line[12:16].strip() == 'CA' \
                and line[16] in ' A' and line[17:20] in ONE:
            key = (line[21], line[22:26], line[26])
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(line[21], []).append(
                (ONE[line[17:20]], int(line[22:26]), line[26].strip(), None,
                 (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    return chains


def _xyz(xs, ys, zs, i):
    try:
        return (float(xs[i]), float(ys[i]), float(zs[i]))
    except (IndexError, ValueError):
        return None


def _chains_cif(fh) -> dict:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    d = MMCIF2Dict(fh)
    col = lambda k: (lambda v: v if isinstance(v, list) else [v])(d.get(f'_atom_site.{k}', []))
    atom, alt = col('label_atom_id'), col('label_alt_id')
    n = len(atom)
    comp = col('auth_comp_id') or col('label_comp_id')
    asym, seqid, lseq = col('auth_asym_id'), col('auth_seq_id'), col('label_seq_id')
    ins = col('pdbx_PDB_ins_code') or ['?'] * n
    model = col('pdbx_PDB_model_num') or ['1'] * n
    xs, ys, zs = col('Cartn_x'), col('Cartn_y'), col('Cartn_z')
    first_model = model[0] if n else '1'
    chains: dict = {}
    seen = set()
    for i in range(n):
        if model[i] != first_model:
            break
        if atom[i] != 'CA' or alt[i] not in ('.', '?', 'A') or lseq[i] in ('.', '?'):
            continue
        key = (asym[i], seqid[i], ins[i])
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(asym[i], []).append(
            (ONE.get(comp[i], 'X'), int(seqid[i]), '' if ins[i] in ('?', '.') else ins[i],
             int(lseq[i]), _xyz(xs, ys, zs, i)))
    return chains


def deposited_chains(path: Path) -> dict:
    """{auth chain: [(one letter, auth_seq_id, ins_code, label_seq_id | None, CA xyz | None),
    ...]} of the first model, file order."""
    with _open_text(path) as fh:
        if '.cif' in path.suffixes:
            return _chains_cif(fh)
        return _chains_pdb(fh)


def _auth(r) -> str:
    return f'{r[1]}{r[2]}'


# ---------------------------------------------------------------- placement
def flank_place(seq: str, b: int, chains: dict, k: int = FLANK, only=None):
    """Step 2: (chain, deposited index left, deposited index right) of a break placed by
    full-length exact flanks in one chain, ('boundary', left chain, right chain) when the
    flanks lie only in different chains, or None."""
    if b < k or b + k > len(seq):
        return None                                  # a cut flank never places a break
    left, right = seq[b - k:b], seq[b:b + k]
    pseq = {c: ''.join(r[0] for r in v) for c, v in chains.items()}
    for c, ps in pseq.items():
        if only is not None and c != only:
            continue
        li = [i for i in range(len(ps)) if ps.startswith(left, i)]
        ri = [i for i in range(len(ps)) if ps.startswith(right, i)]
        pairs = [(i, j) for i in li for j in ri if j >= i + k]
        if pairs:
            i, j = min(pairs, key=lambda p: p[1] - p[0])
            return ('same', c, i + k - 1, j)
    if only is not None:
        return None
    lc = [c for c, ps in pseq.items() if left in ps]
    rc = [c for c, ps in pseq.items() if right in ps]
    if lc and rc:
        return ('boundary', lc, rc)
    return None


def _aligner():
    from Bio import Align
    a = Align.PairwiseAligner()
    a.mode = 'global'
    a.match_score, a.mismatch_score = 2.0, -1.0
    a.open_gap_score, a.extend_gap_score = -4.0, -0.5
    a.end_deletion_score = 0.0           # deposited residues outside the segment: free
    return a


def align_segment(segseq: str, chainseq: str, aligner) -> tuple | None:
    """(score, n_identical, {segment index: deposited index}) of the best alignment."""
    if not segseq or not chainseq:
        return None
    try:
        aln = aligner.align(chainseq, segseq)[0]
    except (OverflowError, ValueError):
        return None
    mp, ident = {}, 0
    for (t0, t1), (q0, q1) in zip(*aln.aligned):
        for d in range(q1 - q0):
            mp[q0 + d] = t0 + d
            ident += chainseq[t0 + d] == segseq[q0 + d]
    return float(aln.score), ident, mp


def assign_segments(seq: str, cuts: list, chains: dict):
    """Step 3: per segment (start, end) the assigned (chain, {index: deposited index}) or
    None, and whether a chain copy had to be reused (extra_copies)."""
    aligner = _aligner()
    bounds = [0] + list(cuts) + [len(seq)]
    segs = list(zip(bounds[:-1], bounds[1:]))
    pseq = {c: ''.join(r[0] for r in v) for c, v in chains.items()}
    used: dict = {c: [] for c in chains}             # chain -> [(lo, hi) deposited index]
    out, prev, extra = [], None, False
    for s, e in segs:
        q = seq[s:e]
        cands = []
        for order, (c, ps) in enumerate(pseq.items()):
            r = align_segment(q, ps, aligner)
            if r is None:
                continue
            score, ident, mp = r
            if ident < min(ALIGN_MIN_IDENTICAL, len(q)) or ident < ALIGN_MIN_IDENTITY * len(q):
                continue
            lo, hi = min(mp.values()), max(mp.values())
            cands.append((score, order, c, mp, lo, hi))
        if not cands:
            out.append(None); prev = None; continue
        best = max(x[0] for x in cands)
        top = [x for x in cands if x[0] >= best - 1e-9]
        pick = None
        if prev is not None:                         # continuation of the previous segment
            pick = next((x for x in top if x[2] == prev[0] and x[4] > prev[1]), None)
        if pick is None:                             # unused copy, file order
            free = [x for x in top if not any(x[4] <= h and x[5] >= l for l, h in used[x[2]])]
            pick = min(free, key=lambda x: x[1]) if free else None
        if pick is None:                             # every copy used: reuse (extra copy)
            pick = min(top, key=lambda x: x[1]); extra = True
        _, _, c, mp, lo, hi = pick
        used[c].append((lo, hi))
        out.append((c, {s + k: v for k, v in mp.items()}))
        prev = (c, hi)
    return segs, out, extra


def _ids(chains, c, i):
    r = chains[c][i]
    return {'auth': _auth(r), 'label': r[3]}


def _dep_ca(chains, c, i, j):
    """Cα–Cα distance of deposited residues i and j of chain c (first model), or None."""
    a, b = chains[c][i][4] if len(chains[c][i]) > 4 else None, \
        chains[c][j][4] if len(chains[c][j]) > 4 else None
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.subtract(a, b)))


def _kind_from_step(step: int, ca: float, cn: float, use_label: bool,
                    dep_ca: float | None = None, dep_step: int | None = None) -> dict:
    """missing_gap / consecutive from the deposited step of one chain, with the physical
    check on the DEPOSITED coordinates (v2.1): the Cα distance of the two placed residues
    in the deposited first model (`dep_ca`, over their own label step `dep_step`) above
    3.80 * step + 0.5 Å is not one polypeptide -> boundary ("physical": True). MISATO
    simulated each molecule alone, so frame-0 distances across a missing loop overstate
    the gap; frame 0 (`ca`, 3.80 Å per step) is used only when a placed residue has no
    deposited Cα (`physical_basis` "frame0"). A step-1 pair with frame-0 C–N > 4 Å in that
    fallback steps by the Cα estimate (flag "renumbered_deletion")."""
    rec = {'label_step': int(step), 'frame0_ca': round(float(ca), 3),
           'n_missing_source': 'label_seq_id' if use_label else 'auth_seq_id (PDB fallback)'}
    if step <= 0:
        return {**rec, 'kind': 'unresolved', 'n_missing': None, 'why': 'deposited step <= 0'}
    if dep_ca is not None:
        d, st, basis = dep_ca, (dep_step or step), 'deposited'
        rec['deposited_ca'] = round(dep_ca, 3)
    else:
        d, st, basis = ca, step, 'frame0'
    lim = CA_STEP_MAX_A * st + CA_SLACK_A
    rec.update(physical_basis=basis, ca_limit=round(lim, 3), excess=round(d - lim, 3))
    if d > lim:
        # v2.2: SEQRES step 1 with deposited Cα <= 6.0 Å is one covalent chain with a
        #   strained / poorly modelled peptide bond (step-1 failures are bimodal: 4.3-5.6 Å
        #   vs >= 6.7 Å, cleaved chains >= 8 Å) -> consecutive, kept whole, flagged.
        if basis == 'deposited' and st == 1 and step == 1 and d <= STRAINED_BOND_MAX_A:
            return {**rec, 'kind': 'consecutive', 'n_missing': 0, 'strained_bond': True}
        return {**rec, 'kind': 'boundary', 'n_missing': int(step - 1), 'physical': True,
                'why': 'Ca distance too long for one polypeptide'}
    if basis == 'frame0' and step == 1 and cn > CN_STRETCH_A:
        est = max(1, int(round(ca / 3.8)))
        return {**rec, 'kind': 'missing_gap' if est > 1 else 'consecutive', 'n_missing': est - 1,
                'n_missing_source': 'ca_distance', 'renumbered_deletion': True}
    return {**rec, 'kind': 'missing_gap' if step > 1 else 'consecutive', 'n_missing': int(step - 1)}


MAX_FLANK_OFFSET = 2     # joint alignment: nearest aligned residue at most 2 from the break
JOINT_WINDOW = 15        # joint alignment: residues per side (bounded by the next breaks)
SHORT_SIDE_MIN = 2       # a side shorter than ALIGN_MIN_IDENTICAL: >= 2 residues, <= 1 mismatch


def _side_ok(mp, q, lo, hi, chainseq) -> tuple[bool, bool]:
    """(passes, long) for one side of a joint alignment: a side of >= ALIGN_MIN_IDENTICAL
    residues needs identity >= ALIGN_MIN_IDENTITY and >= ALIGN_MIN_IDENTICAL identical; a
    shorter side (>= SHORT_SIDE_MIN residues) at most one mismatch (it never places a break
    alone: the other side must be long)."""
    n = hi - lo
    ident = sum(1 for k in range(lo, hi) if k in mp and chainseq[mp[k]] == q[k])
    if n >= ALIGN_MIN_IDENTICAL:
        return ident >= ALIGN_MIN_IDENTICAL and ident >= ALIGN_MIN_IDENTITY * n, True
    return n >= SHORT_SIDE_MIN and ident >= n - 1, False


def joint_place(seq, s0, b, e1, chains, prefer=None):
    """Step 3a: the two segments around the break, [s0, b) + [b, e1), aligned together to
    each deposited chain (so the aligner places the gap between them). Each side must pass
    the identity thresholds; the break is placed on the aligned residues nearest to it (at
    most MAX_FLANK_OFFSET residues away, e.g. when the flank residue differs from the
    deposited one). -> (chain, deposited index left, deposited index right, offset left,
    offset right) of the best chain (`prefer` among equal scores), or None."""
    aligner = _aligner()
    s0, e1 = max(s0, b - JOINT_WINDOW), min(e1, b + JOINT_WINDOW)
    q = seq[s0:e1]
    k = b - s0
    cands = []
    for c, v in chains.items():
        ps = ''.join(x[0] for x in v)
        r = align_segment(q, ps, aligner)
        if r is None:
            continue
        score, ident, mp = r
        (okl, longl), (okr, longr) = _side_ok(mp, q, 0, k, ps), _side_ok(mp, q, k, len(q), ps)
        if not (okl and okr and (longl or longr)):
            continue
        dl = next((d for d in range(MAX_FLANK_OFFSET + 1) if (k - 1 - d) in mp), None)
        dr = next((d for d in range(MAX_FLANK_OFFSET + 1) if (k + d) in mp), None)
        if dl is None or dr is None or mp[k + dr] <= mp[k - 1 - dl]:
            continue
        cands.append((score, c, mp[k - 1 - dl], mp[k + dr], dl, dr))
    if not cands:
        return None
    best = max(x[0] for x in cands)
    top = [x for x in cands if x[0] >= best - 1e-9]
    pick = next((x for x in top if x[1] == prefer), top[0])
    return pick[1:]


def place_break(seq, b, ca, cn, chains, seg_map, use_label, s0=None, e1=None) -> dict:
    """Kind of the break before MISATO residue b (steps 2-5 of the module docstring);
    [s0, e1) = the two MISATO segments around it (for the joint alignment)."""
    side = lambda i: seg_map.get(i)                  # (chain, deposited index) or None
    sl, sr = side(b - 1), side(b)
    seg_rec = {'left_chain': sl[0] if sl else None, 'right_chain': sr[0] if sr else None}
    fp = flank_place(seq, b, chains)
    if fp is not None and fp[0] == 'same' and sl and sr and sl[0] == sr[0] != fp[1]:
        # homo-oligomers: the same flanks in the copy the segment assignment chose
        fp = flank_place(seq, b, chains, only=sl[0]) or fp
    if fp is not None and fp[0] == 'same':
        _, c, i, j = fp
        L_, R_ = _ids(chains, c, i), _ids(chains, c, j)
        step = (R_['label'] - L_['label']) if use_label else (chains[c][j][1] - chains[c][i][1])
        rec = {'placed_by': 'flank', 'left_chain': c, 'right_chain': c,
               'left_auth': L_['auth'], 'right_auth': R_['auth'],
               'left_label': L_['label'], 'right_label': R_['label'],
               'segment_agrees': bool(sl and sr and sl[0] == sr[0] == c)}
        if not use_label and step <= 1 and j == i + 1:
            step = 1
        return {**rec, **_kind_from_step(step, ca, cn, use_label, _dep_ca(chains, c, i, j), step)}
    if fp is not None:                               # flanks only in different chains
        rec = {'placed_by': 'flank', 'kind': 'boundary', 'n_missing': None,
               'n_missing_source': None, 'label_step': None, **seg_rec,
               'segment_agrees': not (sl and sr and sl[0] == sr[0]
                                      and chains[sl[0]][sr[1]][3] is not None
                                      and sr[1] > sl[1])}
        if sl is None or sr is None:                  # name a chain per side when possible
            rec['left_chain'] = rec['left_chain'] or fp[1][0]
            rec['right_chain'] = rec['right_chain'] or fp[2][0]
        for nm, sd in (('left', sl), ('right', sr)):
            if sd:
                ids = _ids(chains, sd[0], sd[1])
                rec[f'{nm}_auth'], rec[f'{nm}_label'] = ids['auth'], ids['label']
        return rec
    jp = (joint_place(seq, s0, b, e1, chains, prefer=sl[0] if sl else None)
          if s0 is not None else None)
    if jp is not None:                               # step 3a: joint alignment, one chain
        c, i, j, dl, dr = jp
        L_, R_ = _ids(chains, c, i), _ids(chains, c, j)
        step = ((R_['label'] - L_['label']) if use_label
                else (chains[c][j][1] - chains[c][i][1])) - dl - dr
        rec = {'placed_by': 'alignment', 'left_chain': c, 'right_chain': c,
               'left_auth': L_['auth'], 'right_auth': R_['auth'],
               'left_label': L_['label'], 'right_label': R_['label']}
        if dl or dr:
            rec['flank_offset'] = [int(dl), int(dr)]
        raw = step + dl + dr                 # label step of the two aligned residues
        return {**rec, **_kind_from_step(step, ca, cn, use_label, _dep_ca(chains, c, i, j), raw)}
    if sl and sr and sl[0] != sr[0]:                 # step 3b: segments in different chains
        L_, R_ = _ids(chains, sl[0], sl[1]), _ids(chains, sr[0], sr[1])
        rec = {'placed_by': 'alignment', **seg_rec, 'left_auth': L_['auth'],
               'right_auth': R_['auth'], 'left_label': L_['label'], 'right_label': R_['label']}
        return {**rec, 'kind': 'boundary', 'n_missing': None, 'n_missing_source': None,
                'label_step': None}
    return {'placed_by': None, 'kind': 'unresolved', 'n_missing': None, 'n_missing_source': None,
            'label_step': None, **seg_rec,
            'why': 'flanks not found and segment alignment below the threshold'}


# ---------------------------------------------------------------- geometric choice (v2.3)
GEOM_WINDOW = 15          # residues per side superposed on the deposited candidates
GEOM_MAX_MISMATCH = 0.2   # a window candidate: contiguous deposited residues, <= 20% mismatches
GEOM_LONG = 4             # a side with fewer residues is "short" (placed by nearest residue)
GEOM_RMSD_MAX = 4.0       # Kabsch RMSD (Å) of a geometric placement (MD frame 0 vs crystal)
SHORT_NEAREST_MAX = 2.0   # short molecule: nearest deposited residue of the same type within 2 Å
GEOM_MAX_CANDS = 24       # window candidates kept per side (fewest mismatches first)
UNPLACED_BONDED_CN_A = 4.0   # an unplaced chain_index change with C–N below this stays whole
GEOM_TIE_A = 0.25         # pairs within 0.25 Å of the best RMSD are ties (symmetric copies)


def _kabsch_rmsd(P, Q):
    """RMSD of P onto Q after optimal superposition; (rmsd, R, pc, qc)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum(1).mean())), R, pc, qc


class _Dep:
    """Deposited chains as arrays: sequence codes, Cα coordinates (NaN when absent)."""
    def __init__(self, chains):
        self.names = list(chains)
        self.seq = {c: np.frombuffer(''.join(r[0] for r in v).encode(), dtype=np.uint8)
                    for c, v in chains.items()}
        self.xyz = {c: np.array([r[4] if len(r) > 4 and r[4] is not None else (np.nan,) * 3
                                 for r in v], dtype=float).reshape(-1, 3)
                    for c, v in chains.items()}

    def window_cands(self, q: str):
        """[(mismatches, chain, start)] of contiguous deposited windows matching q."""
        qa = np.frombuffer(q.encode(), dtype=np.uint8)
        n, out = len(qa), []
        for c in self.names:
            a = self.seq[c]
            m = len(a) - n + 1
            if m <= 0:
                continue
            mism = np.zeros(m, dtype=np.int32)
            for k in range(n):
                mism += a[k:k + m] != qa[k]
            for st in np.where(mism <= GEOM_MAX_MISMATCH * n)[0]:
                out.append((int(mism[st]), c, int(st)))
        out.sort()
        return out[:GEOM_MAX_CANDS]


def _bracket_place(seq, segs, k1, k2, seg_map, chains) -> dict:
    """Short segments k1..k2 between two long segments placed in ONE chain (seg_map): each
    short segment, in order, at its first exact match among the deposited residues between
    the two anchors. -> {MISATO index: (chain, deposited index, 0.0)} or {} if any fails."""
    if k1 == 0 or k2 + 1 >= len(segs):
        return {}
    a, b = seg_map.get(segs[k1 - 1][1] - 1), seg_map.get(segs[k2 + 1][0])
    if a is None or b is None or a[0] != b[0] or b[1] <= a[1]:
        return {}
    c = a[0]
    ps = ''.join(r[0] for r in chains[c])
    pos, out = a[1] + 1, {}
    for k in range(k1, k2 + 1):
        s, e = segs[k]
        j = ps.find(seq[s:e], pos, b[1])
        if j < 0:
            return {}
        for d in range(e - s):
            out[s + d] = (c, j + d, 0.0)
        pos = j + (e - s)
    return out


def short_residue_map(seq, mca, segs, dep: _Dep, seg_map=None, chains=None) -> dict:
    """{MISATO index: (chain, deposited index, distance)} for the residues of short segments
    (< GEOM_LONG): (1) a run of short segments between two long segments placed in one
    chain goes, in order, to its exact matches between them ("bracket"); (2) otherwise the
    adjacent long segment's window (up to GEOM_WINDOW residues) is superposed on each of
    its deposited candidates and each short residue goes to the nearest deposited Cα of the
    same residue type within SHORT_NEAREST_MAX (best candidate = lowest mean distance)."""
    out = {}
    short = [k for k, (s, e) in enumerate(segs) if e - s < GEOM_LONG]
    runs, cur = [], []
    for k in short:
        if cur and k != cur[-1] + 1:
            runs.append(cur); cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    bracketed = set()
    for r in runs:
        got = _bracket_place(seq, segs, r[0], r[-1], seg_map or {}, chains) if chains else {}
        if got:
            out.update(got); bracketed.update(r)
    for k, (s, e) in enumerate(segs):
        if e - s >= GEOM_LONG or k in bracketed:
            continue
        best = None
        for nb in (k - 1, k + 1):                     # neighbouring long segments
            if not 0 <= nb < len(segs) or segs[nb][1] - segs[nb][0] < GEOM_LONG:
                continue
            ns, ne = segs[nb]
            ws, we = (max(ns, ne - GEOM_WINDOW), ne) if nb < k else (ns, min(ne, ns + GEOM_WINDOW))
            for _, c, st in dep.window_cands(seq[ws:we]):
                Q = dep.xyz[c][st:st + (we - ws)]
                if np.isnan(Q).any():
                    continue
                rms, R, pc, qc = _kabsch_rmsd(mca[ws:we], Q)
                if rms > GEOM_RMSD_MAX:
                    continue
                Y = (mca[s:e] - pc) @ R.T + qc
                hits, dsum = {}, 0.0
                for i in range(e - s):
                    bestr = None
                    for c2 in dep.names:
                        same = dep.seq[c2] == ord(seq[s + i])
                        if not same.any():
                            continue
                        dd = np.linalg.norm(dep.xyz[c2] - Y[i], axis=1)
                        dd[~same | np.isnan(dd)] = np.inf
                        j = int(np.argmin(dd))
                        if bestr is None or dd[j] < bestr[2]:
                            bestr = (c2, j, float(dd[j]))
                    if bestr is None or bestr[2] > SHORT_NEAREST_MAX:
                        hits = None; break
                    hits[s + i] = bestr; dsum += bestr[2]
                if hits and (best is None or dsum + rms < best[0]):
                    best = (dsum + rms, hits)
        if best:
            out.update(best[1])
    return out


def geom_place(seq, mca, b, s0, e1, dep: _Dep, short_map: dict):
    """Geometric placement of the break before b: candidates for each side (window matches of
    up to GEOM_WINDOW residues, or the short-residue map), every pair superposed jointly
    (Kabsch of the MISATO frame-0 Cα onto the deposited Cα); -> dict(left=(chain, index),
    right=(chain, index), rmsd, runner_up, n_pairs) of the lowest RMSD, or None."""
    sides = []
    for lo, hi, anchor in ((max(s0, b - GEOM_WINDOW), b, 'end'), (b, min(e1, b + GEOM_WINDOW), 'start')):
        if hi - lo >= GEOM_LONG:
            cands = [(c, st, lo, hi) for _, c, st in dep.window_cands(seq[lo:hi])]
            sides.append(('long', cands, anchor))
        else:
            i = b - 1 if anchor == 'end' else b
            if i not in short_map:
                return None
            c, j, _ = short_map[i]
            sides.append(('short', [(c, j, i, i + 1)], anchor))
    if not sides[0][1] or not sides[1][1]:
        return None
    if sides[0][0] == 'short' and sides[1][0] == 'short':    # both sides short: the map
        (c1, j1, _, _), (c2, j2, _, _) = sides[0][1][0], sides[1][1][0]
        return {'left': (c1, j1), 'right': (c2, j2), 'rmsd': 0.0, 'n_pairs': 1, 'near': [],
                'runner_up': None, 'short_only': True}
    res = []
    for cl in sides[0][1]:
        for cr in sides[1][1]:
            P, Q = [], []
            for (c, st, lo, hi) in (cl, cr):
                P.append(mca[lo:hi]); Q.append(dep.xyz[c][st:st + (hi - lo)])
            P, Q = np.concatenate(P), np.concatenate(Q)
            if len(P) < 3 or np.isnan(Q).any():
                continue
            rms = _kabsch_rmsd(P, Q)[0]
            li = cl[1] + (cl[3] - cl[2]) - 1          # deposited index of residue b - 1
            ri = cr[1]                                # deposited index of residue b
            # one short side: its residue may have drifted (separate MISATO molecule); the
            #   threshold applies to the long side alone, the joint RMSD ranks the pairs
            long_c = cl if sides[0][0] == 'long' else cr
            lrms = rms
            if sides[0][0] != sides[1][0]:
                lq = dep.xyz[long_c[0]][long_c[1]:long_c[1] + (long_c[3] - long_c[2])]
                lrms = _kabsch_rmsd(mca[long_c[2]:long_c[3]], lq)[0]
            if lrms > GEOM_RMSD_MAX:
                continue
            res.append((rms, (cl[0], li), (cr[0], ri)))
    if not res:
        return None
    res.sort(key=lambda x: x[0])
    rms, left, right = res[0]
    if rms > GEOM_RMSD_MAX and sides[0][0] == sides[1][0]:
        return None
    ru = next((x for x in res[1:] if (x[1], x[2]) != (left, right)), None)
    return {'left': left, 'right': right, 'rmsd': rms, 'n_pairs': len(res),
            'near': [(x[1], x[2]) for x in res if x[0] <= rms + GEOM_TIE_A],
            'runner_up': None if ru is None else {'rmsd': round(ru[0], 3), 'left': list(ru[1]),
                                                  'right': list(ru[2])}}


def bonded_label_gaps(seq, segs, dep: _Dep, chains) -> int:
    """Residue pairs inside a MISATO segment (no break: MISATO closed the gap with a bond)
    whose deposited SEQRES step is > 1, counted on exact segment matches (v2.3, (iii))."""
    n = 0
    for s, e in segs:
        if e - s < GEOM_LONG:
            continue
        cands = [x for x in dep.window_cands(seq[s:e]) if x[0] == 0]
        if not cands:
            continue
        _, c, st = cands[0]
        lab = [chains[c][st + k][3] for k in range(e - s)]
        if None in lab:
            continue
        n += int(sum(1 for k in range(1, len(lab)) if lab[k] - lab[k - 1] > 1))
    return n


# ---------------------------------------------------------------- one complex
def _load(pid, rb_dir):
    with np.load(Path(rb_dir) / f'{pid}.npz', allow_pickle=False) as z:
        bb = np.asarray(z['bb_8'][0], dtype=np.float64)
        seq = str(z['seq']); ci = np.asarray(z['chain_index'])
    return bb, seq, ci


def _break_sets(bb, ci):
    cn = np.linalg.norm(bb[:-1, 2] - bb[1:, 0], axis=-1)
    gaps = set((np.where(cn > CN_BREAK_A)[0] + 1).tolist())
    cbound = set((np.where(np.diff(ci) != 0)[0] + 1).tolist())
    return cn, gaps, cbound


def geom_choose(p: dict, g: dict | None, chains, ca, cn, use_label) -> dict:
    """v2.3: the geometric placement `g` decides between candidates. The sequence placement
    `p` is kept (with the geometry recorded) when g agrees with it or g is None; it is
    replaced (placed_by "geometry", the sequence result kept under `sequence_placement`)
    when g places a break the sequence rules left unplaced, or places it on other chains /
    copies / residues among several candidates."""
    if g is None:
        return p
    (cl, li), (cr, ri) = g['left'], g['right']
    L_, R_ = _ids(chains, cl, li), _ids(chains, cr, ri)
    info = {'geom_rmsd': round(g['rmsd'], 3), 'geom_runner_up': g['runner_up'],
            'geom_n_pairs': g['n_pairs']}
    same = (p.get('left_chain'), p.get('right_chain'), p.get('left_label'), p.get('right_label')) \
        == (cl, cr, L_['label'], R_['label'])
    for (al, ai), (ar, aj) in g.get('near', []):      # a tie with the sequence placement
        if (p.get('left_chain'), p.get('right_chain'), p.get('left_label'), p.get('right_label')) \
                == (al, ar, chains[al][ai][3], chains[ar][aj][3]):
            same = True
    placed = p.get('placed_by') is not None and p['kind'] != 'unresolved'
    if same or (placed and g['n_pairs'] <= 1):
        return {**p, **info}
    rec = {'placed_by': 'geometry', 'left_chain': cl, 'right_chain': cr,
           'left_auth': L_['auth'], 'right_auth': R_['auth'],
           'left_label': L_['label'], 'right_label': R_['label'], **info,
           'sequence_placement': {k: p.get(k) for k in ('kind', 'placed_by', 'left_chain',
                                                        'right_chain', 'left_label', 'right_label')}}
    if cl == cr and ri > li:
        step = (R_['label'] - L_['label']) if use_label else (chains[cl][ri][1] - chains[cl][li][1])
        return {**rec, **_kind_from_step(step, ca, cn, use_label, _dep_ca(chains, cl, li, ri), step)}
    return {**rec, 'kind': 'boundary', 'n_missing': None, 'n_missing_source': None,
            'label_step': None}


def complex_breaks(pid: str, rb_dir: Path, dep_dir: Path | None) -> dict:
    """{"L", "breaks", "deposited", "extra_copies"} of one complex (module docstring)."""
    bb, seq, ci = _load(pid, rb_dir)
    L = int(bb.shape[0])
    if len(seq) != L or ci.shape[0] != L:
        raise ValueError(f'{pid}: seq {len(seq)} / chain_index {ci.shape[0]} / bb {L} lengths differ')
    cn, gaps, cbound = _break_sets(bb, ci)
    ca = np.linalg.norm(bb[:-1, 1] - bb[1:, 1], axis=-1)
    allb = sorted(gaps | cbound)
    chains, dep, use_label = None, None, False
    if allb and dep_dir is not None:
        f = deposited_file(Path(dep_dir), pid)
        if f is not None:
            chains, dep = deposited_chains(f), f.name
            use_label = all(r[3] is not None for v in chains.values() for r in v) and bool(chains)
    seg_map, extra = {}, False
    mca = bb[:, 1]
    dep_g, short_map, n_bonded = None, {}, 0
    if chains:
        segs, assigned, extra = assign_segments(seq, allb, chains)
        if use_label:                                  # v2.3 geometry needs mmCIF coordinates
            dep_g = _Dep(chains)
            for (s_, e_), a in zip(segs, assigned):    # sequence map of the segments
                if a is not None:
                    for i, j in a[1].items():
                        seg_map[i] = (a[0], j)
            short_map = short_residue_map(seq, mca, segs, dep_g, seg_map, chains)
            n_bonded = bonded_label_gaps(seq, segs, dep_g, chains)
        for (s, e), a in zip(segs, assigned):
            if a is not None:
                c, mp = a
                for i, j in mp.items():
                    seg_map[i] = (c, j)
    breaks = []
    for b in allb:
        rec = {'pos': int(b), 'cn': round(float(cn[b - 1]), 3), 'ca': round(float(ca[b - 1]), 3)}
        if not chains:
            rec['origin'] = 'chain_index' if b in cbound else 'inner'
            if b in cbound:
                rec.update(kind='boundary', source='chain_index_unplaced', n_missing=None,
                           placed_by=None, why='no deposited file')
            else:
                rec.update(kind='unresolved', source='no_deposited_file', n_missing=None,
                           placed_by=None)
            breaks.append(rec); continue
        k = allb.index(b)
        s0 = allb[k - 1] if k > 0 else 0
        e1 = allb[k + 1] if k + 1 < len(allb) else L
        p = place_break(seq, b, float(ca[b - 1]), float(cn[b - 1]), chains, seg_map, use_label,
                        s0, e1)
        if dep_g is not None:
            p = geom_choose(p, geom_place(seq, mca, b, s0, e1, dep_g, short_map), chains,
                            float(ca[b - 1]), float(cn[b - 1]), use_label)
        rec['origin'] = 'chain_index' if b in cbound else 'inner'
        if p.get('physical'):
            rec['source'] = 'physical'
        elif b in cbound:
            if p['kind'] == 'unresolved' and float(cn[b - 1]) < UNPLACED_BONDED_CN_A:
                # v2.3 (ProToken re-check): an unplaced chain_index change whose frame-0 C–N
                #   is a bond (< 4 Å) is kept whole, step 1, flagged
                p = {**p, 'kind': 'consecutive', 'n_missing': 0, 'label_step': 1,
                     'n_missing_source': None, 'unplaced_bonded': True}
                rec['source'] = 'unplaced_bonded'
            elif p['kind'] == 'unresolved':
                p = {**p, 'kind': 'boundary', 'why': p.get('why', 'not placed')}
                rec['source'] = 'chain_index_unplaced'
            else:
                rec['source'] = 'chain_index'
        else:
            rec['source'] = 'deposited'
        rec.update(p)
        breaks.append(rec)
    n_dep = sum(len(v) for v in (chains or {}).values())
    return {'L': L, 'breaks': breaks, 'deposited': dep, 'extra_copies': bool(extra),
            'n_deposited_residues': n_dep, 'n_bonded_label_gaps': int(n_bonded)}


def _job(args):
    pid, rb_dir, dep_dir = args
    try:
        return pid, complex_breaks(pid, rb_dir, dep_dir), None
    except Exception as e:                       # noqa: BLE001  reported, fails the build
        return pid, None, f'{type(e).__name__}: {e}'


def _needs_file(pid: str, rb_dir: Path) -> bool:
    bb, _, ci = _load(pid, rb_dir)
    _, gaps, cbound = _break_sets(bb, ci)
    return bool(gaps | cbound)


# ---------------------------------------------------------------- download
def _fetch(pid: str, dep_dir: Path, timeout: float, retries: int):
    import urllib.request
    url = f'https://files.rcsb.org/download/{pid.upper()}.cif.gz'
    dst = dep_dir / f'{pid.upper()}.cif.gz'
    tmp = dep_dir / f'.{pid.upper()}.cif.gz.part'
    err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            gzip.decompress(data)                # must be a complete gzip stream
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            return pid, {'file': dst.name, 'sha256': hashlib.sha256(data).hexdigest(),
                         'url': url, 'downloaded': time.strftime('%Y-%m-%d')}, None
        except Exception as e:                   # noqa: BLE001
            err = f'{type(e).__name__}: {e}'
            if tmp.exists():
                tmp.unlink()
            time.sleep(2.0 * (attempt + 1))
    return pid, None, err


def read_manifest(dep_dir: Path) -> dict:
    p = Path(dep_dir) / '_manifest.json'
    return json.loads(p.read_text()) if p.exists() else {'files': {}}


def write_manifest(dep_dir: Path, man: dict) -> None:
    p = Path(dep_dir) / '_manifest.json'
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    os.replace(tmp, p)


def download(pids, dep_dir: Path, workers: int = 8, timeout: float = 60.0,
             retries: int = 3) -> dict:
    """Fetch <PID>.cif.gz from RCSB for the pids without a local entry (at most `workers`
    at a time); the manifest records sha256, URL and date. -> {pid: error}."""
    dep_dir = Path(dep_dir)
    dep_dir.mkdir(parents=True, exist_ok=True)
    man = read_manifest(dep_dir)
    todo = [p for p in pids if deposited_file(dep_dir, p) is None]
    failed = {}
    with ThreadPoolExecutor(max(1, min(int(workers), 8))) as ex:
        for i, (pid, rec, err) in enumerate(ex.map(lambda p: _fetch(p, dep_dir, timeout, retries),
                                                    todo)):
            if err:
                failed[pid] = err
            else:
                man['files'][rec['file']] = rec
            if (i + 1) % 500 == 0:
                write_manifest(dep_dir, man)
                print(f'  [download] {i + 1}/{len(todo)}', flush=True)
    write_manifest(dep_dir, man)
    return failed


# ---------------------------------------------------------------- summary
def summary(complexes: dict) -> dict:
    allb = [b for c in complexes.values() for b in c['breaks']]
    kinds = Counter((b['kind'], b['source']) for b in allb)
    inner = [b for b in allb if b.get('origin', 'inner') == 'inner']
    gaps = [b['n_missing'] for b in allb if b['kind'] == 'missing_gap']
    return {'n_complexes': len(complexes),
            'n_breaks_by_kind_source': {f'{k}/{s}': n for (k, s), n in sorted(kinds.items())},
            'n_breaks_by_placement': dict(Counter(str(b.get('placed_by')) for b in allb)),
            'n_inner_breaks': len(inner),
            'n_chain_index_changes': sum(b.get('origin') == 'chain_index'
                                         or b['source'].startswith('chain_index') for b in allb),
            'n_chain_index_unplaced': sum(b['source'] == 'chain_index_unplaced' for b in allb),
            'n_physical_boundary': sum(b['source'] == 'physical' for b in allb),
            'n_physical_boundary_borderline_lt3A': sum(b['source'] == 'physical' and b['excess'] < 3
                                                       for b in allb),
            'n_renumbered_deletion': sum(bool(b.get('renumbered_deletion')) for b in allb),
            'n_strained_bond': sum(bool(b.get('strained_bond')) for b in allb),
            'n_placed_by_geometry': sum(b.get('placed_by') == 'geometry' for b in allb),
            'n_unplaced_bonded': sum(b['source'] == 'unplaced_bonded' for b in allb),
            'n_bonded_label_gaps': sum(int(c.get('n_bonded_label_gaps', 0)) for c in complexes.values()),
            'n_complexes_bonded_label_gaps': sum(bool(c.get('n_bonded_label_gaps')) for c in complexes.values()),
            'n_physical_basis': dict(Counter(b['physical_basis'] for b in allb
                                             if b.get('physical_basis'))),
            'n_flank_vs_segment_disagree': sum(b.get('segment_agrees') is False for b in allb),
            'n_pdb_fallback_breaks': sum(str(b.get('n_missing_source', '')).startswith('auth')
                                         for b in allb),
            'n_complexes_extra_copies': sum(c['extra_copies'] for c in complexes.values()),
            'n_complexes_with_inner_break': sum(any(b.get('origin', 'inner') == 'inner'
                                                    for b in c['breaks'])
                                                for c in complexes.values()),
            'n_complexes_with_hidden_boundary': sum(
                any(b['kind'] == 'boundary' and b['source'] == 'deposited' for b in c['breaks'])
                for c in complexes.values()),
            'missing_gap_n_missing_median': float(np.median(gaps)) if gaps else None,
            'missing_gap_n_missing_max': int(max(gaps)) if gaps else None,
            'align_min_identity': ALIGN_MIN_IDENTITY, 'align_min_identical': ALIGN_MIN_IDENTICAL}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--deposited-dir', default=str(RCSB_MMCIF_DIR),
                    help='folder of <PDBID>.cif.gz (or .cif / .pdb[.gz]) deposited entries')
    ap.add_argument('--download', action='store_true',
                    help='fetch missing mmCIF files from RCSB into --deposited-dir')
    ap.add_argument('--download-only', action='store_true', help='download, write nothing else')
    ap.add_argument('--rb-dir', default=str(MISATO_REAL_BB_DIR))
    ap.add_argument('--out', default=str(MISATO_CHAIN_BREAKS))
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--pids', default=None, help='comma-separated subset (needs --out)')
    args = ap.parse_args(argv)
    out, rb, dep = Path(args.out), Path(args.rb_dir), Path(args.deposited_dir)
    if args.pids and out.resolve() == Path(MISATO_CHAIN_BREAKS).resolve() and not args.download_only:
        print('[error] --pids writes a partial table: pass --out <scratch file>'); return 1
    if out.exists() and not args.overwrite and not args.download_only:
        print(f'{out} exists; pass --overwrite to rebuild (every MISATO baseline cache made '
              f'with the old table must then be rebuilt)'); return 1
    pids = (args.pids.split(',') if args.pids else
            sorted(p.stem for p in rb.glob('*.npz') if not p.name.startswith('_')))
    t0 = time.time()
    if args.download or args.download_only:
        with ProcessPoolExecutor(args.workers) as ex:
            need = [p for p, n in zip(pids, ex.map(_needs_file, pids, [rb] * len(pids),
                                                   chunksize=64)) if n]
        failed_dl = download(need, dep, workers=min(args.workers, 8))
        print(f'[download] {len(need)} complexes need a deposited file; {len(failed_dl)} failed '
              f'{dict(list(failed_dl.items())[:5])}', flush=True)
        if args.download_only:
            return 1 if failed_dl else 0
    complexes, failed = {}, {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (pid, rec, err) in enumerate(ex.map(_job, [(p, rb, dep) for p in pids],
                                                   chunksize=16)):
            if err:
                failed[pid] = err
            else:
                complexes[pid] = rec
            if (i + 1) % 2000 == 0:
                print(f'  {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)', flush=True)
    if failed:
        print(f'[error] {len(failed)} complexes failed, nothing written: '
              f'{dict(list(failed.items())[:5])}'); return 1
    files_used = sorted({c['deposited'] for c in complexes.values() if c['deposited']})
    shas = {name: hashlib.sha256((dep / name).read_bytes()).hexdigest() for name in files_used}
    h = hashlib.sha256()
    for name in files_used:
        h.update(name.encode()); h.update(bytes.fromhex(shas[name]))
    man = read_manifest(dep)['files']
    table = {'schema': SCHEMA, 'rule_version': RULE_VERSION, 'rule': RULE, 'numbering': NUMBERING, 'cn_break_A': CN_BREAK_A,
             'flank': FLANK, 'complexes': complexes}
    from rmsf_common import misato_real_bb_scan
    from ensembits.data_versions import write_provenance
    from ensembits.featurize import file_sha256
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_text(json.dumps(table, separators=(',', ':')))
    os.replace(tmp, out)
    s = summary(complexes)
    write_provenance(out, {'builder': 'data_prep/build_misato_chain_breaks.py', 'schema': SCHEMA,
                           'rule_version': RULE_VERSION,
                           'output': str(out), 'sha256': file_sha256(out), 'rule': RULE,
                           'numbering': NUMBERING, 'inputs': misato_real_bb_scan(rb),
                           'deposited_dir': str(dep), 'n_deposited_files': len(files_used),
                           'deposited_files_sha256': h.hexdigest(),
                           'deposited_files': shas,
                           'download_dates': dict(Counter(
                               (man.get(n) or {}).get('downloaded', 'unknown') for n in files_used)),
                           'n_deposited_pdb_format': sum('.pdb' in n or '.ent' in n for n in files_used),
                           'partial': bool(args.pids),
                           'created': time.strftime('%Y-%m-%d %H:%M:%S'), **s})
    print(json.dumps(s, indent=1))
    print(f'wrote {out} (+ .provenance.json) in {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
