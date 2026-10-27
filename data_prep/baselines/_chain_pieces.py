"""Chain pieces and gap-aware residue numbering of the baseline token / feature builders.

What it does
------------
`chain_pieces(dataset, pid, L)` returns `(pieces, residue_index)`:
  * `pieces`: half-open residue intervals `[(start, end), ...]` covering `[0, L)` in
    order, one per true polypeptide chain; each baseline encoder (mini3di, ProtProfileMD,
    AminoAseed, ESM3 structure encoder, ProToken) is run on each piece separately and the
    per-piece outputs are concatenated;
  * `residue_index`: `(L,)` int64, the residue number inside its piece (0 at the first
    residue of every piece), passed to the encoders that take one (AminoAseed's
    residue_index input, ESM3 `encode(..., residue_index=)`, ProToken's PDB residue
    numbers = residue_index + 1). mini3di / Foldseek / ProtProfileMD use the pieces only.

MISATO. A MISATO `chain_index` (misato_real_bb, from the MISATO `molecules_begin_atom_index`)
is a MISATO MOLECULE, not a polypeptide chain: one molecule often holds several deposited
chains one after the other (thrombin light + heavy chain, 3V2P chains A–E, ...), and one
deposited chain is often split into two molecules at a missing loop (thrombin H147→H150).
The pieces come from `data/cached_descriptors/misato_chain_breaks.json`
(`data_prep/build_misato_chain_breaks.py`, schema v2), which places every backbone break
(C–N > 2.5 Å in frame 0, and every chain_index change) in the deposited PDB entry:
  * `boundary` (different deposited chains or copies, a chain_index change that cannot be
    placed, or a same-chain pair too far apart to be one polypeptide) and `unresolved`
    breaks start a new piece;
  * `missing_gap` and `consecutive` breaks stay inside the piece; the residue index
    advances by the SEQRES step (label_seq_id difference) = 1 + n_missing at a missing_gap
    (round(Cα distance / 3.8) when n_missing is unknown), by 1 at a consecutive break.
A complex missing from the table, or whose length differs from it, is an error.

mdCATH-div and BioEmu-CATH: one piece, the whole protein, residue index 0..L-1 (single
chains; no C–N > 2.5 Å in any frame of their real backbones, scan of 2026-10-18), so their
tokens are unchanged.

Authors' decisions of 2026-10-18: the baselines are cut only at true chain boundaries,
matching Ensembits (whose descriptor keeps the protein whole and only zeroes the dihedrals
across C–N > 2.5 Å); option A = true chains from the deposited entries with gap-aware
numbering. Before: every builder cut at C–N > 2.5 Å
(`ensembits.baselines._real_or_ideal_bb.split_chains_by_peptide_bond`, code_fix P5), which
also cut chains at missing-residue gaps. The sidecar value is
`rmsf_common.chain_split_for(dataset)` ("true_chain_v2" for MISATO since the v2 table of
2026-10-18, with the sha256 of the chain-break table, `rmsf_common.chain_breaks_record()`;
v1 caches are refused).

Paper
-----
`appendix.tex` \\label{app:baselines} (the baselines are applied per chain).

Origin
------
New (2026-10-18). Replaces the 2.5 Å split in the builders of this folder;
`split_chains_by_peptide_bond` itself is unchanged (other callers, e.g. the CryptoBench
and fold-pair scripts, still use it).

Used by the reported model: no (baseline caches).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/misato_chain_breaks.json   (MISATO only)
  writes  nothing

How to run
----------
Imported by the builders of `data_prep/baselines/`:
    from _chain_pieces import chain_pieces
    pieces, residue_index = chain_pieces('misato', pid, L)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))   # rmsf_common

SCHEMA = 'misato-chain-breaks-v2'
CUT_KINDS = ('boundary', 'unresolved')          # start a new piece
KEEP_KINDS = ('missing_gap', 'consecutive')     # stay inside the piece
CA_STEP_A = 3.8                                 # Cα–Cα distance of one residue step
_TABLES: dict = {}


def load_breaks(path=None) -> dict:
    """{pid: {"L", "breaks"}} of misato_chain_breaks.json (cached per path and mtime)."""
    if path is None:
        from rmsf_common import MISATO_CHAIN_BREAKS as path
    path = Path(path)
    if not path.exists():
        raise ValueError(f'{path.name} missing: run data_prep/build_misato_chain_breaks.py')
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key not in _TABLES:
        t = json.loads(path.read_text())
        if t.get('schema') != SCHEMA:
            raise ValueError(f'{path.name}: schema {t.get("schema")!r}, expected {SCHEMA!r}')
        _TABLES.clear()
        _TABLES[key] = t['complexes']
    return _TABLES[key]


def pieces_from_breaks(L: int, breaks) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Pieces and residue index of one protein from its break records (see the docstring)."""
    L = int(L)
    step = np.ones(L, dtype=np.int64)            # step[i] = index(i) - index(i-1)
    cuts = []
    for b in sorted(breaks, key=lambda r: r['pos']):
        pos, kind = int(b['pos']), b['kind']
        if not 0 < pos < L:
            raise ValueError(f'break at {pos} outside (0, {L})')
        if kind in CUT_KINDS:
            cuts.append(pos)
        elif kind == 'missing_gap':
            n = b.get('n_missing')
            step[pos] = 1 + int(n) if n is not None else max(1, int(round(float(b['ca']) / CA_STEP_A)))
        elif kind != 'consecutive':
            raise ValueError(f'unknown break kind {kind!r}')
    bounds = [0] + sorted(set(cuts)) + [L]
    pieces = list(zip(bounds[:-1], bounds[1:]))
    ri = np.zeros(L, dtype=np.int64)
    for s, e in pieces:
        if e - s > 1:
            ri[s + 1:e] = np.cumsum(step[s + 1:e])
    return pieces, ri


def chain_pieces(dataset: str, pid: str, L: int, table=None) -> tuple[list, np.ndarray]:
    """(pieces, residue_index) of one protein: MISATO from the chain-break table,
    mdCATH / BioEmu the whole protein numbered 0..L-1."""
    L = int(L)
    if dataset in ('mdcath', 'mdcath_div', 'bioemu'):
        return [(0, L)], np.arange(L, dtype=np.int64)
    if dataset != 'misato':
        raise ValueError(f'unknown dataset {dataset!r}')
    table = load_breaks() if table is None else table
    rec = table.get(pid)
    if rec is None:
        raise ValueError(f'{pid} not in misato_chain_breaks.json (rebuild it with '
                         f'data_prep/build_misato_chain_breaks.py)')
    if int(rec['L']) != L:
        raise ValueError(f'{pid}: misato_chain_breaks.json has L={rec["L"]}, the structure {L}')
    return pieces_from_breaks(L, rec['breaks'])


def check_pieces(pieces, L: int) -> list[tuple[int, int]]:
    """The pieces must be non-empty, in order and cover [0, L) exactly; -> pieces."""
    pieces = [(int(s), int(e)) for s, e in pieces]
    pos = 0
    for s, e in pieces:
        if s != pos or e <= s:
            raise ValueError(f'pieces {pieces[:5]} do not tile [0, {L})')
        pos = e
    if pos != L:
        raise ValueError(f'pieces end at {pos}, the structure has {L} residues')
    return pieces
