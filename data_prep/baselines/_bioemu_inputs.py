"""BioEmu-CATH (and mdCATH-div) inputs of the baseline builders (frames, backbones, Cβ, sequences).

What it does
------------
Gives the baseline builders of this folder (written for MISATO) the same inputs for
the 1,090 BioEmu-CATH systems of `data/cached_descriptors/bioemu_cath_real_bb`:

  * `systems()`: the systems in `layout_rows.json` order (the row order of every
    BioEmu RMSF cache);
  * `CaFrames()`: a read-only mapping {system: (10, L, 3) Cα of bb_10}, used where the
    MISATO builders read `coords_ca8_misato.pkl` (frame 0 = the single frame; all 10 =
    the multi-frame input);
  * `load_bb_for_frame(dataset, pid, frame, ca)`: for "bioemu" the real N/CA/C/O of
    that frame from bb_10 and the real Cβ from `bioemu_cath_extras/<system>.npz`
    (cb_10, NaN for Gly), source "real" (the Cα must match); any other dataset is passed
    to `ensembits.baselines._real_or_ideal_bb.load_bb_for_frame` unchanged;
  * `sequence(system)`: one letter per kept residue from the extras (force-field names
    normalised by data_prep/build_bioemu_extras.py);
  * `MdcathCaFrames()`: {domain: (10, L, 3) ca_10} of the mdCATH-div source (F2).

Paper
-----
`appendix.tex` \\label{app:baselines} (the baselines); BioEmu-CATH RMSF column of
`2_results.tex` \\label{result:rmsf} (phase 2, 2026-10-16).

Origin
------
New (2026-10-16, phase 2: RMSF probe on BioEmu-CATH).

Used by the reported model: no (baseline caches).

Inputs / outputs
----------------
Reads data/cached_descriptors/bioemu_cath_real_bb/{layout_rows.json,<system>.npz} and
data/cached_descriptors/bioemu_cath_extras/<system>.npz. Writes nothing.

How to run
----------
Library module (imported by the builders' `--dataset bioemu`).
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))
from rmsf_common import BIOEMU_REAL_BB_DIR, BIOEMU_EXTRAS_DIR, BIOEMU_LAYOUT  # noqa: E402
from ensembits.paths import MDCATH_DIV_SRC  # noqa: E402


def systems(layout=None) -> list:
    """System names in layout_rows.json order."""
    return [r['system'] for r in json.loads(Path(layout or BIOEMU_LAYOUT).read_text())]


def _bb(system: str) -> np.ndarray:
    with np.load(BIOEMU_REAL_BB_DIR / f'{system}.npz', allow_pickle=False) as z:
        return np.asarray(z['bb_10'], dtype=np.float32)


def _extras(system: str) -> dict:
    p = BIOEMU_EXTRAS_DIR / f'{system}.npz'
    if not p.exists():
        raise FileNotFoundError(f'{p} missing; run data_prep/build_bioemu_extras.py')
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in ('cb_10', 'seq')}


class CaFrames(Mapping):
    """{system: (10, L, 3) float32 Cα of bb_10}, loaded on access."""
    def __init__(self, names=None):
        self._names = list(names or systems())

    def __getitem__(self, k):
        if k not in self._names:
            raise KeyError(k)
        return _bb(k)[:, :, 1]

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


class MdcathCaFrames(Mapping):
    """{domain: (10, L, 3) Cα of the mdCATH-div source ca_10}, manifest order, loaded on
    access (2026-10-16, F2: the mdCATH options of the mini3di builders; ca_10[0] == ca_5[0])."""
    def __init__(self, src=None):
        self._src = Path(src or MDCATH_DIV_SRC)
        self._names = list(json.loads((self._src / '_manifest.json').read_text())
                           ['representatives'].keys())

    def __getitem__(self, k):
        if k not in self._names:
            raise KeyError(k)
        with np.load(self._src / f'{k}.npz', allow_pickle=False) as z:
            return np.asarray(z['ca_10'], dtype=np.float32)

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


def load_bb_for_frame(dataset: str, pid: str, frame_idx: int, ca_frame: np.ndarray,
                      match_atol: float = 1e-3):
    """(bb (L, 4, 3), cb (L, 3) with NaN for Gly, source) of one frame."""
    if dataset != 'bioemu':
        from ensembits.baselines._real_or_ideal_bb import load_bb_for_frame as _lb
        return _lb(dataset, pid, frame_idx, ca_frame, match_atol)
    bb = _bb(pid)[frame_idx]
    if bb.shape[0] != np.asarray(ca_frame).shape[0] or not np.allclose(
            bb[:, 1], ca_frame, atol=match_atol):
        raise ValueError(f'{pid} frame {frame_idx}: Cα does not match bioemu_cath_real_bb')
    cb = np.asarray(_extras(pid)['cb_10'][frame_idx], dtype=np.float32)
    if cb.shape[0] != bb.shape[0]:
        raise ValueError(f'{pid}: extras cb_10 has {cb.shape[0]} residues, bb_10 {bb.shape[0]}')
    return bb, cb, 'real'


def sequence(system: str) -> str:
    """One-letter sequence of the kept residues (extras)."""
    return str(_extras(system)['seq'])
