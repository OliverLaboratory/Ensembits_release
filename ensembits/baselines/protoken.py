"""ProToken-1.0 structural-VQ tokenizer wrapper.

What it does
------------
ProToken's encoder reads PDB files through its own `data_process.preprocess`
module, so `tokenize(ca, sequence, bb=None)` writes a minimal single-chain PDB
(N / CA / C / O) to a temporary folder and runs the TF SavedModel encoder
(models bucketed by length 512 / 1024 / 2048). With `bb` (real (L, 4, 3)
backbone) the real atoms are written; without it the backbone is rebuilt from
C-alpha with ideal geometry (`_backbone.reconstruct_backbone_ideal`).
Returns the per-residue VQ index in [0, 512); `codebook` is the (512, 32)
frozen codebook. Heavy dependencies (TensorFlow, JAX, the ProToken source tree)
are imported in the constructor.

Paper
-----
`appendix.tex` \\label{app:baselines}, *ProToken* ("the encoder (consuming a
minimal N/C-alpha/C/O backbone) emits a VQ index in [0, 512) ... frozen 32-D
codebook").

Origin
------
Copied from `Ensembits/utils/baselines/protoken.py`. Logic unchanged; docstring
rewritten. The source tree path is passed by the caller (`repo_path`); in this
codebase it is `ensembits.paths.PROTOKEN_SRC`. 2026-10-18 (authors' decision, option A):
`tokenize(..., residue_numbers=)` / `_write_minimal_pdb(..., residue_numbers=)` write given
PDB residue numbers (default 1..L, unchanged), so the MISATO true chains keep their
missing-residue gaps in ProToken's residue index. Editing ensembits/ changes the trainer's
code hash only for future training runs; the checkpoints' recorded code_sha256 (read from
their config by ensembits.featurize.checkpoint_info) is unaffected.

Used by the reported model: no (ProToken baseline rows). Used by
`data_prep/baselines/build_protoken_tokens.py` and `build_protoken_pooled.py`.

Inputs / outputs
----------------
reads `<repo_path>/ProToken_Code_Book.pkl`, `<repo_path>/data_process/`, the
model files under `<repo_path>`; writes only a temporary PDB per call

How to run
----------
    from ensembits.baselines import ProToken
    tok = ProToken(repo_path=PROTOKEN_SRC, device="0")
    ids = tok.tokenize(ca, sequence, bb=bb)
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

from ._backbone import reconstruct_backbone_ideal

_THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}


def _write_minimal_pdb(bb: np.ndarray, seq: str, out_path: str,
                       residue_numbers=None) -> None:
    """Single-chain PDB with N / Cα / C / O atoms.

    `bb` is `(L, 4, 3)` with order `[N, CA, C, O]`. `residue_numbers` (2026-10-18): the
    PDB residue numbers (resSeq), default 1..L; ProToken's preprocess reads them as its
    residue index, so a chain with missing residues keeps its gaps.
    """
    L = bb.shape[0]
    # FIXED(baseline fidelity audit, 2026-10-18, option A): residue numbers can be given
    #   (gap-aware numbering of the MISATO true chains); default 1..L as before.
    nums = (np.arange(1, L + 1) if residue_numbers is None
            else np.asarray(residue_numbers, dtype=np.int64))
    if nums.shape != (L,) or (np.diff(nums) <= 0).any() or nums.min() < -999 or nums.max() > 9999:
        raise ValueError("residue_numbers must be (L,), increasing, within PDB's resSeq range")
    lines = []
    atom_idx = 1
    for r in range(L):
        aa1 = seq[r] if r < len(seq) else "X"
        aa3 = _THREE_LETTER.get(aa1, "UNK")
        for ai, (aname, elem) in enumerate(
                zip(["N", "CA", "C", "O"], ["N", "C", "C", "O"])):
            x, y, z = bb[r, ai]
            # PDB columns: serial 7-11, name 13-16, resname 18-20,
            # chain 22, resSeq 23-26, x 31-38, y 39-46, z 47-54,
            # occupancy 55-60, tempFactor 61-66, element 77-78.
            lines.append(
                f"ATOM  {atom_idx:5d}  {aname:<3s} {aa3:>3s} A"
                f"{int(nums[r]):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {elem:>2s}\n")
            atom_idx += 1
    lines.append("END\n")
    with open(out_path, "w") as f:
        f.writelines(lines)


def _backbone_from_ca(ca: np.ndarray) -> np.ndarray:
    """Cα `(L, 3)` → `(L, 4, 3)` `[N, CA, C, O]` via ideal geometry."""
    n, c, o = reconstruct_backbone_ideal(ca)
    return np.stack([n, ca, c, o], axis=1).astype(np.float32)


class ProToken:
    """Thin wrapper around ProToken-1.0's TF-SavedModel encoder.

    Loading ProToken requires its source tree on `sys.path` (it ships
    a TF SavedModel + a Python preprocessing module). Pass
    `repo_path` as the directory that contains
    `data_process/`, `ProToken_Code_Book.pkl`, and the `model_*` dirs.
    """

    NUM_CODES: int = 512
    CODE_DIM: int = 32

    def __init__(self, repo_path: str | Path, device: str = "0") -> None:
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(
                f"ProToken source not found at {repo_path}. Clone "
                "https://github.com/issacAzazel/ProToken and point "
                "`repo_path` at its `full/` directory.")

        cb_path = repo_path / "ProToken_Code_Book.pkl"
        if not cb_path.exists():
            raise FileNotFoundError(
                f"ProToken codebook not found at {cb_path}; the upstream "
                "release ships it next to the model files.")

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        if device:
            os.environ["CUDA_VISIBLE_DEVICES"] = device

        try:
            import jax
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "ProToken requires `tensorflow` and `jax`. Install with "
                "`pip install tensorflow jax jaxlib`."
            ) from e

        # ProToken's preprocess uses the deprecated jax.tree_map / tree_multimap
        # aliases (removed in jax 0.6).
        if not hasattr(jax, "tree_map"):
            jax.tree_map = jax.tree_util.tree_map
        if not hasattr(jax, "tree_multimap"):
            jax.tree_multimap = jax.tree_util.tree_map

        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
            sys.path.insert(0, str(repo_path / "data_process"))

        from data_process.preprocess import (  # type: ignore[import]
            protoken_encoder_preprocess, init_protoken_model,
        )
        with open(cb_path, "rb") as f:
            cb = pickle.load(f)
        self._codebook = np.asarray(cb, dtype=np.float32)
        self._init_model = init_protoken_model
        self._preprocess = protoken_encoder_preprocess
        self._repo_path = repo_path
        self._models: dict[int, object] = {}

    @property
    def codebook(self) -> np.ndarray:
        """`(M, 32) float32` — ProToken's frozen codebook."""
        return self._codebook

    def _get_model(self, seq_len: int):
        if seq_len <= 512:
            bucket = 512
        elif seq_len <= 1024:
            bucket = 1024
        else:
            bucket = 2048
        if bucket not in self._models:
            self._models[bucket] = self._init_model(seq_len, str(self._repo_path))
        return self._models[bucket]

    def tokenize(self, ca: np.ndarray, sequence: str,
                 bb: np.ndarray | None = None, residue_numbers=None) -> np.ndarray:
        """Tokenize one chain.

        Args:
            ca: `(L, 3)` Cα coords. Used to determine length and as the
                Cα channel; also drives the fallback reconstruction
                when `bb` is None.
            sequence: AA sequence string. Needed to write a valid PDB
                for ProToken's upstream preprocessor; pass `'X'`-padded
                if you don't have it.
            bb: optional `(L, 4, 3)` real backbone `[N, CA, C, O]`. Use
                this whenever real all-atom coords are available — it
                preserves the φ/ψ torsion features that the encoder
                was trained on. When None, falls back to the
                ideal-geometry reconstruction in
                `_backbone.reconstruct_backbone_ideal`.
            residue_numbers: optional `(L,)` increasing PDB residue numbers
                (2026-10-18; default 1..L), e.g. the gap-aware numbering of a chain
                with missing residues.

        Returns:
            `(L,) int64` token IDs (padding stripped).
        """
        if ca.ndim != 2 or ca.shape[-1] != 3:
            raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")
        seq_len = ca.shape[0]
        if bb is None:
            bb_arr = _backbone_from_ca(ca.astype(np.float32))
        else:
            bb_arr = np.asarray(bb, dtype=np.float32)
            if bb_arr.shape != (seq_len, 4, 3):
                raise ValueError(
                    f"bb must have shape ({seq_len}, 4, 3), got {bb_arr.shape}")
        with tempfile.TemporaryDirectory() as tdir:
            pdb_path = os.path.join(tdir, "protein.pdb")
            _write_minimal_pdb(bb_arr, sequence, pdb_path, residue_numbers)
            inputs, aux, sl = self._preprocess(tdir, task_mode="single")
        if sl != seq_len:
            seq_len = sl
        model = self._get_model(seq_len)
        out = model.encoder(*inputs)
        idx = np.asarray(out["protoken_index"]).astype(np.int64)
        mask = np.asarray(aux["seq_mask"])
        return idx[mask > 0]
