"""ProToken-1.0 structural-VQ tokenizer wrapper.

Source: https://github.com/issacAzazel/ProToken
Per-residue input: backbone N/Cα/C/O atoms — the encoder reads PDB
files via the upstream ``data_process.preprocess`` module, so we
synthesise a minimal PDB on the fly from whatever backbone is
available (real all-atom coords if provided, else ideal-geometry
reconstruction from Cα).
Per-residue output: VQ index in ``[0, 512)`` → ``(512, 32)`` codebook
lookup → 32-D feature.

Heavy upstream deps: TensorFlow + JAX + the ProToken source tree
(TF SavedModel + Python preprocessing module). Install via the
optional extra: ``pip install -e .[protoken]``.

Multi-chain proteins are tokenized chain-by-chain via the peptide-bond
detector — ProToken assumes single-chain inputs and produces wrong
tokens at artificial cross-chain peptide bonds.

Usage::

    tok = ProToken(
        repo_path="…/ProToken/full",
        device="0",                       # CUDA_VISIBLE_DEVICES (empty = CPU)
    )
    ids = tok.tokenize(ca, seq, bb=bb)   # (L,) int64 (padding stripped)
    cb = tok.codebook                     # (512, 32) float32
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ._backbone import reconstruct_backbone_ideal, split_chains_by_peptide_bond

_THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}


def _write_minimal_pdb(bb: np.ndarray, seq: str, out_path: str) -> None:
    """Single-chain PDB with N/Cα/C/O atoms.

    ``bb`` is ``(L, 4, 3)`` with order ``[N, CA, C, O]``.
    """
    L = bb.shape[0]
    lines = []
    atom_idx = 1
    for r in range(L):
        aa1 = seq[r] if r < len(seq) else "X"
        aa3 = _THREE_LETTER.get(aa1, "UNK")
        for ai, (aname, elem) in enumerate(
                zip(["N", "CA", "C", "O"], ["N", "C", "C", "O"])):
            x, y, z = bb[r, ai]
            lines.append(
                f"ATOM  {atom_idx:5d}  {aname:<3s} {aa3:>3s} A"
                f"{r+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {elem:>2s}\n")
            atom_idx += 1
    lines.append("END\n")
    with open(out_path, "w") as f:
        f.writelines(lines)


def _backbone_from_ca(ca: np.ndarray) -> np.ndarray:
    """``(L, 3)`` Cα → ``(L, 4, 3)`` ``[N, CA, C, O]`` via ideal geometry."""
    n, c, o = reconstruct_backbone_ideal(ca)
    return np.stack([n, ca, c, o], axis=1).astype(np.float32)


class ProToken:
    """Thin wrapper around ProToken-1.0's TF-SavedModel encoder."""

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
                "ProToken requires `tensorflow` and `jax`. "
                "Install: `pip install -e .[protoken]`."
            ) from e

        # ProToken's preprocess uses deprecated jax.tree_map / tree_multimap
        # (removed in jax 0.6) — restore the aliases.
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
        """``(M, 32)`` float32 — ProToken's frozen codebook."""
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

    def _tokenize_single_chain(self, ca: np.ndarray, sequence: str,
                                 bb: Optional[np.ndarray]) -> np.ndarray:
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
            _write_minimal_pdb(bb_arr, sequence, pdb_path)
            inputs, aux, sl = self._preprocess(tdir, task_mode="single")
        if sl != seq_len:
            seq_len = sl
        model = self._get_model(seq_len)
        out = model.encoder(*inputs)
        idx = np.asarray(out["protoken_index"]).astype(np.int64)
        mask = np.asarray(aux["seq_mask"])
        return idx[mask > 0]

    def tokenize(self, ca: np.ndarray, sequence: str,
                 bb: Optional[np.ndarray] = None,
                 *, chain_split: bool = True) -> np.ndarray:
        """Tokenize one protein (single- or multi-chain).

        Args:
            ca: ``(L, 3)`` Cα coordinates.
            sequence: AA sequence string. Needed for the synthetic PDB
                that drives ProToken's upstream preprocessor; pass
                ``'X' * L`` if unknown.
            bb: optional ``(L, 4, 3)`` real backbone ``[N, CA, C, O]``.
            chain_split: if True (default) and ``bb`` is provided, split
                the protein on peptide-bond breaks (>5 Å) and tokenize
                each chain independently, then concatenate.

        Returns:
            ``(L,) int64`` token IDs (padding stripped).
        """
        if ca.ndim != 2 or ca.shape[-1] != 3:
            raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")

        if chain_split and bb is not None:
            chains = split_chains_by_peptide_bond(np.asarray(bb, dtype=np.float32))
            if len(chains) > 1:
                outs = [self._tokenize_single_chain(
                            ca[s:e], sequence[s:e], bb[s:e])
                        for s, e in chains]
                return np.concatenate(outs)
        return self._tokenize_single_chain(ca, sequence, bb)
