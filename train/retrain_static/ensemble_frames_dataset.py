"""Dataset that feeds BioEmu-CATH + MISATO ensemble frames to StructTokenBench's
VQ-VAE trainer as independent single structures.

What it does
------------
Each (protein, frame) pair becomes one training item delivering the same
backbone coordinates that Ensembits trained on, minus the set encoder.
Implements the PretrainPDBDataset contract so the upstream trainer runs
unmodified.

BioEmu-CATH files store backbone (bb_10) without a sequence field; the
sequence is loaded from the companion extras file
(data/cached_descriptors/bioemu_cath_extras/<system>.npz).

Paper
-----
`appendix.tex` static retrain appendix: "identical corpus, splits and budget".

Origin
------
Copied from submission_exp/src/rebuttal_analyses/sametrain_baselines/
ensemble_frames_dataset.py. Changed: BioEmu extras handling for seq,
paths resolved through ensembits.paths, docstring rewritten.

How to run
----------
Not run standalone; imported by train_static.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from esm.utils.constants import esm3 as C

from protein_chain import WrappedProteinChain
import util as stb_util

ATOM37 = 37
SPLIT_KEY = {"train": "train", "validation": "val", "test": "test"}


class EnsembleFramesDataset(torch.utils.data.Dataset):
    """BioEmu-CATH + MISATO ensemble frames as independent single structures."""

    SPLIT_NAME = {"test": []}

    def __init__(self, **kwargs):
        self.split = kwargs["split"]
        self.py_logger = kwargs["py_logger"]
        self.seq_tokenizer = kwargs["seq_tokenizer"]
        self.truncation_length = int(kwargs.get("truncation_length", 512))
        self.filter_length = int(kwargs.get("filter_length", 512))
        self.crop_mode = kwargs.get("crop_mode", "random")
        self.frames_per_protein = int(kwargs.get("frames_per_protein", 10))
        self.manifest_path = kwargs["manifest_path"]
        self.extras_dir = kwargs.get("extras_dir", "")
        self.fast_dev_run = bool(kwargs.get("fast_dev_run", False))
        self.structure_pad_token_id = -1
        self._epoch = 0

        manifest = json.load(open(self.manifest_path))
        proteins = manifest[SPLIT_KEY[self.split]]

        if self.crop_mode == "filter":
            proteins = [p for p in proteins if p["L"] <= self.filter_length]

        self.items = [(p, f) for p in proteins
                      for f in range(min(self.frames_per_protein, p["P"]))]
        if self.fast_dev_run:
            self.items = self.items[:500]

        n_cropped = sum(1 for p, _ in self.items if p["L"] > self.truncation_length)
        self.py_logger.info(
            f"[EnsembleFramesDataset:{self.split}] {len(proteins)} proteins -> "
            f"{len(self.items)} single-frame structures "
            f"(crop_mode={self.crop_mode}, truncation={self.truncation_length}, "
            f"{n_cropped} items cropped)"
        )

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def shard(self, shard_idx: int, num_shards: int):
        self.py_logger.info(
            f"[EnsembleFramesDataset:{self.split}] shard {shard_idx}/{num_shards}")
        self.items = self.items[shard_idx::num_shards]

    def _crop_start(self, entry, frame, length):
        span = length - self.truncation_length
        if span <= 0:
            return 0
        if self.crop_mode == "random" and self.split == "train":
            seed = hash((entry["pid"], frame, self._epoch)) & 0xFFFFFFFF
            return int(np.random.RandomState(seed).randint(0, span + 1))
        return 0

    def _load_seq(self, entry, d):
        """Load sequence from the npz, or from the BioEmu extras file."""
        if "seq" in d:
            return str(d["seq"])
        if self.extras_dir:
            extras_path = os.path.join(self.extras_dir, os.path.basename(entry["path"]))
            if os.path.exists(extras_path):
                ext = np.load(extras_path, allow_pickle=True)
                return str(ext["seq"])
        return None

    def __getitem__(self, index):
        entry, frame = self.items[index]
        try:
            with np.load(entry["path"], allow_pickle=True) as d:
                bb = d["bb_10"][frame]
                seq = self._load_seq(entry, d)
        except Exception as e:
            self.py_logger.info(f"[EnsembleFramesDataset] failed {entry['path']}: {e}")
            return None

        if seq is None:
            self.py_logger.info(
                f"[EnsembleFramesDataset] no seq for {entry['pid']}")
            return None

        L = bb.shape[0]
        if len(seq) != L:
            self.py_logger.info(
                f"[EnsembleFramesDataset] seq/coord length mismatch {entry['pid']}")
            return None

        start = self._crop_start(entry, frame, L)
        end = min(start + self.truncation_length, L)
        bb, seq = bb[start:end], seq[start:end]
        length = end - start

        atom37 = np.full((length, ATOM37, 3), np.nan, dtype=np.float32)
        atom37[:, :3, :] = bb[:, :3, :]

        finite = np.isfinite(atom37[:, :3, :]).all(axis=(1, 2))
        if not finite.all():
            if finite.sum() < 16:
                return None
            atom37 = atom37[finite]
            seq = "".join(c for c, k in zip(seq, finite) if k)

        chain = WrappedProteinChain.from_atom37(
            atom37, id=entry["pid"], sequence=seq,
            residue_index=np.arange(1, len(seq) + 1),
        )
        coords, _plddt, residue_index = chain.to_structure_encoder_inputs()

        sequence = chain.sequence.replace(C.MASK_STR_SHORT, "<mask>")
        seq_ids = torch.tensor(
            self.seq_tokenizer.encode(sequence, add_special_tokens=False),
            dtype=torch.int64,
        )
        if len(seq_ids) != coords.shape[1]:
            self.py_logger.info(
                f"[EnsembleFramesDataset] token/coord mismatch {entry['pid']}")
            return None

        return coords[0], residue_index[0], seq_ids, chain

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        coords, residue_index, seq_ids, pdb_chain = tuple(zip(*batch))

        coords = stb_util.pad_structures(
            coords, constant_value=torch.inf,
            truncation_length=self.truncation_length,
        )
        attention_mask = coords[:, :, 0, 0] == torch.inf
        residue_index = stb_util.pad_structures(
            residue_index, constant_value=0,
            truncation_length=self.truncation_length,
            pad_length=coords.shape[1],
        )
        assert C.SEQUENCE_PAD_TOKEN == 1
        seq_ids = stb_util.pad_structures(
            seq_ids, constant_value=1,
            truncation_length=self.truncation_length,
            pad_length=coords.shape[1],
        )
        return {"input_list": (coords, attention_mask, residue_index, seq_ids,
                               pdb_chain)}

    def sanity_check(self):
        raise AssertionError("not needed")

    def load_all_structures(self):
        raise AssertionError("not in use")
