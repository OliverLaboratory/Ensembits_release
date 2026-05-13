"""ESM2-650M pseudo-likelihood baseline (sequence-only, no structure).

Used only by the ProteinGym α-blend metric: for each variant, computes
the zero-shot pseudo-log-likelihood under the ESM2-650M masked
language model, then blends with a structural-tokenizer score via
``score = α · LL_esm2 + (1 - α) · LL_struct``.

ProteinGym ships pre-computed ESM2-650M scores per assay in its
``zero_shot_substitutions_scores/ESM2/650M/`` directory; the repo
consumes those by default via ``data/proteingym/pg_esm2_scores.json``.
Only use this module if you want to recompute the scores from scratch.

Install the optional extra: ``pip install -e .[esm2]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ESM2:
    """Zero-shot pseudo-log-likelihood scorer using ESM2-650M.

    Per variant, ``pll(variant) - pll(wt)`` over the mutated positions
    is the standard ProteinGym ESM2 score.
    """

    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

    def __init__(self, model_name: str | None = None, device: str = "cuda"):
        try:
            import torch
            from transformers import AutoTokenizer, EsmForMaskedLM
        except ImportError as e:
            raise ImportError(
                "ESM2 requires `torch` and `transformers`. "
                "Install: `pip install -e .[esm2]`."
            ) from e
        name = model_name or self.MODEL_NAME
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = EsmForMaskedLM.from_pretrained(name).to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._torch = torch
        self._device = device

    def per_residue_logits(self, sequence: str) -> np.ndarray:
        """``(L, V)`` per-residue log-softmax over the ESM2 vocab.

        Returns the standard masked-LM logits without modification —
        the caller is responsible for indexing AA tokens to compute a
        ProteinGym-style pseudo-LL.
        """
        torch = self._torch
        enc = self._tokenizer(sequence, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits[0, 1:-1]   # strip BOS/EOS
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.cpu().numpy().astype(np.float32)


def load_proteingym_esm2_scores(path: str | Path) -> dict:
    """Load the pre-computed ESM2-650M scores JSON shipped with ProteinGym.

    Expected format::

        {"ASSAY_NAME": {"variant_id": score_float, ...}, ...}
    """
    import json
    with open(path) as f:
        return json.load(f)
