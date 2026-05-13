"""Loader + live tokenization for a trained Ensembits RVQ-VAE model.

Drop-in for downstream probes:

    from ensembits.featurize import load_model, tokenize_ensemble, load_bb_for_pid

    ens = load_model("ckpt/combined_esm3")
    bb = load_bb_for_pid("12asA00", dataset="mdcath")   # (P, L, 3, 3) N/CA/C
    ca = bb[:, :, 1, :]                                  # (P, L, 3) Cα
    tokens = tokenize_ensemble(ens, ca, bb_all=bb)       # (L,) int64

The descriptor dispatch resolves the model's ``descriptor_mode`` from
``config.json`` and selects the matching descriptor function. The
shipped tokenizer uses ``descriptor_mode = 'esm3desc'`` (live ESM3 K=16
relative-frame compute on real backbone).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .descriptors import compute_esm3_descriptor as _esm3_desc
from .tokenizer import RVQVAETokenizer

# Descriptor function registry.
# All shipped models use ``esm3desc``; the other entries are reserved for
# future descriptor families (and never expected to fire on the shipped
# checkpoint).
_DESCRIPTOR_FNS = {
    "esm3desc": _esm3_desc,
}

# Modes whose input is full N/CA/C backbone (not Cα-only).
_BACKBONE_MODES = ("esm3desc",)


def _resolve_descriptor_mode(cfg: dict) -> str:
    """Canonical descriptor_mode for the given model config.

    Some legacy configs carry a misleading ``descriptor_mode='dynamic'``
    while the actual descriptor is ESM3-style; the ``descriptor`` field
    (e.g. ``'esm3desc_K16'``) is authoritative when present.
    """
    descriptor = cfg.get("descriptor")
    if descriptor and str(descriptor).startswith("esm3desc"):
        return "esm3desc"
    return cfg.get("descriptor_mode", "esm3desc")


# ============================================================
# Model loading
# ============================================================

@dataclass
class Ensembits:
    """Container for a loaded tokenizer + its config + normalization stats."""

    model: RVQVAETokenizer
    mean: np.ndarray         # (D,) float32
    std: np.ndarray          # (D,) float32
    config: dict
    device: str


def load_model(model_dir: Path | str, device: str | None = None) -> Ensembits:
    """Load a trained tokenizer from a checkpoint directory.

    Expects ``{model_dir}/{config.json, best.pt, stats.npz}``.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    stats = np.load(model_dir / "stats.npz")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RVQVAETokenizer(
        input_dim=cfg["input_dim"],
        num_prototypes=cfg["num_prototypes"],
        codebook_sizes=cfg["codebook_sizes"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        commitment_cost=cfg["commitment_cost"],
        n_encoder_layers=cfg["n_encoder_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        n_queries=cfg["n_queries"],
        n_heads=cfg["n_heads"],
        vq_update=cfg["vq_update"],
        encoder_type=cfg.get("encoder_type", "set_transformer"),
        loss_type=cfg.get("loss_type", "hungarian"),
        aminoaseed_projection=cfg.get("aminoaseed_projection", False),
    )
    state = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    return Ensembits(
        model=model,
        mean=stats["mean"].astype(np.float32),
        std=stats["std"].astype(np.float32),
        config=cfg,
        device=device,
    )


def codebook_size(ens: Ensembits, level: int = 0) -> int:
    """Codebook size at the given RVQ level (0 = primary alphabet)."""
    return int(ens.config["codebook_sizes"][level])


# ============================================================
# Descriptor + tokenization
# ============================================================

def _descriptor(ens: Ensembits, ca_all: np.ndarray,
                bb_all: np.ndarray | None = None) -> torch.Tensor:
    cfg = ens.config
    mode = _resolve_descriptor_mode(cfg)
    fn = _DESCRIPTOR_FNS.get(mode)
    if fn is None:
        raise ValueError(
            f"Unknown descriptor_mode {mode!r}; registered: "
            f"{list(_DESCRIPTOR_FNS.keys())}")
    if mode in _BACKBONE_MODES:
        if bb_all is None:
            raise ValueError(
                f"descriptor_mode={mode!r} requires bb_all (P, L, 3, 3) "
                f"N/CA/C; got None. Use load_bb_for_pid().")
        desc = fn(bb_all.astype(np.float32), k=cfg["k"])      # (P, L, D)
    else:
        desc = fn(ca_all.astype(np.float32), k=cfg["k"],
                  dihedral=cfg.get("dihedral", False))         # (P, L, D)
    desc = desc.transpose(1, 0, 2)                            # (L, P, D)
    desc = (desc - ens.mean) / ens.std
    return torch.as_tensor(desc, dtype=torch.float32, device=ens.device)


@torch.no_grad()
def tokenize_ensemble(ens: Ensembits, ca_all: np.ndarray,
                       bb_all: np.ndarray | None = None) -> np.ndarray:
    """Tokenize a single protein ensemble (primary codebook only).

    For ``esm3desc`` tokenizers (the shipped model), pass ``bb_all`` of
    shape ``(P, L, 3, 3)`` with N/CA/C in the third axis. Cα-only
    tokenizers (flexcode-family models, not shipped here) ignore
    ``bb_all``.

    Returns:
        tokens: ``(L,)`` int64 in ``[0, codebook_sizes[0])``.
    """
    x = _descriptor(ens, ca_all, bb_all=bb_all)
    return ens.model.encode(x).cpu().numpy().astype(np.int64)


@torch.no_grad()
def tokenize_ensemble_all(ens: Ensembits, ca_all: np.ndarray,
                          bb_all: np.ndarray | None = None) -> np.ndarray:
    """Tokenize with the full RVQ token tuple.

    Returns:
        tokens: ``(L, K)`` int64 where ``K = len(codebook_sizes)``; column
        k is the k-th residual-VQ level in ``[0, codebook_sizes[k])``.
    """
    x = _descriptor(ens, ca_all, bb_all=bb_all)
    all_tokens = ens.model.encode_all(x)
    return np.stack([t.cpu().numpy() for t in all_tokens], axis=1).astype(np.int64)


# ============================================================
# Real-backbone cache lookup
# ============================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_bb_for_pid(pid: str, dataset: str = "mdcath", P: int = 10,
                    data_dir: Path | str = DEFAULT_DATA_DIR) -> np.ndarray | None:
    """Load ``(P, L, 3, 3)`` float32 N/CA/C frames for a pid.

    Reads:
        - ``{data_dir}/mdcath_real_bb/<pid>.npz``  for mdcath pids
        - ``{data_dir}/misato_real_bb/<pid>.npz``  for misato pids

    Returns ``None`` if the cache file is missing or doesn't contain a
    ``bb_{P}`` array.
    """
    data_dir = Path(data_dir)
    if dataset == "mdcath":
        path = data_dir / "mdcath_real_bb" / f"{pid}.npz"
    elif dataset == "misato":
        path = data_dir / "misato_real_bb" / f"{pid}.npz"
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected mdcath or misato")
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    key = f"bb_{P}"
    if key not in d.files:
        return None
    bb = np.asarray(d[key], dtype=np.float32)         # (P, L, 4, 3) N/CA/C/O
    return np.stack([bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]], axis=2)


# ============================================================
# One-hot helpers (used by some probes)
# ============================================================

def onehot_tokens(tokens: np.ndarray, M: int) -> np.ndarray:
    """``(L,)`` token ids → ``(L, M)`` float32 one-hot matrix."""
    oh = np.zeros((len(tokens), M), dtype=np.float32)
    oh[np.arange(len(tokens)), tokens] = 1.0
    return oh


def onehot_fulltoken(tokens_all: np.ndarray, codebook_sizes: list[int]) -> np.ndarray:
    """``(L, K)`` RVQ tokens → ``(L, sum(codebook_sizes))`` multi-hot concat."""
    L, K = tokens_all.shape
    assert K == len(codebook_sizes), f"K={K} vs {len(codebook_sizes)}"
    feat = np.zeros((L, sum(codebook_sizes)), dtype=np.float32)
    offset = 0
    idx = np.arange(L)
    for k, M in enumerate(codebook_sizes):
        feat[idx, offset + tokens_all[:, k]] = 1.0
        offset += M
    return feat
