"""Load a trained Ensembits checkpoint and turn ensembles into tokens.

What it does
------------
  * `load_model(model_dir)` rebuilds the RVQ-VAE from `config.json`, loads
    `best.pt`, and returns an `Ensembits` bundle (model + descriptor mean/std +
    config). The reported checkpoint (λ = 10) is in `ckpt/bioemu_cath_misato_lam10/`.
  * `tokenize_ensemble(ens, ca, bb)` / `tokenize_ensemble_all(...)` compute the
    256-D descriptor of P frames, standardise it with the stored mean/std
    (identity for the reported model, which was trained without
    standardisation), run the set encoder and the residual quantizer, and return
    the level-1 tokens (L,) or all levels (L, N_q).
  * `load_bb_for_pid`, `onehot_tokens`, `onehot_fulltoken`, `codebook_size`:
    small helpers used by the downstream scripts.
  * (2026-10-16, code_fix P12 / P17) `check_checkpoint_finished` (refuse a folder
    without config.json or an unfinished run), `checkpoint_provenance` (variant,
    sha256 of best.pt, code_sha256) and the `<cache>.provenance.json` sidecar
    helpers used by every downstream writer / reader of Ensembits caches.

Paper
-----
  * `1_method.tex`, \\subsection{Problem formulation and Ensembits}: the
    tokenizer T maps a descriptor multiset of any size P to a token.
  * `1_method.tex`, \\paragraph{Residual quantizer.}: only the first level
    tokens (M_1 = 2048) are used downstream (`tokenize_ensemble`).
  * `appendix.tex`, \\label{app:model-impl}: "Descriptors are used without
    standardization" (stats.npz holds mean 0, std 1).

Origin
------
Copied from `scripts/baselines/ensembits_featurize.py`. Changes:
  * the model class is imported from `ensembits.tokenizer` (the training code's
    copy) instead of `src.tokenizer`; `checks/check_tokens_match_original.py`
    verifies the tokens are identical;
  * the legacy descriptor modes (3Di "fused/fixed/dynamic" and the 192-D
    "esm3desc") were removed because the reported model does not use them;
    `_descriptor` now computes the 256-D descriptor (`descriptor_mode =
    "esm3desc_dihedral"`). In the original file this mode was not registered and
    `tokenize_ensemble` raised for it; the downstream caches for the reported
    model were built by `downstream/_pretokenize_misato_dihedral.py`, which uses
    the same descriptor and the same standardisation, so the tokens agree.
  * `load_frames` / `frame_mode_suffix` (CA-only frame loaders for old models)
    were dropped.

Used by the reported model: yes (`load_model` everywhere downstream).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ensembits.paths import CKPT_DIR, REPO_ROOT
from ensembits.descriptor import _descriptor as _dihedral_descriptor

PROJECT_ROOT = REPO_ROOT                 # data/ and output/ live in the research repo
DEFAULT_MODEL_DIR = CKPT_DIR             # the checkpoint reported in the paper


@dataclass
class Ensembits:
    """A loaded tokenizer plus what is needed to feed it."""
    model: object            # VQVAETokenizer (v1) or RVQVAETokenizer (v2)
    mean: np.ndarray         # (D,) float32  descriptor mean used at training time
    std: np.ndarray          # (D,) float32  descriptor std  (identity for the reported model)
    config: dict             # the checkpoint's config.json
    device: str
    version: str             # "v1" (single VQ) or "v2" (RVQ, the reported model)
    # Dihedral sign convention the checkpoint was trained with (code_fix §1):
    # config key "dihedral_convention" if present, else "legacy_plus180" (all
    # checkpoints trained before the fix).
    dihedral_convention: str = "legacy_plus180"
    # Chain-break cutoff (Å) for phi/psi (code_fix §6): config key "chain_break_cutoff"
    # if present, else "auto" (None for legacy checkpoints).
    chain_break_cutoff: object = "auto"


class UnfinishedCheckpointError(RuntimeError):
    """A checkpoint folder without config.json, or whose training run has not finished."""


def check_checkpoint_finished(model_dir: Path | str, allow_unfinished: bool = False) -> dict:
    """Return the checkpoint's config.json; refuse an unfinished or config-less folder.

    FIXED(code_fix P12, 2026-10-16): old: consumers read config.json without checks, and a
      killed run left none (the trainer wrote it only at the end). New: the trainer writes
      config.json at start with "status": "running" / "finished_at": null and updates it at
      the end; every consumer refuses (exception) a folder without config.json or with
      status "running" (or finished_at null) unless allow_unfinished=True (the scripts'
      --allow-unfinished flag). Configs of checkpoints trained before the fix have neither
      key and are accepted.
    """
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        # A missing config cannot be loaded at all, so allow_unfinished does not help here.
        raise UnfinishedCheckpointError(
            f"{model_dir}: no config.json (training not started, killed before the first "
            f"write, or not a checkpoint folder)")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # FIXED(recon B.1, 2026-10-24): also refuse "failed" (exit-4 duplicate stop).
    unfinished = (cfg.get("status") in ("running", "failed")
                  or ("finished_at" in cfg and cfg["finished_at"] is None))
    if unfinished and not allow_unfinished:
        raise UnfinishedCheckpointError(
            f"{model_dir}: config.json says the training run has not finished "
            f"(status={cfg.get('status')!r}, finished_at={cfg.get('finished_at')!r}); "
            f"pass --allow-unfinished to use it anyway")
    if unfinished:
        print(f"[warn] {model_dir}: using an UNFINISHED checkpoint (--allow-unfinished)", flush=True)
    return cfg


# ---- provenance of token caches and result JSONs (code_fix P17) ----------------------
_SHA_CACHE: dict = {}


def file_sha256(path: Path | str) -> str:
    """SHA-256 of a file (cached per path, size and mtime)."""
    import hashlib
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def checkpoint_provenance(model_dir: Path | str) -> dict:
    """The "checkpoint" block written into token-cache sidecars and result JSONs.

    FIXED(code_fix P17, 2026-10-16): new. {variant: folder name, best_pt_sha256,
      code_sha256: the trainer's hash of ensembits/ + train/ (None for checkpoints
      trained before 2026-10-15)}.
    """
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    best = model_dir / "best.pt"
    return {"variant": model_dir.name,
            "best_pt_sha256": file_sha256(best) if best.exists() else None,
            "code_sha256": cfg.get("code_sha256")}


def provenance_sidecar(cache_path: Path | str) -> Path:
    """`<cache>.provenance.json` next to a cache file (never a key inside the cache)."""
    p = Path(cache_path)
    return p.with_name(p.name + ".provenance.json")


def write_provenance_sidecar(cache_path: Path | str, prov: dict, **extra) -> Path:
    """Write {"checkpoint": prov, **extra} to `<cache>.provenance.json`."""
    side = provenance_sidecar(cache_path)
    side.write_text(json.dumps({"checkpoint": prov, **extra}, indent=1))
    return side


class ProvenanceMismatchError(RuntimeError):
    """A cache or result written by another checkpoint than the one being evaluated."""


def same_checkpoint(a: dict | None, b: dict | None) -> bool:
    """True when two checkpoint blocks name the same weights (best.pt hash, and code hash
    when both record one)."""
    if not a or not b:
        return False
    if a.get("best_pt_sha256") != b.get("best_pt_sha256"):
        return False
    ca, cb = a.get("code_sha256"), b.get("code_sha256")
    return ca is None or cb is None or ca == cb


def check_recorded_checkpoint(recorded: dict | None, prov: dict, what: str,
                              force: bool = False) -> str:
    """Compare a recorded checkpoint block with the current one; raise on a mismatch.

    Returns "match", "forced" (mismatch or missing, accepted with force=True) or
    "legacy" (nothing recorded and the current checkpoint predates the fixed trainer,
    i.e. has no code_sha256: caches of such checkpoints were written without
    provenance and are accepted with a warning). A missing record for a checkpoint
    trained by the fixed trainer is refused: every writer of this codebase records
    one, so such a file predates the run.
    """
    if recorded is not None and same_checkpoint(recorded, prov):
        return "match"
    if force:
        print(f"[warn] {what}: checkpoint provenance {'differs' if recorded else 'missing'}; "
              f"accepted because of --force", flush=True)
        return "forced"
    if recorded is None:
        if prov.get("code_sha256") is None:
            print(f"[warn] {what}: no provenance recorded (written before 2026-10-16); "
                  f"accepted for the legacy checkpoint {prov.get('variant')}", flush=True)
            return "legacy"
        raise ProvenanceMismatchError(
            f"{what}: no checkpoint provenance recorded, but {prov.get('variant')} was trained "
            f"by the fixed trainer; the file predates this checkpoint. Move it aside or pass --force")
    raise ProvenanceMismatchError(
        f"{what}: written by checkpoint {recorded.get('variant')} "
        f"(best.pt {str(recorded.get('best_pt_sha256'))[:12]}), not by {prov.get('variant')} "
        f"(best.pt {str(prov.get('best_pt_sha256'))[:12]}). Move it aside or pass --force")


def check_provenance_sidecar(cache_path: Path | str, prov: dict, force: bool = False) -> str:
    """check_recorded_checkpoint against `<cache>.provenance.json` (see there)."""
    side = provenance_sidecar(cache_path)
    recorded = json.loads(side.read_text()).get("checkpoint") if side.exists() else None
    return check_recorded_checkpoint(recorded, prov, f"{Path(cache_path).name}", force=force)


def load_model(model_dir: Path | str = DEFAULT_MODEL_DIR,
               device: str | None = None, allow_unfinished: bool = False) -> Ensembits:
    """Rebuild the tokenizer from `model_dir/config.json` and load `best.pt`.

    Paper: architecture in `appendix.tex` \\label{app:model-impl} (set encoder,
    N_q = 3 codebooks [2048, 128, 128], 3-layer MLP decoder).
    """
    model_dir = Path(model_dir)
    # FIXED(code_fix P12, 2026-10-16): old: json.load(config.json) without checks. New:
    #   refuse a folder without config.json or with an unfinished run (see
    #   check_checkpoint_finished) unless allow_unfinished=True.
    cfg = check_checkpoint_finished(model_dir, allow_unfinished=allow_unfinished)
    stats = np.load(model_dir / "stats.npz")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tcls = cfg.get("tokenizer_class", "VQVAETokenizer")
    if tcls == "RVQVAETokenizer":
        from ensembits.tokenizer.rvqvae import RVQVAETokenizer  # noqa: E402
        model = RVQVAETokenizer(
            input_dim=cfg["input_dim"],               # 256 = D_f
            num_prototypes=cfg["num_prototypes"],     # 10 = P_max decoder outputs
            codebook_sizes=cfg["codebook_sizes"],     # [2048, 128, 128]
            hidden_dim=cfg["hidden_dim"],             # 256
            latent_dim=cfg["latent_dim"],             # 128 = D
            commitment_cost=cfg["commitment_cost"],   # 0.5 = beta
            n_encoder_layers=cfg["n_encoder_layers"], # 4 self-attention blocks
            n_decoder_layers=cfg["n_decoder_layers"], # 3 hidden layers
            n_queries=cfg["n_queries"],               # 8 learnable queries
            n_heads=cfg["n_heads"],                   # 4
            vq_update=cfg["vq_update"],               # "ema"
            encoder_type=cfg.get("encoder_type", "set_transformer"),
            loss_type=cfg.get("loss_type", "hungarian"),
            aminoaseed_projection=cfg.get("aminoaseed_projection", False),
        )
        version = "v2"
    else:
        from ensembits.tokenizer.vqvae import VQVAETokenizer  # noqa: E402
        model = VQVAETokenizer(
            input_dim=cfg["input_dim"],
            num_prototypes=cfg["num_prototypes"],
            codebook_size=cfg["codebook_size"],
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            commitment_cost=cfg["commitment_cost"],
            encoder_type=cfg["encoder_type"],
            loss_type=cfg["loss_type"],
            vq_update=cfg["vq_update"],
            aminoaseed=cfg["aminoaseed"],
        )
        version = "v1"
    state = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    return Ensembits(
        model=model,
        mean=stats["mean"].astype(np.float32),
        std=stats["std"].astype(np.float32),
        config=cfg,
        device=device,
        version=version,
        dihedral_convention=checkpoint_dihedral_convention(cfg),
        chain_break_cutoff=cfg.get("chain_break_cutoff", "auto"),
    )


def checkpoint_dihedral_convention(cfg: dict) -> str:
    """Dihedral convention of a checkpoint (code_fix §1).

    Checkpoints written by the fixed trainer store `dihedral_convention`
    ("iupac"); older ones do not and were trained on the +180 deg features.
    """
    return cfg.get("dihedral_convention", "legacy_plus180")


def codebook_size(ens: "Ensembits") -> int:
    """Primary codebook size (v1: single; v2: first level of the RVQ = M_1)."""
    cfg = ens.config
    if "codebook_size" in cfg:
        return int(cfg["codebook_size"])
    return int(cfg["codebook_sizes"][0])


def default_cache_path(model_dir: Path | str, dataset: str) -> Path:
    """Convention for token cache files:
    ensembits_tokens_<dataset>.pkl for a dir named 'final_model',
    ensembits_<tag>_tokens_<dataset>.pkl otherwise."""
    model_dir = Path(model_dir)
    name = model_dir.name
    if name == "final_model":
        fname = f"ensembits_tokens_{dataset}.pkl"
    else:
        tag = name.replace("final_model_", "").replace("final_model", "") or name
        fname = f"ensembits_{tag}_tokens_{dataset}.pkl"
    return PROJECT_ROOT / "data" / "cached_descriptors" / fname


def _descriptor(ens: Ensembits, ca_all: np.ndarray | None,
                bb_all: np.ndarray | None = None) -> "torch.Tensor":
    """Descriptor of one ensemble, shaped for the encoder: (L, P, 256).

    bb_all: (P, L, 3, 3) N/CA/C coordinates (required; the 256-D descriptor
            needs real backbone atoms). ca_all is accepted for signature
            compatibility with the original file and ignored.
    """
    mode = ens.config.get("descriptor_mode", "")
    if mode != "esm3desc_dihedral" and ens.config.get("input_dim") != 256:
        raise ValueError(
            f"this codebase only carries the 256-D descriptor of the reported "
            f"model; got descriptor_mode={mode!r}")
    if bb_all is None:
        raise ValueError("the 256-D descriptor needs bb_all (P, L, 3, 3) N/CA/C")
    desc = _dihedral_descriptor(torch.from_numpy(np.asarray(bb_all, np.float32)),
                                str(ens.device),
                                convention=ens.dihedral_convention,
                                chain_break_cutoff=ens.chain_break_cutoff).numpy()  # (P, L, 256)
    desc = desc.transpose(1, 0, 2)                            # (L, P, 256)
    desc = (desc - ens.mean) / ens.std                        # identity for the reported model
    return torch.as_tensor(desc, dtype=torch.float32, device=ens.device)


@torch.no_grad()
def tokenize_ensemble(ens: Ensembits, ca_all: np.ndarray | None,
                      bb_all: np.ndarray | None = None) -> np.ndarray:
    """Tokenize one protein ensemble; returns level-1 tokens (L,) in [0, M_1).

    Any number of frames P >= 1 works (the set encoder is P-agnostic);
    P = 1 is the single-frame mode enabled by SFTD (`1_method.tex`,
    \\subsection{Single-frame-to-token distillation}).
    """
    x = _descriptor(ens, ca_all, bb_all=bb_all)
    tokens = ens.model.encode(x).cpu().numpy().astype(np.int64)
    return tokens


@torch.no_grad()
def tokenize_ensemble_all(ens: Ensembits, ca_all: np.ndarray | None,
                          bb_all: np.ndarray | None = None) -> np.ndarray:
    """Tokenize with the full RVQ token tuple: (L, N_q), column l = level l."""
    if ens.version != "v2":
        raise ValueError("tokenize_ensemble_all only supported for v2 (RVQ) models")
    x = _descriptor(ens, ca_all, bb_all=bb_all)
    all_tokens = ens.model.encode_all(x)          # list of (L,) LongTensor
    arr = np.stack([t.cpu().numpy() for t in all_tokens], axis=1)
    return arr.astype(np.int64)


def load_bb_for_pid(pid: str, dataset: str = "mdcath",
                    P: int = 10) -> np.ndarray | None:
    """Load (P, L, 3, 3) float32 N/CA/C from the real-backbone cache for a pid.

    dataset='mdcath' → `data/cached_descriptors/mdcath_real_bb/<pid>.npz`
    dataset='misato' → `data/cached_descriptors/misato_real_bb/<pid>.npz`
    Returns None if the pid has no real-bb cache or the file lacks bb_{P}.
    (Cache layout: bb_{P} is (P, L, 4, 3) with atoms N, CA, C, O; built by
    `data_prep/build_misato_real_bb*.py` and `data_prep/build_bioemu_real_bb.py`.)
    """
    if dataset == "mdcath":
        path = PROJECT_ROOT / "data" / "cached_descriptors" / "mdcath_real_bb" / f"{pid}.npz"
    elif dataset == "misato":
        path = PROJECT_ROOT / "data" / "cached_descriptors" / "misato_real_bb" / f"{pid}.npz"
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    key = f"bb_{P}"
    if key not in d.files:
        return None
    bb = np.asarray(d[key], dtype=np.float32)     # (P, L, 4, 3) N/CA/C/O
    return np.stack([bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]], axis=2)


def onehot_fulltoken(tokens_all: np.ndarray, codebook_sizes: list[int]) -> np.ndarray:
    """Per-residue multi-hot vector concatenating one-hots over each RVQ level.

    tokens_all: (L, N_q) from tokenize_ensemble_all; returns (L, sum(M_l)).
    """
    L, K = tokens_all.shape
    assert K == len(codebook_sizes)
    total = sum(codebook_sizes)
    feat = np.zeros((L, total), dtype=np.float32)
    offset = 0
    idx = np.arange(L)
    for k, M in enumerate(codebook_sizes):
        feat[idx, offset + tokens_all[:, k]] = 1.0
        offset += M
    return feat


def onehot_tokens(tokens: np.ndarray, M: int) -> np.ndarray:
    """Convert (L,) token ids -> (L, M) one-hot float32."""
    oh = np.zeros((len(tokens), M), dtype=np.float32)
    oh[np.arange(len(tokens)), tokens] = 1.0
    return oh
