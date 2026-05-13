"""Populate ``ensembits-repro/{data,ckpt}/`` with symlinks to the canonical
caches under ``/home/shik2/multiconf-token/``.

Lets the rest of the repo work entirely off paths inside ``ensembits-repro/``,
matching the "reproduction repo" contract. Idempotent: skips links that
already point at the right target.

After this runs, the repo layout is:

    ensembits-repro/
    ├── ckpt/combined_esm3/{best.pt, config.json, stats.npz, history.json}
    └── data/
        ├── mdcath_real_bb/   (symlink → canonical dir)
        ├── misato_real_bb/   (symlink → canonical dir)
        ├── tokens/<baseline>_<dataset>.npz       (per-tokenizer post-fix caches)
        ├── codebooks/<baseline>.npy
        ├── labels/{rmsf_mdcath.npy, rmsf_misato.npz, misato_*.{json,npz}}
        └── splits/{mdcath_splits.json, misato_splits.json}
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_ROOT = Path("/home/shik2/multiconf-token")
CACHE = SRC_ROOT / "data" / "cached_descriptors"
ENS_DATA = SRC_ROOT / "Ensembits" / "data"
ANNOT = SRC_ROOT / "data" / "annotations"
SHIPPED_CKPT = SRC_ROOT / "output" / "final_model_combined_mdcath_misato_P10_esm3desc_K16_rvq_2048_128_128_varP_consMSE01_distillMax_P10_realbb"

# ── What to link ────────────────────────────────────────────────────

# (target_relative_to_REPO, source_absolute_path)
LINKS: list[tuple[str, Path]] = []

# 1. Real-backbone caches
LINKS.append(("data/mdcath_real_bb",  CACHE / "mdcath_real_bb"))
LINKS.append(("data/misato_real_bb",  CACHE / "misato_real_bb"))

# 2. Shipped checkpoint
for fn in ("best.pt", "config.json", "stats.npz", "history.json"):
    LINKS.append((f"ckpt/combined_esm3/{fn}", SHIPPED_CKPT / fn))

# 3. Per-baseline token caches (post-fix) + codebooks
_TOKEN_FILES = {
    # baseline → (mdcath_cache, misato_cache, codebook_or_None)
    "3di_tokens":   ("mini3di_tokens_mdcath.npz", "mini3di_tokens_misato.npz", "mini3di_centroids.npy"),
    "mini3di":      ("mini3di_features_mdcath.npz", "mini3di_features_misato.npz", None),
    # vote_3di is a per-residue (L,) int64 token (majority vote across 8 mini3di
    # frames). Canonical applies a 2-D mini3di centroid lookup at load time
    # (ec_classify_conv1d_misato.py:150-167) — we expose the same centroid file
    # as `--codebook` so `load_features_per_pid` does the lookup automatically.
    "vote_3di":     (None, "vote_3di_misato.npz", "mini3di_centroids.npy"),
    "aminoaseed":   ("aminoaseed_tokens_mdcath.npz", "aminoaseed_tokens_misato.npz", "aminoaseed_codebook.npy"),
    "esm3struct":   ("esm3struct_tokens_mdcath.npz", "esm3struct_tokens_misato.npz", "esm3struct_codebook.npy"),
    "protoken":     ("protoken_tokens_mdcath.npz",  "protoken_tokens_misato.npz",  "protoken_codebook.npy"),
    # protprofile_K* features are (L, 20) probability histograms; the canonical
    # conv1d input is (L, 60) — each bin's probability interleaved with its 2-D
    # mini3di centroid (see ec_classify_conv1d_misato.py:116-131). We expose
    # `mini3di_centroids.npy` as the `--codebook` for these baselines so that
    # `load_features_per_pid` applies the centroid-interleave transform
    # automatically.
    "protprofile_K8":  (None, "protprofile_K8_misato.npz", "mini3di_centroids.npy"),
    "protprofile_K10": ("protprofile_K10_mdcath.npz", None, "mini3di_centroids.npy"),
    "protprofile_K5":  ("protprofile_K5_mdcath.npz",  "protprofile_K5_misato.npz",  "mini3di_centroids.npy"),
}
for tag, (md_fn, mi_fn, cb_fn) in _TOKEN_FILES.items():
    if md_fn:
        LINKS.append((f"data/tokens/{tag}_mdcath.npz", CACHE / md_fn))
    if mi_fn:
        LINKS.append((f"data/tokens/{tag}_misato.npz", CACHE / mi_fn))
    if cb_fn:
        LINKS.append((f"data/codebooks/{tag}.npy", CACHE / cb_fn))

# 4. Ours/combined/ESM3 token caches (the shipped variant)
LINKS.append((
    "data/tokens/ours_combined_esm3_misato.npz",
    CACHE / "ensembits_combined_mdcath_misato_P10_esm3desc_K16_rvq_2048_128_128_varP_consMSE01_distillMax_P10_realbb_tokens_misato.npz",
))
LINKS.append((
    "data/tokens/ours_combined_esm3_P1inf_misato.npz",
    CACHE / "ensembits_combined_mdcath_misato_P10_esm3desc_K16_rvq_2048_128_128_varP_consMSE01_distillMax_P10_realbb_P1inf_tokens_misato.npz",
))
# Flat mdcath tokens shipped with the checkpoint dir
LINKS.append(("data/tokens/ours_combined_esm3_mdcath_flat.npz",
              SHIPPED_CKPT / "tokens_mdcath_flat.npz"))

# 5. Labels (mdcath flat + misato per-pid)
LINKS.append(("data/labels/rmsf_mdcath.npy", ENS_DATA / "rmsf.npy"))
LINKS.append(("data/labels/aa_mdcath.npy",   ENS_DATA / "aa_labels.npy"))
LINKS.append(("data/labels/rmsf_misato.npz", ENS_DATA / "misato_rmsf.npz"))
LINKS.append(("data/labels/misato_binding_site.npz", CACHE / "misato_binding_site.npz"))
LINKS.append(("data/labels/misato_ligand_maccs.npz", CACHE / "misato_ligand_maccs.npz"))
LINKS.append(("data/labels/misato_affinity.json", ANNOT / "misato_affinity.json"))
LINKS.append(("data/labels/misato_ec.json", ANNOT / "misato_ec.json"))
LINKS.append(("data/labels/misato_go.json", ANNOT / "misato_go.json"))

# 6. Splits
LINKS.append(("data/splits/mdcath_splits.json", ENS_DATA / "splits.json"))
LINKS.append(("data/splits/misato_splits.json", ENS_DATA / "misato_splits.json"))

# 7. Local s_1 cache for ANOVA
LINKS.append(("data/local_s1_cache.npz",
              SRC_ROOT / "experiments" / "results" / "shared_states" / "token_pca_residues_aligned.npz"))

# 8. ProteinGym mutation-effects (per-tokenizer WT/mut + codebook + DMS + ESM2 anchor)
LINKS.append(("data/pg/dms_scores.json", ENS_DATA / "pg_dms_scores.json"))
LINKS.append(("data/pg/esm2_scores.json", ENS_DATA / "pg_esm2_scores.json"))
_PG_TOKENIZERS = ["3di_tokens", "aminoaseed", "esm3struct", "protoken", "combined_esm3"]
# Some tokenizers' PG codebooks live under non-PG-prefix names; fall back to those.
_PG_CB_FALLBACK = {
    "protoken": ENS_DATA / "protoken.codebook.npy",
}
for tag in _PG_TOKENIZERS:
    wt = ENS_DATA / f"pg_wt_tokens.{tag}.npz"
    mt = ENS_DATA / f"pg_mut_tokens.{tag}.npz"
    cb = ENS_DATA / f"pg_codebook_L0.{tag}.npy"
    if not cb.exists():
        cb = _PG_CB_FALLBACK.get(tag, cb)
    if wt.exists():
        LINKS.append((f"data/pg/wt_tokens/{tag}.npz", wt))
    if mt.exists():
        LINKS.append((f"data/pg/mut_tokens/{tag}.npz", mt))
    if cb and cb.exists():
        LINKS.append((f"data/pg/codebooks/{tag}.npy", cb))


# ── Run ─────────────────────────────────────────────────────────────

def _materialize_ours_codebook(repo: Path, dry_run: bool = False) -> None:
    """Extract the L0 (2048, 128) codebook from ``ckpt/combined_esm3/best.pt``
    and save it at ``data/codebooks/ours_combined_esm3.npy`` so probes
    can consume the shipped tokenizer's integer-token caches without
    needing to re-derive the codebook at runtime."""
    out = repo / "data" / "codebooks" / "ours_combined_esm3.npy"
    if out.exists():
        return
    ckpt = repo / "ckpt" / "combined_esm3" / "best.pt"
    if not ckpt.exists():
        print(f"  MISS: {out.name}  (best.pt not present; skipping codebook derivation)")
        return
    if dry_run:
        print(f"  PLAN: derive {out.relative_to(repo)} from best.pt L0 codebook")
        return
    try:
        import torch
        import numpy as np
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"  SKIP: could not load best.pt ({type(exc).__name__}: {exc})")
        return
    keys = [k for k in state if "vq_levels.0.codebook.weight" in k]
    if not keys:
        print(f"  SKIP: L0 codebook key not in best.pt; nothing to write")
        return
    w = state[keys[0]].numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, w)
    print(f"  DERIVED: {out.relative_to(repo)}  shape={tuple(w.shape)} from best.pt")


def main(dry_run: bool = False) -> int:
    n_made = n_skip = n_miss = n_already = 0
    for rel, src in LINKS:
        dst = REPO / rel
        if not src.exists():
            print(f"  MISS: {rel}  (source: {src})")
            n_miss += 1
            continue
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                n_already += 1
                continue
            print(f"  REPL: {rel}  (was: {dst.resolve()})")
            if not dry_run:
                dst.unlink()
        elif dst.exists():
            # File that isn't a symlink — refuse to overwrite blindly.
            print(f"  HOLD: {rel} already exists as file (not symlink); skipping")
            n_skip += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"  PLAN: {rel}  →  {src}")
            n_skip += 1
        else:
            dst.symlink_to(src)
            print(f"  LINK: {rel}  →  {src}")
            n_made += 1
    # Derive data/codebooks/ours_combined_esm3.npy from best.pt so probes
    # that consume `ours/combined/ESM3` integer-token caches have the
    # matching codebook on disk without a separate Zenodo entry.
    _materialize_ours_codebook(REPO, dry_run=dry_run)

    print(f"\n{'DRY-RUN ' if dry_run else ''}summary: "
          f"made={n_made}  already={n_already}  skipped={n_skip}  missing={n_miss}")
    return n_miss


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be linked without modifying the filesystem.")
    args = ap.parse_args()
    sys.exit(main(dry_run=args.dry_run))
