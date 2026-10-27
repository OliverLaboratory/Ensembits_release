"""Check that this codebase reproduces the original pipeline exactly.

What it checks
--------------
On one protein (1ctfA00, 10 MD frames from mdCATH at 379 K, as used for the
descriptor figure):
  1. `ensembits.descriptor._descriptor` (this codebase) gives the same 256-D
     descriptor as `submission_exp/src/build_esm3desc_dihedral._descriptor`
     (original research code).
  2. The model rebuilt by `ensembits.featurize.load_model` (this codebase; model
     classes from the trainer's package) gives the same tokens at all three RVQ
     levels as `scripts/baselines/ensembits_featurize.load_model` (original;
     model classes from `src/tokenizer`), for P = 10 and for P = 1.

Paper: none directly; this guarantees that the code shipped here is the code
behind the reported numbers.

Run (environment with PyTorch, esm, mdtraj):
    python checks/check_tokens_match_original.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT, LEGACY_CKPT_DIR  # noqa: E402

# Input: N/CA/C of 10 frames of 1ctfA00, written by
# camera_ready/fig_descriptor/extract.py (frames_P10.pdb).
PDB = REPO_ROOT / "camera_ready" / "fig_descriptor" / "build" / "frames_P10.pdb"


def read_ncac(pdb: Path) -> np.ndarray:
    """Multi-model PDB -> (P, L, 3, 3) N/CA/C."""
    models, cur = [], {}
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM"):
            name = line[12:16].strip(); resi = int(line[22:26])
            if name in ("N", "CA", "C"):
                cur.setdefault(resi, {})[name] = [float(line[30:38]), float(line[38:46]),
                                                  float(line[46:54])]
        elif line.startswith("ENDMDL"):
            models.append(np.array([[cur[r][a] for a in ("N", "CA", "C")] for r in sorted(cur)]))
            cur = {}
    return np.stack(models).astype(np.float32)


def main():
    coords = torch.from_numpy(read_ncac(PDB))                    # (10, 68, 3, 3)

    # --- 1. descriptor ------------------------------------------------------
    from ensembits.descriptor import _descriptor as new_desc
    sys.path.insert(0, str(REPO_ROOT / "submission_exp" / "src"))
    from build_esm3desc_dihedral import _descriptor as old_desc
    # The original code used the +180 deg dihedral convention (code_fix §1); compare
    # against it with convention="legacy_plus180".
    d_new = new_desc(coords, "cpu", convention="legacy_plus180"); d_old = old_desc(coords, "cpu")
    assert torch.equal(d_new, d_old), "descriptor differs from the original"
    print(f"[ok] descriptor identical {tuple(d_new.shape)}")

    # --- 2. tokens --------------------------------------------------------------
    from ensembits.featurize import load_model as new_load
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))
    sys.path.insert(1, str(REPO_ROOT))
    import importlib
    old_feat = importlib.import_module("ensembits_featurize")
    orig_dir = REPO_ROOT / "output" / "dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO"
    # FIXED(code_fix P18, 2026-10-16): old: CKPT_DIR (the reported checkpoint, which becomes the
    #   retrained one after E0). New: LEGACY_CKPT_DIR, the pre-fix checkpoint this check is about.
    new = new_load(LEGACY_CKPT_DIR, device="cpu"); old = old_feat.load_model(orig_dir, device="cpu")
    x = d_old.transpose(0, 1).contiguous()                       # (L, P, 256); stats are identity
    for P in (10, 1):
        with torch.no_grad():
            t_new = np.stack([t.numpy() for t in new.model.encode_all(x[:, :P])], -1)
            t_old = np.stack([t.numpy() for t in old.model.encode_all(x[:, :P])], -1)
        assert np.array_equal(t_new, t_old), f"tokens differ at P={P}"
        print(f"[ok] tokens identical at P={P}: {t_new.shape}, residue 42 -> {t_new[41].tolist()}")


if __name__ == "__main__":
    main()
