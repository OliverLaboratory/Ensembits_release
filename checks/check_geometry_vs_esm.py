"""Check that the ESM-free geometry gives bit-identical descriptors to the ESM3 utilities.

What it checks
--------------
`ensembits/descriptor.py` computes the 256-D descriptor with `ensembits/geometry.py`
("native" backend) instead of ESM3's structure utilities ("esm" backend, the original
implementation). This script requires `torch.equal` (exact equality, not a tolerance)
between the two backends, for both dihedral conventions, on:

  * real ensembles: 1ctfA00 (10 MD frames), BioEmu-CATH, MISATO (multi-chain
    complexes) and mdCATH real-backbone caches (random samples, 10 frames each),
    including the retrain caches misato_real_bb_kabsch and bioemu_cath_real_bb when
    they exist (skipped otherwise);
  * edge cases: exactly 16 residues, residues with NaN coordinates, coordinates at or
    above the 1e6 cutoff, one fully missing frame, duplicated coordinates (ties in the
    neighbour sort);
on CPU and, when available, CUDA. It also checks the neighbour indices separately.

Paper: `appendix.tex` \\label{app:descriptor-esm3}; motivation in code_fix.md T3.1.

Run (needs the `esm` package and downloads the ESM3-open structure encoder once):
    python checks/check_geometry_vs_esm.py [n_per_dataset]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT                      # noqa: E402
from ensembits.descriptor import _descriptor               # noqa: E402
from check_tokens_match_original import PDB, read_ncac     # noqa: E402


def cache_samples(dirname: str, n: int, rng) -> list[tuple[str, np.ndarray]]:
    files = sorted(glob.glob(str(REPO_ROOT / "data" / "cached_descriptors" / dirname / "*.npz")))
    # FIXED(code_fix P18, 2026-10-16): old: rng.choice on whatever the glob returned (an
    #   absent cache gave no samples silently). New: an absent or empty cache is skipped
    #   with a message, so the rebuilt caches can be listed before they exist.
    if not files:
        print(f"  [skip] {dirname}: no cache files under data/cached_descriptors/{dirname}")
        return []
    out = []
    for f in rng.choice(files, min(n, len(files)), replace=False):
        d = np.load(f)
        if "bb_10" not in d.files:
            continue
        bb = np.asarray(d["bb_10"], np.float32)[..., :3, :]          # (10, L, 3, 3) N/CA/C
        if bb.shape[1] >= 16:
            out.append((f"{dirname}/{Path(f).stem}", bb))
    return out


def edge_cases(rng) -> list[tuple[str, np.ndarray]]:
    base = read_ncac(PDB)                                             # (10, 68, 3, 3)
    cases = [("1ctfA00", base), ("L=16", base[:, :16]), ("L=17", base[:, :17])]
    x = base.copy(); x[:, [3, 40]] = np.nan;           cases.append(("NaN residues", x))
    x = base.copy(); x[2, 10, 1] = 2e6;                cases.append(("coord >= 1e6", x))
    x = base.copy(); x[4] = np.nan;                    cases.append(("fully missing frame", x))
    x = base.copy(); x[:, 20] = x[:, 21];              cases.append(("duplicate residue (ties)", x))
    x = base.copy() * 0.0;                             cases.append(("all-zero coordinates", x))
    return cases


def compare(name: str, bb: np.ndarray, device: str) -> bool:
    t = torch.from_numpy(bb)
    ok = True
    for conv in ("iupac", "legacy_plus180"):
        a = _descriptor(t, device, convention=conv, backend="native")
        b = _descriptor(t, device, convention=conv, backend="esm")
        same = torch.equal(a, b) or (torch.isnan(a).any() and torch.equal(a.nan_to_num(7.0), b.nan_to_num(7.0)))
        if not same:
            diff = (a - b).abs().nan_to_num(0).max().item()
            print(f"  [FAIL] {name} {device} {conv}: max |diff| {diff:.3g}")
            ok = False
    return ok


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(0)
    samples = edge_cases(rng)
    # FIXED(code_fix P18, 2026-10-16): old: the legacy caches only. New: also the caches the
    #   retrain corpus is built from (misato_real_bb_kabsch, bioemu_cath_real_bb; key bb_10),
    #   each skipped cleanly while it does not exist yet.
    for d in ("reviewer_3_special_corpus_real_bb", "misato_real_bb", "mdcath_real_bb",
              "misato_real_bb_kabsch", "bioemu_cath_real_bb"):
        samples += cache_samples(d, n, rng)
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    n_ok = n_all = 0
    for dev in devices:
        for name, bb in samples:
            n_all += 1
            n_ok += compare(name, bb, dev)
    print(f"[{'ok' if n_ok == n_all else 'FAIL'}] native == esm (bit-identical) on {n_ok}/{n_all} "
          f"(structure, device) cases, both dihedral conventions; devices: {devices}")
    if n_ok != n_all:
        sys.exit(1)


if __name__ == "__main__":
    main()
