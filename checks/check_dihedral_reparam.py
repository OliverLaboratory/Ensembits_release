"""Check that the planned φ/ψ sign fix can be applied without retraining.

What it checks
--------------
The descriptor's φ/ψ come out shifted by 180°, i.e. the stored features are
(-sin, -cos) of the true angles (see `ensembits/descriptor.py`, `_dihedral`).
Fixing the formula negates the 64 dihedral columns of every descriptor
(4 values x 16 neighbour slots). Because the encoder's first layer is linear and
the stored normalisation is the identity, negating the matching 64 input columns
of that layer — and the matching 640 output rows of the decoder (10 states x 64)
— gives exactly the same model on corrected inputs. This script verifies it:

  tokens(old checkpoint, old features) == tokens(converted checkpoint, fixed features)
  recon(converted, fixed) == recon(old, old) with its dihedral block negated

on 1ctfA00 (10 MD frames). It does not modify any file.

Paper: `appendix.tex` \\label{app:descriptor-esm3} describes the true sin/cos;
this is how the code will be brought in line (camera_ready/code_fix.md §1).

Run:  python checks/check_dihedral_reparam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import LEGACY_CKPT_DIR              # noqa: E402
from ensembits.descriptor import _descriptor             # noqa: E402
from ensembits.featurize import load_model               # noqa: E402
from check_tokens_match_original import PDB, read_ncac   # noqa: E402

# Columns holding (sin phi, cos phi, sin psi, cos psi) in each 16-number slot.
DIH = torch.tensor([16 * j + c for j in range(16) for c in range(12, 16)])


def main():
    coords = torch.from_numpy(read_ncac(PDB))
    desc_old = _descriptor(coords, "cpu", convention="legacy_plus180")  # what old checkpoints saw
    desc_new = _descriptor(coords, "cpu", convention="iupac")           # the fixed code
    # The fix must change exactly the 64 dihedral columns, by a sign flip.
    flip = desc_old.clone(); flip[..., DIH] *= -1
    assert torch.allclose(desc_new, flip, atol=1e-6), "iupac != legacy with dihedral columns negated"
    print("[ok] iupac descriptor == legacy descriptor with the 64 dihedral columns negated")
    # FIXED(code_fix P18, 2026-10-16): old: CKPT_DIR (the reported checkpoint, which becomes the
    #   retrained one after E0). New: LEGACY_CKPT_DIR, the pre-fix checkpoint this check is about.
    m = load_model(LEGACY_CKPT_DIR, device="cpu").model.eval()
    x = lambda d: d.transpose(0, 1).contiguous()                       # (P, L, 256) -> (L, P, 256)
    with torch.no_grad():
        t_old = np.stack([t.numpy() for t in m.encode_all(x(desc_old))], -1)
        r_old = m.decode_all(m.encode_all(x(desc_old)))
        # The conversion described in code_fix.md §1 (option A).
        m.encoder.per_element_mlp[0].weight[:, DIH] *= -1
        rows = torch.cat([s * 256 + DIH for s in range(m.decoder.num_states)])
        m.decoder.mlp[-1].weight[rows] *= -1
        m.decoder.mlp[-1].bias[rows] *= -1
        t_new = np.stack([t.numpy() for t in m.encode_all(x(desc_new))], -1)
        r_new = m.decode_all(m.encode_all(x(desc_new)))
    r_old[..., DIH] *= -1
    assert np.array_equal(t_old, t_new), "tokens changed"
    print(f"[ok] tokens identical at all levels {t_new.shape}")
    print(f"[ok] reconstruction max |diff| after sign map: {float((r_new - r_old).abs().max()):.3g}")


if __name__ == "__main__":
    main()
