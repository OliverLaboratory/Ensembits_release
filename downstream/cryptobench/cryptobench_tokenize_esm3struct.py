"""CryptoBench: tokenize every chain with the ESM3 structure tokenizer (ESM3-struct baseline).

What it does
------------
Step 2c of the CryptoBench pipeline (see `README.md` in this folder). Since 2026-10-21 a
thin wrapper: `cryptobench_tokenize_baselines.py --tokenizer esm3struct` (whole deposited
chain, gap-aware residue index from label_seq_id passed as `encode(..., residue_index=)`,
camera-ready builder `data_prep/baselines/build_esm3struct_tokens.tokenize_protein`, raw
4096 x 128 codebook read from the loaded encoder). Default tag `esm3struct_wc_<set>`.

Paper
-----
Not in the paper yet: CryptoBench goes into the main text
(`camera_ready/require_fix.md` #24, `camera_ready/code_fix.md` E3).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Baseline as in `appendix.tex` \\label{app:baselines}.

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/cryptobench_tokenize_esm3struct.py`
(literal Hugging Face token removed, locations from `_cb_paths.py`).
FIXED(CB audit C findings 2 / 5, 2026-10-21): old: research-repo
  `build_esm3struct_tokens.tokenize_protein` without pieces, i.e. the chain cut at
  C-N > 5 A and no residue index, codebook copied from the MISATO cache. New: delegates to
  `cryptobench_tokenize_baselines.py --tokenizer esm3struct` (see its header).

Used by the reported model: no (ESM3-struct baseline row).

Inputs / outputs
----------------
As `cryptobench_tokenize_baselines.py`; output `cryptobench_tokens_<tag>.npz`, `_codebook.npy`,
`.npz.provenance.json`.

How to run
----------
  CUDA_VISIBLE_DEVICES=0 $PY downstream/cryptobench/cryptobench_tokenize_esm3struct.py --set apo
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cryptobench_tokenize_baselines as TB  # noqa: E402

if __name__ == "__main__":
    sys.exit(TB.main(["--tokenizer", "esm3struct", *sys.argv[1:]]))
