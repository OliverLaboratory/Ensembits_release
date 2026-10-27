"""ProteinGym scoring of the baseline structure tokenizers (3Di, AminoAseed, ESM3, ProToken).

What it does
------------
Runs the ProteinGym scorer of `pg_score.py` (same variants, same score (a), same
per-assay JSON and per-variant npz, same provenance) with one baseline tokenizer
in place of Ensembits. Each ESMFold structure of the atom14 cache
(`data/proteingym/esmfold_cache_atom14/<assay>/<sha1[:16]>.npy`, layout
N, CA, C, O, CB, ...) is tokenized as one chain, and every variant is scored by
sum_i ||C[t_wt,i] - C[t_mut,i]||_2 with the baseline's own codebook C:

  method       reads                              codebook C
  3di          N, CA, C + real CB (slot 4);        mini3di 2-D centroids (20, 2)
               Gly (CB = 0 in atom14) gets
               mini3di's own virtual CB
  aminoaseed   N, CA, C                            PROJECTED codebook
                                                   linear_proj(codebook.weight) (512, 1024),
                                                   as in every other task
  esm3struct   N, CA, C                            ESM3 structure-encoder codebook (4096, 128)
  protoken     N, CA, C, O (slot 3) + the real     ProToken codebook (512, 32)
               sequence of the structure

Every structure passes `pg_common.check_atom14_slots` (median |C-O| = 1.23 A,
|CA-CB| = 1.53 A), so a slot error cannot recur silently.

Fixes relative to the rows of the paper's table (benchmark_audit D9,
preretrain_audit_r2 Q10), which came from the research script
`submission_exp/src/build_pg_token_caches.py` + `Ensembits/downstreams/proteingym.py`:
  * that script unpacked atom14 as N, CA, C, CB, O: 3Di read the O atom as its CB
    (invalid row) and ProToken read CB as its O;
  * the AminoAseed row used the raw codebook (`pg_codebook_L0.aminoaseed.npy`), not
    the projected one the quantizer compares against;
  * ESM2 / DMS were read in float64 there but float32 in the Ensembits scorer (ties);
    now one scorer for all;
  * alpha was the test-set best; now chosen by leave-one-UniProt-out CV in
    `pg_aggregate.py`.
The TF32 setting of ProToken is left as in the MISATO tasks (benchmark_audit D2:
unchanged). ESMFold structures are single chains, so no chain splitting.

Paper
-----
`2_results.tex`, \\paragraph{Mutation Effect Prediction} and Table
\\label{tab:proteingym_ensemble} (baseline rows); `appendix.tex` \\label{app:baselines}
("On ProteinGym, every method reads the ESMFold predicted structure (N/Calpha/C,
plus O for ProToken and Cbeta for 3Di)").

Origin
------
New (2026-10-19, benchmark_audit D9). Ports the tokenizer adapters of the research
script `submission_exp/src/build_pg_token_caches.py` (with the atom14 slots
corrected) onto `pg_score.py`, using the baseline wrappers of `ensembits.baselines`
(the same ones the MISATO baseline builders use).

Used by the reported model: no (baselines only).

Inputs / outputs
----------------
Reads (REPO_ROOT):
  data/proteingym/esmfold_cache_atom14/<assay>/{manifest.json,<hash>.npy}
  data/external/proteingym_zero_shot_scores/<assay>.csv          (PROTEINGYM_ZERO_SHOT_DIR)
  aminoaseed: data/struct_tokenizer_ckpts/codebook_512x1024-1e+19-linear-fixed-last.ckpt/
              checkpoint/mp_rank_00_model_states.pt + StructTokenBench src (STRUCTTOKENBENCH_SRC)
  protoken:   data/protoken/full/ (PROTOKEN_SRC)
  esm3struct: ESM3 structure encoder weights (Hugging Face cache; HF_TOKEN from the
              environment if the hub asks for it; never written anywhere)
Writes (REPO_ROOT), as pg_score.py:
  output/proteingym/per_assay_pg_<tag>/<assay>.json, per_assay_pg_<tag>.json,
  per_assay_pg_<tag>.assays_scored.json, per_variant_pg_<tag>/<assay>.npz
  (default tag: bl_<method>_d9)

How to run
----------
One GPU (CUDA_VISIBLE_DEVICES selects it; ProToken needs exactly one):
    CUDA_VISIBLE_DEVICES=2 python downstream/pg_score_baselines.py --method 3di --gpus 0
    CUDA_VISIBLE_DEVICES=2 python downstream/pg_score_baselines.py --method aminoaseed --gpus 0
    CUDA_VISIBLE_DEVICES=2 python downstream/pg_score_baselines.py --method esm3struct --gpus 0
    CUDA_VISIBLE_DEVICES=2 python downstream/pg_score_baselines.py --method protoken --gpus 0
Any pg_score.py option is passed through (--assays, --assay-list, --out-tag, --force,
--print-scores).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT                         # noqa: E402
from ensembits.paths import STRUCTTOKENBENCH_SRC, PROTOKEN_SRC        # noqa: E402
import pg_common as C                                                 # noqa: E402
import pg_score                                                       # noqa: E402

# Same structure cache as Ensembits (pg_score_dihedral.py).
pg_score.CACHE_DIR = ROOT / 'data' / 'proteingym' / 'esmfold_cache_atom14'

AMINOASEED_CKPT = ROOT / ('data/struct_tokenizer_ckpts/'
                          'codebook_512x1024-1e+19-linear-fixed-last.ckpt/checkpoint/'
                          'mp_rank_00_model_states.pt')
METHODS = ('3di', 'aminoaseed', 'esm3struct', 'protoken',
           'aminoaseed_retrained', 'esm3struct_retrained')


def atom14_parts(a14: np.ndarray, seq: str):
    """(N, CA, C, O, CB-or-NaN) float32 from an atom14 structure, after the slot check.

    CB is NaN for Gly and for any residue whose CB slot is all zeros (atom14 leaves
    absent atoms at the origin)."""
    C.check_atom14_slots(a14, seq)
    a = np.asarray(a14, dtype=np.float32)
    cb = a[:, C.ATOM14_CB].copy()
    absent = np.array([c == 'G' for c in seq]) | ~np.any(cb != 0, axis=-1)
    cb[absent] = np.nan
    return a[:, C.ATOM14_N], a[:, C.ATOM14_CA], a[:, C.ATOM14_C], a[:, C.ATOM14_O], cb


# ---------------------------------------------------------------- tokenizers
# Each: tokenize(tok, a14, seq) -> ((L,) tokens, (L, 1) tokens); `tok` is what load() made.
def tokenize_3di(tok, a14, seq):
    from ensembits.baselines import three_di_tokens
    from ensembits.baselines._real_or_ideal_bb import cb_from_bb_or_real
    n, ca, c, o, cb = atom14_parts(a14, seq)
    # Real CB where present; Gly gets mini3di's own tetrahedral CB (its training convention).
    cb_full = cb_from_bb_or_real(np.stack([n, ca, c, o], axis=1), cb)
    t = three_di_tokens(ca, n=n, c=c, cb=cb_full)
    return t, t[:, None]


def tokenize_nca_c(tok, a14, seq):
    """AminoAseed and ESM3: real N, CA, C (O and CB are not read by these encoders)."""
    n, ca, c, _o, _cb = atom14_parts(a14, seq)
    t = np.asarray(tok.tokenize(ca, n=n, c=c), dtype=np.int64)
    return t, t[:, None]


def tokenize_protoken(tok, a14, seq):
    if seq is None:
        raise ValueError('ProToken needs the sequence of the structure')
    n, ca, c, o, _cb = atom14_parts(a14, seq)
    t = np.asarray(tok.tokenize(ca, seq, bb=np.stack([n, ca, c, o], axis=1)), dtype=np.int64)
    if len(t) != len(seq):
        raise ValueError(f'ProToken returned {len(t)} tokens for {len(seq)} residues')
    return t, t[:, None]


def make_loader(method: str):
    """(load(device) -> (tokenizer, [codebook]), tokenize fn, weights files for provenance)."""
    if method == '3di':
        from ensembits.baselines import mini3di_centroids
        return (lambda device: (None, [mini3di_centroids()])), tokenize_3di, []
    if method == 'aminoaseed':
        def load(device):
            from ensembits.baselines import AminoAseed
            tok = AminoAseed(weights_path=str(AMINOASEED_CKPT),
                             repo_path=str(STRUCTTOKENBENCH_SRC), device=device)
            # The PROJECTED codebook (tok.codebook = linear_proj(codebook.weight)).
            return tok, [tok.codebook]
        return load, tokenize_nca_c, [AMINOASEED_CKPT]
    if method == 'esm3struct':
        def load(device):
            from ensembits.baselines import ESM3Struct
            tok = ESM3Struct(weights_path=None, device=device)
            return tok, [tok.codebook]
        return load, tokenize_nca_c, []
    if method == 'protoken':
        def load(device):
            from ensembits.baselines import ProToken
            # device '' = keep the caller's CUDA_VISIBLE_DEVICES (JAX / TF use the visible GPU).
            tok = ProToken(repo_path=str(PROTOKEN_SRC), device='')
            return tok, [tok.codebook]
        return load, tokenize_protoken, [PROTOKEN_SRC / 'ProToken_Code_Book.pkl']
    if method in ('aminoaseed_retrained', 'esm3struct_retrained'):
        config_name = method.replace('_retrained', '') + '_retrained'
        config_path = CODE_ROOT / 'train' / 'retrain_static' / 'config' / f'{config_name}.yaml'
        ckpt_dir = ROOT / 'output' / 'static_retrain' / f'{config_name.replace("_retrained", "")}_converged' / 'checkpoints'

        def _find_best_ckpt(d):
            """Find the best-val checkpoint (not last.ckpt)."""
            import glob
            candidates = sorted(glob.glob(str(d / 'epoch=*.ckpt')))
            if not candidates:
                raise FileNotFoundError(f'no best checkpoint in {d}')
            return Path(candidates[-1])

        best_ckpt = _find_best_ckpt(ckpt_dir)

        def load(device, _cfg=str(config_path), _ckpt=str(best_ckpt)):
            from ensembits.baselines import RetrainedTokenizer
            tok = RetrainedTokenizer(config_path=_cfg, checkpoint_path=_ckpt, device=device)
            return tok, [tok.codebook]
        return load, tokenize_nca_c, [best_ckpt]
    raise ValueError(f'unknown method {method!r}')


def provenance(method: str, codebook: np.ndarray, weights: list) -> dict:
    """Checkpoint-like block: 'best_pt_sha256' = sha256 over weight files + codebook bytes."""
    h = hashlib.sha256()
    for w in weights:
        h.update(C.file_sha256(w).encode())
    cb = np.ascontiguousarray(codebook, dtype=np.float32)
    h.update(cb.tobytes())
    here = Path(__file__).resolve().parent
    return {'variant': f'baseline:{method}',
            'best_pt_sha256': h.hexdigest(),
            'codebook_shape': list(cb.shape),
            'codebook_sha256': hashlib.sha256(cb.tobytes()).hexdigest(),
            'weights': [pg_score._rel(w) for w in weights],
            'code_sha256': C.code_sha256([here / 'pg_score_baselines.py', here / 'pg_score.py',
                                          here / 'pg_common.py'])}


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--method', choices=METHODS, required=True)
    known, rest = pre.parse_known_args(argv)
    if '-h' in rest or '--help' in rest:
        print(__doc__); pg_score.build_parser().print_help(); return 0
    if not any(a == '--out-tag' or a.startswith('--out-tag=') for a in rest):
        rest += ['--out-tag', f'bl_{known.method}_d9']
    gpus = pg_score.build_parser().parse_args(rest).gpus
    gpu_list = [g for g in gpus.split(',') if g.strip()]
    if known.method == 'protoken' and len(gpu_list) != 1:
        print('[abort] protoken runs on exactly one GPU (--gpus 0 with CUDA_VISIBLE_DEVICES)',
              file=sys.stderr)
        return 1
    load, tokenize, weights = make_loader(known.method)
    # Load once on the first GPU (provenance needs the codebook); other GPUs load their own.
    first = f'cuda:{gpu_list[0]}'
    tok0, cbs0 = load(first)
    prov = provenance(known.method, cbs0[0], weights)
    loaded = {first: (tok0, cbs0)}
    def load_cached(device):
        return loaded.pop(device) if device in loaded else load(device)
    pg_score.tokenize_ca = tokenize
    return pg_score.main(rest, baseline={'load': load_cached, 'provenance': prov})


if __name__ == '__main__':
    sys.exit(main())
