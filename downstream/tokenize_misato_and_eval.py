"""EC classification on MISATO from level-1 Ensembits tokens, plus shared helpers.

What it does
------------
  1. `tokenize_misato()` loads the per-pid level-1 token cache
     `data/cached_descriptors/ensembits_<tag>_tokens_misato.npz` (built by
     `_pretokenize_misato_dihedral.py`) and the checkpoint's level-1 codebook
     (M_1 = 2048 vectors of dimension 128). Only if the cache is missing does it
     try to tokenize on the fly (this fallback cannot work for the 256-D model,
     see the note in the function).
  2. Each protein becomes a sequence of token ids; a 1D-CNN head
     (`TokenConv1D`: frozen codebook embedding -> 3 x [Conv1d(k=5) + GELU +
     Dropout] -> masked mean + max pooling -> 2-layer MLP) is trained with
     multi-label BCE on Enzyme Commission labels truncated to depth 1, 2 and 3.
  3. Early stopping on validation BCE; the best-validation weights are scored on
     the test split with top-1 hit, macro / micro AP and macro / micro F1@0.5.
  4. One JSON per depth plus a summary JSON per (variant, split, seed, tag).

The helpers defined here (`load_model_robust`, `tokenize_misato`,
`TokenConv1D`, `collate`, `evaluate`, `metrics`, `train_one`,
`load_split_map`, `tag_of`, path constants) are imported by name by
`tokenize_misato_GO_eval.py`, `tokenize_misato_binding_eval.py` and the
`probe_rmsf_*.py` scripts, so their names must not change.

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool alphabet carry functional
    information?} \\label{sec:downstreams}, \\paragraph{Enzyme Commission,
    Gene Ontology, and binding site/affinity prediction.}
  * `appendix.tex`, \\section{Downstream task experiments}
    \\label{app:downstreams}, \\paragraph{EC Function Prediction} (head,
    loss, five metrics; Table \\label{tab:ec_d1_structure}).
  * `appendix.tex`, \\section{All Experiments} \\label{app:experiments},
    \\subsection{EC Function Prediction} (depths 1-3 x structure / sequence /
    random splits, mean +- std over 10 seeds).
  * `appendix.tex`, \\section{Baselines} \\label{app:baselines}: rows
    "\\tool, P=1" (`--p-inference 1`) and "\\tool, P=full" (default).

Origin
------
Copied from `submission_exp/src/tokenize_misato_and_eval.py`. Changes: imports
and sys.path setup only (`ensembits_featurize` -> `ensembits.featurize`,
`src.tokenizer.vqvae` -> `ensembits.tokenizer.vqvae`; the research-repo entry
`scripts/baselines` was dropped from sys.path because nothing is imported from
it any more), plus comments. Logic, defaults, file names and CLI flags are
unchanged. (The original docstring said the codebook embedding is fine-tuned
with the head; in the code it is frozen, `freeze_emb=True`, and no caller
changes that.)
Fixes 2026-10-16 (FIXED markers): load_model_robust / tokenize_misato refuse an
unfinished or config-less checkpoint unless allow_unfinished (--allow-unfinished)
(code_fix P12); the token cache's provenance sidecar is checked on reuse and written
by the fallback; result JSONs carry a `checkpoint` block (P17).

Pre-run fixes 2026-10-19 (camera_ready/function_reaudit.md; FIXED markers): TokenConv1D
zeroes padded positions (Q23; old: pad id 0 embedded as codebook[0]); the EC items / classes
come from `function_common.run_ec_depth` (shared with the baseline probe: classes from all
labelled training complexes, missing tokens = error, sorted pid order instead of a set);
validation metrics at the selected epoch in r['val']; `token_cache_path`; the token cache's
path + sha256 and the protocol tag in every result JSON; atomic writes; test metrics are
no longer printed.

Function CV 2026-10-19 (authors' decision option A; FIXED markers): `--cv-mode
{sequence,structure} --fold r` scores rotation r of the EC cross-validation
(data/misato_function_cv_v1.json via downstream/function_cv.py: distinct proteins, grouped
label-stratified 5 folds; test = fold r, val = fold r+1 mod 5); the class lists must equal
the CV file's; every JSON records the CV file (path, sha256), mode and fold
(`function_cv`). `--split` is kept for the old fixed-split runs (not the paper rows).

FIXED(function input standardisation, authors' decision 2026-10-19): the frozen level-1
codebook is standardised per dimension over its rows by default (function_common
.standardize_lookup_table, the rule of every lookup feature of every method; helper
`codebook_input_std`, also used by the GO and binding probes); every result JSON records
`input_standardisation` {rule, version, stats_sha256, ...}. `--no-input-standardisation` feeds
the raw codebook (old protocol, output tag + "rawin"). Old: the non-paper option
`--standardize-codebook` of the EC/GO conditioning check (removed; now the default).

Used by the reported model: yes. The paper's EC numbers come from this script,
run by `downstream/_run_model_battery.py` for 3 splits x seeds 0-9 x {P=full,
P=1} with `--depths 1,2,3 --tag stab --warmup 5 --min-epochs 20 --patience 12
--cosine`.

Inputs / outputs
----------------
Read  : `output/<variant>/{config.json,best.pt,stats.npz}` (checkpoint)
        `data/cached_descriptors/ensembits_<tag>_tokens_misato.npz` (tokens)
        `data/annotations/misato_ec_v2.json` (EC numbers per pid; D8, 2026-10-18:
        polyprotein chains labelled by their mature products, see `ec_labels.py`;
        old: `misato_ec.json`)
        `data/misato_splits_v2.json` (sequence / structure / random splits)
        `data/misato_function_cv_v1.json` (+ sidecar; with --cv-mode)
        `data/cached_descriptors/coords_ca8_misato.pkl` (only in the fallback)
Write : `output/ec_conv1d_misato/tok_<tag><suffix>__d{1,2,3}.json` and
        `tok_<tag><suffix>__summary.json` (suffix: [_P<k>inf][_cv_<mode>_f<r> | _<split>]
        [_<tag>][_seed<s>])
        Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): <depth json stem>.preds.npz per EC depth (class
        scores of the selected model, labels, class list, val and test), written next to
        the JSON (downstream/protein_preds.py) and recorded with its sha256 in
        `per_protein_preds` (depth JSON and summary); read only by downstream/protein_level_stats.py.
All paths are relative to REPO_ROOT (the research repo).

How to run
----------
  python downstream/tokenize_misato_and_eval.py \\
      --variant dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
      --device cuda:0 --split structure --seed 0 --depths 1,2,3 \\
      --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine \\
      [--p-inference 1]
  # paper rows (2026-10-19): one rotation of the EC cross-validation instead of --split
  python downstream/tokenize_misato_and_eval.py --variant <variant> --device cuda:0 \\
      --cv-mode structure --fold 0 --seed 0 --depths 1,2,3 --tag stab --warmup 5 \\
      --min-epochs 20 --patience 12 --cosine [--p-inference 1]
"""
from __future__ import annotations

import os
# Keep CPU thread pools small (several of these jobs run in parallel).
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

from ensembits.featurize import load_model, tokenize_ensemble, Ensembits, checkpoint_dihedral_convention, load_bb_for_pid  # noqa: E402
from ensembits.featurize import (check_checkpoint_finished, checkpoint_provenance,  # noqa: E402
                                 check_provenance_sidecar, write_provenance_sidecar)


# ---------- checkpoint loading ----------

def load_model_robust(variant_dir: Path, device: str,
                      allow_unfinished: bool = False) -> Ensembits:
    """Load a checkpoint, working around a mislabelled config in old VQ runs.

    For the reported RVQ model this simply calls `ensembits.featurize.load_model`.

    Some VQ-4096 runs were trained with --tokenizer vq but the trainer wrote
    `tokenizer_class: RVQVAETokenizer` to config.json. Detect from state-dict
    keys and rewrite config so load_model picks the right class."""
    # FIXED(code_fix P12, 2026-10-16): old: json.load(config.json) without checks. New: a
    #   folder without config.json or with an unfinished training run ("status":
    #   "running") raises UnfinishedCheckpointError unless allow_unfinished=True.
    variant_dir = Path(variant_dir)
    cfg = check_checkpoint_finished(variant_dir, allow_unfinished=allow_unfinished)
    state = torch.load(variant_dir / 'best.pt', map_location='cpu',
                        weights_only=True)
    has_vq_levels = any(k.startswith('vq_levels.') for k in state)
    has_plain_vq = any(k.startswith('vq.') for k in state)
    if has_plain_vq and not has_vq_levels:
        # Single-codebook VQ checkpoint (ablation runs only; not the reported model).
        # Patch config in-memory and bypass load_model's class chooser
        cfg = dict(cfg)
        cfg['tokenizer_class'] = 'VQVAETokenizer'
        cfg['codebook_size'] = cfg['codebook_sizes'][0]
        # Save patched config beside the file with a different name and load
        # via a tiny direct loader (avoids reading the on-disk config).
        from ensembits.tokenizer.vqvae import VQVAETokenizer
        model = VQVAETokenizer(
            input_dim=cfg['input_dim'],
            num_prototypes=cfg['num_prototypes'],
            codebook_size=cfg['codebook_size'],
            hidden_dim=cfg['hidden_dim'],
            latent_dim=cfg['latent_dim'],
            commitment_cost=cfg['commitment_cost'],
            encoder_type=cfg['encoder_type'],
            loss_type=cfg['loss_type'],
            vq_update=cfg['vq_update'],
            n_queries=cfg['n_queries'],
            n_heads=cfg['n_heads'],
            n_encoder_layers=cfg['n_encoder_layers'],
            n_decoder_layers=cfg['n_decoder_layers'],
        )
        model.load_state_dict(state)
        model = model.to(device).eval()
        stats = np.load(variant_dir / 'stats.npz')
        return Ensembits(model=model,
                         mean=stats['mean'].astype(np.float32),
                         std=stats['std'].astype(np.float32),
                         config=cfg, device=device, version='v1',
                         # code_fix §1: dihedral convention the checkpoint was trained with.
                         dihedral_convention=checkpoint_dihedral_convention(cfg),
                         chain_break_cutoff=cfg.get('chain_break_cutoff', 'auto'))
    # Otherwise fall back to the standard loader
    return load_model(variant_dir, device=device, allow_unfinished=allow_unfinished)

# ---------- data paths (all under REPO_ROOT) ----------
# FIXED(benchmark_audit D8, 2026-10-18): old = data/annotations/misato_ec.json (per-PDB SIFTS
#   union: polyprotein chains carried every EC number of the precursor). New = the v2 labels
#   (ec_labels.EC_LABELS = data/annotations/misato_ec_v2.json, data_prep/build_misato_ec_labels.py),
#   loaded with ec_labels.load_ec_labels (refuses a file without the v2 sidecar); every result
#   JSON records "ec_labels" {path, sha256, schema}.
sys.path.insert(0, str(Path(__file__).resolve().parent))   # downstream/ (ec_labels)
from ec_labels import EC_LABELS, load_ec_labels  # noqa: E402
PAIRS = ROOT / 'data' / 'cached_descriptors' / 'aa2ensembits_pairs_misato_v2.pkl'
COORDS = ROOT / 'data' / 'cached_descriptors' / 'coords_ca8_misato.pkl'
CACHE = ROOT / 'data' / 'cached_descriptors'
OUT_DIR = ROOT / 'output' / 'ec_conv1d_misato'
# PAIRS is not used in this file (kept from the original).


def tag_of(variant: str) -> str:
    """Cache / output tag of a checkpoint folder name (strips `final_model`)."""
    return variant.replace('final_model_', '').replace('final_model', '') or variant


def token_cache_path(variant_dir: Path, p_inference: int | None = None) -> Path:
    """The MISATO level-1 token cache of a checkpoint folder (same name as tokenize_misato).

    New 2026-10-19 (function_reaudit pre-run fix 2): the probes record its path + sha256."""
    tag = tag_of(Path(variant_dir).name)
    if p_inference is not None:
        tag = f'{tag}_P{p_inference}inf'
    return CACHE / f'ensembits_{tag}_tokens_misato.npz'


def tokenize_misato(variant_dir: Path, device: str, force: bool = False,
                     p_inference: int | None = None,
                     allow_old_parser: bool = False,
                     allow_unfinished: bool = False,
                     force_provenance: bool = False) -> tuple:
    """Tokenize all misato proteins using the variant's tokenizer.

    Returns: (tokens dict[pid] -> (L,) int64, codebook (M, d) tensor, M).

    If p_inference is set, restricts inference to the first p_inference frames
    of each protein's ensemble (bb_8 of misato_real_bb for the 256-D models; was
    documented as ca_5, docs fixed 2026-10-16). For fused-mode tokenizers (fixed P),
    replicates the single frame to fill the trained P.
    """
    # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv.
    # Token caches are per checkpoint (tag below), so the retrained (iupac) model
    # writes new caches; old ones stay valid for old checkpoints.
    tag = tag_of(variant_dir.name)
    if p_inference is not None:
        tag = f'{tag}_P{p_inference}inf'
    cache_path = CACHE / f'ensembits_{tag}_tokens_misato.npz'
    coords_path = COORDS

    ens = load_model_robust(variant_dir, device=device, allow_unfinished=allow_unfinished)
    prov = checkpoint_provenance(variant_dir)
    M = ens.config.get('codebook_size') or ens.config['codebook_sizes'][0]
    # Pull the L0 codebook (level-1 code vectors, (M_1, latent_dim) = (2048, 128)
    # for the reported model); the downstream heads embed tokens with it.
    # Paper: 1_method.tex \paragraph{Residual quantizer.} — only level-1 tokens are used downstream.
    state = ens.model.state_dict()
    if 'vq_levels.0.codebook.weight' in state:
        cb = state['vq_levels.0.codebook.weight'].detach().cpu().clone()
    else:
        cb = state['vq.codebook.weight'].detach().cpu().clone()

    # Normal path: read the cache built by _pretokenize_misato_dihedral.py.
    if cache_path.exists() and not force:
        # FIXED(code_fix P17, 2026-10-16): old: the cache was read by name only. New: its
        #   <cache>.provenance.json must name this checkpoint (best.pt sha256 + code_sha256);
        #   a mismatch raises ProvenanceMismatchError (force_provenance=True accepts it).
        check_provenance_sidecar(cache_path, prov, force=force_provenance)
        d = np.load(cache_path)
        tokens = {pid: d[pid].astype(np.int64) for pid in d.files}
        print(f'[tokenize] loaded {len(tokens)} cached tokens from '
              f'{cache_path.name}', flush=True)
        return tokens, cb, M

    # Fallback: tokenize on the fly.
    # FIXED(code_fix §3 / T2.4, 2026-10-15): 256-D models read the real N/CA/C of
    #   the 8 downstream frames (misato_real_bb bb_8, the input of
    #   _pretokenize_misato_dihedral.py), so the fallback gives the same tokens as
    #   the cache builder. Before, it passed CA-only coordinates and every protein
    #   failed. CA-only models (old 3Di descriptors, 'fused' mode) still read
    #   coords_ca8_misato.pkl.
    uses_bb = (ens.config.get('descriptor_mode') == 'esm3desc_dihedral'
               or ens.config.get('input_dim') == 256)
    tokens: dict = {}
    skipped = 0
    t0 = time.time()
    if uses_bb:
        # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
        from ensembits.data_versions import require_misato_parser
        require_misato_parser(CACHE / 'misato_real_bb', allow_old=allow_old_parser)
        coords = None
        pids = sorted(p.stem for p in (CACHE / 'misato_real_bb').glob('*.npz'))
    else:
        coords = pickle.load(open(coords_path, 'rb'))
        pids = sorted(coords.keys())
    print(f'[tokenize] {len(pids)} misato proteins under '
          f'{variant_dir.name}'
          + (f' (P_inference={p_inference})' if p_inference else ''),
          flush=True)

    mode = ens.config.get('descriptor_mode', 'fused')
    P_trained = ens.config.get('num_prototypes', 5)
    for i, pid in enumerate(pids):
        try:
            if uses_bb:
                bb = load_bb_for_pid(pid, 'misato', 8)                # (8, L, 3, 3) N/CA/C
                if bb is None:
                    raise FileNotFoundError(pid)
                if p_inference is not None:
                    bb = bb[:p_inference]                             # first frames (P=1: frame 0)
                tokens[pid] = tokenize_ensemble(ens, None, bb_all=bb).astype(np.int64)
                if (i + 1) % 500 == 0:
                    print(f'  {i+1}/{len(pids)} ({time.time()-t0:.0f}s)', flush=True)
                continue
            ca5 = np.asarray(coords[pid], dtype=np.float32)
            if p_inference is not None:
                # Take the first p_inference frames; for fused mode (fixed P),
                # replicate frame 0 to fill the trained P.
                if mode == 'fused':
                    ca_in = np.repeat(ca5[0:1], P_trained, axis=0)
                else:
                    ca_in = ca5[:p_inference]
            else:
                ca_in = ca5
            tok = tokenize_ensemble(ens, ca_in)
        except Exception:
            skipped += 1
            continue
        tokens[pid] = tok.astype(np.int64)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(pids)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'[tokenize] {len(tokens)} ok, {skipped} skipped '
          f'({time.time()-t0:.0f}s)', flush=True)
    # Don't write an empty (or near-empty) cache — would clobber a good
    # cache built by an external helper (e.g. _pretokenize_misato_esm3.py
    # for variants whose descriptor differs from the live-compute path).
    if len(tokens) >= max(1, len(pids) // 2):
        np.savez(cache_path, **{pid: t for pid, t in tokens.items()})
        # FIXED(code_fix P17, 2026-10-16): new (old: none): provenance sidecar next to the cache.
        write_provenance_sidecar(cache_path, prov, p_inference=p_inference,
                                 n_pids=len(tokens), n_pids_total=len(pids),
                                 written_by='downstream/tokenize_misato_and_eval.py (fallback)')
        print(f'  wrote {cache_path.name}', flush=True)
    else:
        print(f'  refusing to write near-empty cache '
              f'({len(tokens)}/{len(pids)} pids ok); leaving existing cache '
              f'on disk if any', flush=True)
    return tokens, cb, M


# ---------- conv1d head with codebook embedding init ----------

class TokenConv1D(nn.Module):
    """Per-protein multi-label head on token ids (used for EC and GO).

    Token ids -> embedding (frozen level-1 codebook by default, or random and
    trainable with `random_init`) -> `depth` x [Conv1d(kernel 5, `hidden`) +
    GELU + Dropout] -> masked mean and max pooling over residues -> MLP ->
    per-class logits.

    Paper: appendix.tex \\label{app:downstreams} \\paragraph{EC Function
    Prediction} — "a stack of three Conv1D layers, masked mean+max pooling
    across residues, and a small MLP head producing per-class logits".
    """
    def __init__(self, codebook: torch.Tensor, n_classes: int,
                 hidden: int = 128, depth: int = 3, kernel: int = 5,
                 dropout: float = 0.1, freeze_emb: bool = True,
                 random_init: bool = False, rand_emb_dim: int | None = None,
                 zero_pad: bool = True):
        super().__init__()
        # FIXED(function_reaudit Q23, 2026-10-19): padded positions (mask False) enter the
        #   convolution as zero vectors, like the zero-padded features of the baselines.
        #   Old (zero_pad=False): the pad id 0 was embedded, so every padded position carried
        #   codebook[0] and the kernel-5 convolution mixed it into the last real residues.
        self.zero_pad = zero_pad
        M, d_emb = codebook.shape
        if random_init:
            d_emb = rand_emb_dim or d_emb
            self.embedding = nn.Embedding(M, d_emb)
            nn.init.normal_(self.embedding.weight, std=0.02)
        else:
            # Token embedding = the tokenizer's level-1 code vectors, frozen by
            # default (no caller passes freeze_emb=False).
            # Paper: appendix.tex \label{app:downstreams} — "Per-residue features are obtained as token embeddings".
            self.embedding = nn.Embedding.from_pretrained(
                codebook.float(), freeze=freeze_emb, padding_idx=None)
        layers = []
        prev = d_emb
        for _ in range(depth):
            layers.append(nn.Conv1d(prev, hidden, kernel, padding=kernel // 2))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, tokens, mask):
        # Padding positions (mask False) are zero vectors (Q23) and excluded from both poolings.
        x = self.embedding(tokens)               # (B, L, d_emb)
        if self.zero_pad:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        m = mask.unsqueeze(-1).float()
        avg = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        very_neg = torch.full_like(x, float('-inf'))
        x_max = torch.where(mask.unsqueeze(-1), x, very_neg).max(1).values
        x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
        return self.head(torch.cat([avg, x_max], dim=-1))


# ---------- splits & labels (same convention as conv1d_misato) ----------

# FIXED(benchmark_audit D3, 2026-10-18): old = data/misato_splits.json (UniProt / first-CATH-H
#   grouping; leaky). New = ensembits.paths.MISATO_SPLITS (data/misato_splits_v2.json,
#   data_prep/build_misato_splits_v2.py), read through data_versions.load_misato_splits (v2
#   schema required); every result JSON records misato_splits_record() (path + sha256).
from ensembits.paths import MISATO_SPLITS  # noqa: E402
from ensembits.data_versions import load_misato_splits, misato_splits_record  # noqa: E402


def load_split_map(split_name: str = 'sequence') -> dict:
    """Map UPPERCASE pid -> 'train' / 'val' / 'test' for one MISATO split.

    `ensembits.paths.MISATO_SPLITS` (data/misato_splits_v2.json) holds the 'sequence',
    'structure', 'random' (and an unused 'ligand_disjoint') splits.
    Paper: 2_results.tex \\label{sec:downstreams} — sequence, structure and random
    splits; appendix.tex \\label{app:experiments} describes each split.
    Open(audit D.6): the paper text of the split rule must describe v2 (MMseqs2 30 %
    identity clusters; CATH H over all chains, no-CATH via sequence clusters). Text pass item.
    """
    s = load_misato_splits(MISATO_SPLITS)
    if split_name not in s:
        raise ValueError(f'unknown split {split_name!r} (have {list(s)})')
    out = {}
    for sp in ('train', 'val', 'test'):
        for pid in s[split_name].get(sp, []):
            out[pid.upper()] = sp
    return out


# FIXED(benchmark_audit D8, 2026-10-18): old = the first `depth` fields whatever they were,
#   so a partial EC number became a class of its own ('3.1.-.-' at depth 3 -> '3.1.-',
#   '3.-.-.-' at depth 2 -> '3.-'). New = a partial EC number contributes at depth d only if
#   its first d fields are all specified; no class ever contains '-'. Identical copy in
#   baselines/ec_classify_conv1d_misato.py.
def truncate_ec(ec, depth):
    """EC number truncated to `depth` fields, or None.

    '3.4.21.4' at depth 2 -> '3.4'. A partial (wildcard) EC number counts only at depths
    whose fields are all specified: '3.1.-.-' -> '3' (d1), '3.1' (d2), None (d3);
    '3.-.-.-' -> '3' (d1), None (d2, d3). None also if the number has fewer than `depth`
    fields. Preliminary numbers ('1.1.1.n12') are specified fields.
    """
    parts = str(ec).strip().split('.')
    if len(parts) < depth:
        return None
    head = parts[:depth]
    if any(p in ('-', '') for p in head):
        return None
    return '.'.join(head)


def build_label_sets(ec_raw: dict, depth: int) -> dict:
    """pid -> set of EC numbers truncated to `depth` (pids with none dropped).

    Paper: appendix.tex \\label{app:downstreams} \\paragraph{EC Function
    Prediction} — EC numbers collapsed to depth 1 (headline), 2 and 3.
    """
    out: dict = {}
    for pid, lst in ec_raw.items():
        if not lst:
            continue
        s = set()
        for e in lst:
            t = truncate_ec(e, depth)
            if t is not None:
                s.add(t)
        if s:
            out[pid] = s
    return out


# ---------- training / metrics (identical to conv1d_misato) ----------

def collate(batch, pad_id: int = 0):
    """Pad token sequences to the batch max length; return (ids, mask, labels)."""
    toks = [b['x'] for b in batch]
    Y = torch.stack([torch.as_tensor(b['y'], dtype=torch.float32) for b in batch], 0)
    L_max = max(t.shape[0] for t in toks)
    X = torch.full((len(toks), L_max), pad_id, dtype=torch.long)
    M = torch.zeros((len(toks), L_max), dtype=torch.bool)
    for i, t in enumerate(toks):
        L = t.shape[0]
        X[i, :L] = torch.as_tensor(t, dtype=torch.long)
        M[i, :L] = True
    return X, M, Y


def evaluate(model, items, batch_size, device):
    """Sigmoid class probabilities and labels for every item (no grad)."""
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for s in range(0, len(items), batch_size):
            batch = items[s:s + batch_size]
            X, M, Y = collate(batch)
            X, M = X.to(device), M.to(device)
            p = torch.sigmoid(model(X, M)).cpu().numpy()
            all_logits.append(p); all_y.append(Y.numpy())
    return np.concatenate(all_logits, 0), np.concatenate(all_y, 0)


# FIXED(benchmark_audit D7 / Q-metrics, 2026-10-18): `metrics` moved to
#   downstream/multilabel_metrics.py (one copy for EC and GO, Ensembits and baselines;
#   old: two identical-by-intent copies here and in baselines/ec_classify_conv1d_misato.py). Same five metrics
#   as before; new: protein-centric Fmax (CAFA, thresholds 0.00-1.00 step 0.01, over all
#   classes of the task) in `fmax` / `fmax_threshold`.
from multilabel_metrics import metrics  # noqa: E402,F401
# Shared function-probe rules (function_reaudit pre-run fixes, 2026-10-19).
from function_common import (run_ec_depth, protocol_record, token_cache_record,  # noqa: E402
                             atomic_write_json, InputStandardisation, standardize_lookup_table,
                             add_input_std_arg, input_std_tag)
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import PREDS_KEY, items_preds, write_function_preds  # noqa: E402
# EC / GO cross-validation (authors' decision option A, 2026-10-19).
from function_cv import (add_cv_args, check_cv_args, load_function_cv, cv_split_map,  # noqa: E402
                         cv_result_block, check_classes, cv_tag)


# FIXED(function input standardisation, authors' decision 2026-10-19): new; shared by the
#   Ensembits EC, GO and binding / affinity probes.
def codebook_input_std(codebook, variant: str, enabled: bool = True,
                       random_init: bool = False, rand_emb_dim=None) -> tuple:
    """(codebook for the head, InputStandardisation) of an Ensembits function probe.

    Default: the frozen level-1 codebook standardised per dimension over its rows
    (function_common.standardize_lookup_table, the same rule as every lookup baseline).
    enabled=False (--no-input-standardisation): the raw codebook. random_init: the head learns
    its own embedding (no fixed input; codebook unchanged, rule 'trainable').
    """
    method = f'Ensembits {variant}'
    if not enabled:
        return codebook, InputStandardisation.disabled(method)
    if random_init:
        d = int(rand_emb_dim or codebook.shape[1])
        return codebook, InputStandardisation(method, [{'name': method, 'kind': 'trainable',
                                                        'dims': [0, d]}])
    cb, rec = standardize_lookup_table(codebook.detach().cpu().numpy(),
                                       'Ensembits level-1 codebook')
    return torch.from_numpy(cb), InputStandardisation.lookup(method, cb.shape[1], [rec])


def _make_sched(opt, warmup, total, cosine):
    """Per-epoch LR multiplier: linear warmup then optional cosine decay to 0.
    No-op when warmup=0 & cosine=False (reproduces the original protocol).

    The paper's numbers use warmup=5 epochs then cosine over `epochs` (the
    battery's "stab" recipe)."""
    import math
    def lr_lambda(ep):
        if warmup > 0 and ep < warmup:
            return (ep + 1) / warmup
        if cosine and total > warmup:
            prog = (ep - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# FIXED(require_fix #19, 2026-10-15): appendix "Probe training" paragraph lists the recipe.
def train_one(items_tr, items_va, items_te, codebook, n_classes, device,
              epochs=40, batch_size=32, lr=1e-3, weight_decay=1e-4,
              patience=8, random_init: bool = False,
              rand_emb_dim: int | None = None, seed: int = 0,
              min_epochs: int = 0, warmup: int = 0, cosine: bool = False,
              return_preds: bool = False) -> dict:
    """Train one TokenConv1D with multi-label BCE and score it on the test split.

    AdamW (lr 1e-3, weight decay 1e-4), batch 32, up to `epochs` epochs; the
    seed fixes both the weight init and the shuffling order. Early stopping on
    validation BCE (improvement > 1e-4) after at least `min_epochs` epochs and
    `patience` epochs without improvement; the best-validation weights are
    evaluated on test.
    return_preds: also return the selected model's val / test class scores and labels per
    item under the private key '_preds' (popped by the caller for the per-protein sidecar,
    final_fix_list A4 (a)).
    """
    torch.manual_seed(seed)
    model = TokenConv1D(codebook=codebook, n_classes=n_classes,
                         random_init=random_init,
                         rand_emb_dim=rand_emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _make_sched(opt, warmup, epochs, cosine)
    g = torch.Generator(device='cpu').manual_seed(seed)
    best_va = float('inf'); best_state = None; no_imp = 0
    n_tr = len(items_tr)
    for ep in range(epochs):
        # One training epoch over a seeded random permutation of the train set.
        model.train()
        perm = torch.randperm(n_tr, generator=g).tolist()
        for s in range(0, n_tr, batch_size):
            batch = [items_tr[i] for i in perm[s:s + batch_size]]
            X, M, Y = collate(batch)
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(X, M), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        va_loss = 0.0; n = 0
        with torch.no_grad():
            for s in range(0, len(items_va), batch_size):
                batch = items_va[s:s + batch_size]
                X, M, Y = collate(batch)
                X, M, Y = X.to(device), M.to(device), Y.to(device)
                l = F.binary_cross_entropy_with_logits(model(X, M), Y)
                va_loss += l.item() * X.shape[0]; n += X.shape[0]
        va_loss /= max(n, 1)
        # Keep the best-validation weights; stop after `patience` epochs
        # without improvement (never before `min_epochs`).
        sched.step()
        if va_loss < best_va - 1e-4:
            best_va = va_loss; no_imp = 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if ep + 1 >= min_epochs and no_imp >= patience:
            break
    # Score the best-validation model on the test split.
    if best_state is not None:
        model.load_state_dict(best_state)
    p, y = evaluate(model, items_te, batch_size, device)
    m = metrics(p, y)
    m.update({'best_val_loss': best_va, 'n_train': len(items_tr),
              'n_val': len(items_va), 'n_test': len(items_te),
              'n_classes': int(n_classes), 'epochs_run': ep + 1})
    # FIXED(function_reaudit pre-run fix 5, 2026-10-19): new: the same metrics on the
    #   validation split with the selected (best validation BCE) weights, in m['val'].
    pv, yv = evaluate(model, items_va, batch_size, device)
    m['val'] = {**metrics(pv, yv), 'best_val_loss': best_va, 'n_val': len(items_va)}
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-item scores of the selected model.
    if return_preds:
        m['_preds'] = {'val': items_preds(items_va, pv, yv), 'test': items_preds(items_te, p, y)}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True,
                    help='Tokenizer dir name in output/ (e.g. final_model_v2)')
    ap.add_argument('--depths', default='1,2,3')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--force-tokenize', action='store_true')
    ap.add_argument('--p-inference', type=int, default=None,
                    help='Restrict inference to first P frames of each ensemble. '
                         'For fused tokenizers, replicates frame 0 to match trained P.')
    ap.add_argument('--random-init-embedding', action='store_true',
                    help='Init token embedding randomly (ignore codebook). The '
                         'embedding is still learnable; this isolates the discrete '
                         'token clustering from the codebook geometry.')
    ap.add_argument('--rand-emb-dim', type=int, default=None,
                    help='Embedding dim for random init. Default: matches codebook dim.')
    ap.add_argument('--split', default='sequence',
                    choices=['sequence', 'structure', 'random'],
                    help='fixed v2 split (old protocol; the paper rows use --cv-mode / --fold)')
    # FIXED(function CV, authors' decision option A, 2026-10-19): new --cv-mode / --fold:
    #   distinct proteins (one representative complex per UniProt accession set) and grouped,
    #   label-stratified 5-fold CV (data/misato_function_cv_v1.json, downstream/function_cv.py);
    #   test = fold r, val = fold r+1 mod 5, train = the rest. Old (still available): --split.
    add_cv_args(ap)
    # Paper: appendix.tex \label{app:experiments} — mean +- std over 10 seeds
    # (the battery runs --seed 0..9).
    ap.add_argument('--seed', type=int, default=0)
    # Downstream-head stabilization (opt-in; defaults reproduce the paper protocol)
    # NOTE: the defaults reproduce the original (pre-camera-ready) protocol. The
    # numbers reported in the paper were run with --warmup 5 --min-epochs 20
    # --patience 12 --cosine --tag stab (RECIPE in _run_model_battery.py).
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--tag', default='')
    # FIXED(function input standardisation, 2026-10-19): the codebook is standardised over its
    #   rows by default; --no-input-standardisation feeds it raw (old protocol, tag + rawin).
    #   Old: the non-paper opt-in --standardize-codebook (removed).
    add_input_std_arg(ap)
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='accept a checkpoint whose training run has not finished (code_fix P12)')
    args = ap.parse_args()
    is_cv = check_cv_args(ap, args)
    input_std_tag(args)
    stab = dict(min_epochs=args.min_epochs, warmup=args.warmup, cosine=args.cosine)

    # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variant_dir = ROOT / 'output' / args.variant
    if not variant_dir.exists():
        print(f'ERR: {variant_dir} does not exist', file=sys.stderr); return 2

    feats, codebook, M = tokenize_misato(variant_dir, device=args.device,
                                            force=args.force_tokenize,
                                            p_inference=args.p_inference,
                                            allow_unfinished=args.allow_unfinished)
    # FIXED(function input standardisation, 2026-10-19): codebook standardised over its rows
    #   (default; the rule of every lookup feature of every method).
    codebook, in_std = codebook_input_std(codebook, args.variant,
                                          enabled=not args.no_input_standardisation,
                                          random_init=args.random_init_embedding,
                                          rand_emb_dim=args.rand_emb_dim)
    print(f'[input-std] {in_std.base_record()["rule"]}', flush=True)
    in_dim = codebook.shape[1]
    print(f'[setup] variant={args.variant}  M={M}  d_emb={in_dim}  '
          f'n_proteins={len(feats)}', flush=True)
    # FIXED(code_fix P17, 2026-10-16): every result JSON carries the checkpoint block.
    ckpt_block = checkpoint_provenance(variant_dir)

    # Load the split and the raw EC labels; build the output-file suffix.
    split_rec = misato_splits_record(MISATO_SPLITS)   # FIXED(benchmark_audit D3, 2026-10-18)
    cv_block = {}
    if is_cv:
        # FIXED(function CV, 2026-10-19): rotation `fold` of the EC CV (representatives only).
        cv, cv_rec = load_function_cv()
        split_map = cv_split_map(cv, 'ec', args.cv_mode, args.fold)
        cv_block = cv_result_block(cv_rec, cv, 'ec', args.cv_mode, args.fold)
    else:
        split_map = load_split_map(args.split)
    ec_raw, ec_rec = load_ec_labels(EC_LABELS)        # FIXED(benchmark_audit D8, 2026-10-18)
    # FIXED(function_reaudit pre-run fix 2, 2026-10-19): path + sha256 of the token cache
    #   (its sidecar was checked by tokenize_misato) and the protocol tag in every JSON.
    feat_recs = [token_cache_record(token_cache_path(variant_dir, args.p_inference))]
    summary = {'variant': args.variant, 'M': int(M), 'd_emb': int(in_dim),
               'p_inference': args.p_inference,
               'split': f'cv/{args.cv_mode}/fold{args.fold}' if is_cv else args.split,
               'cv_mode': args.cv_mode, 'cv_fold': args.fold, **cv_block,
               'random_init_embedding': args.random_init_embedding,
               'rand_emb_dim': args.rand_emb_dim, 'by_depth': {},
               'checkpoint': ckpt_block, **split_rec, 'ec_labels': ec_rec,
               **protocol_record(feat_recs, input_std=in_std.run_record())}
    suffix = ''
    if args.p_inference is not None:
        suffix += f'_P{args.p_inference}inf'
    if args.random_init_embedding:
        suffix += f'_randEmb{args.rand_emb_dim or in_dim}'
    if is_cv:                                          # function CV (2026-10-19)
        suffix += f'_{cv_tag(args.cv_mode, args.fold)}'
    elif args.split != 'sequence':
        suffix += f'_{args.split}'
    if args.tag:
        suffix += f'_{args.tag}'
    if args.seed != 0:
        suffix += f'_seed{args.seed}'
    summary['seed'] = args.seed
    tag = f'tok_{tag_of(args.variant)}{suffix}'

    # One independent run per EC depth.
    # FIXED(function_reaudit pre-run fixes 2 / 6, 2026-10-19): function_common.run_ec_depth
    #   (shared with the baseline EC probe): classes from every labelled training complex of
    #   the split (old: training complexes with tokens), every labelled complex must have
    #   tokens, items in sorted pid order (old: a set, so the order varied with
    #   PYTHONHASHSEED), class list, item hashes and validation metrics stored.
    for depth in [int(x) for x in args.depths.split(',')]:
        r = run_ec_depth(depth, build_label_sets(ec_raw, depth), split_map, feats,
                         f'Ensembits {args.variant}',
                         lambda b_tr, b_va, b_te, n_cls: train_one(
                             b_tr, b_va, b_te, codebook, n_cls, args.device,
                             epochs=args.epochs, batch_size=args.batch_size,
                             random_init=args.random_init_embedding,
                             rand_emb_dim=args.rand_emb_dim, seed=args.seed,
                             patience=args.patience, return_preds=True, **stab),
                         input_std=in_std)            # 2026-10-19: r['input_standardisation']
        preds = r.pop('_preds', None)                 # A4 (a): sidecar below
        summary['by_depth'][depth] = r
        if 'error' in r:
            continue
        if is_cv:                                      # classes must equal the CV file's
            check_classes(cv, 'ec', args.cv_mode, args.fold, r['ec_classes'], depth)
        r = {**r, 'checkpoint': ckpt_block, **split_rec, **cv_block}     # FIXED(code_fix P17, 2026-10-16): new key, per-depth JSON only (old: none); D3: split record
        r['ec_labels'] = ec_rec                       # FIXED(benchmark_audit D8, 2026-10-18): label record
        r.update(protocol_record(feat_recs, input_std=r['input_standardisation']))   # 2026-10-19
        out_path = OUT_DIR / f'{tag}__d{depth}.json'
        # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar <out>.preds.npz
        #   (class scores of the selected model + labels, val and test, class list) recorded in
        #   `per_protein_preds` of the depth JSON and of the summary's by_depth entry.
        r[PREDS_KEY] = write_function_preds(
            out_path, preds, r, task='EC', classes=r['ec_classes'],
            method=f'Ensembits {args.variant}', p_inference=args.p_inference, seed=args.seed,
            split=summary['split'], cv_mode=args.cv_mode, cv_fold=args.fold,
            checkpoint_best_pt_sha256=(ckpt_block or {}).get('best_pt_sha256'))
        summary['by_depth'][depth][PREDS_KEY] = r[PREDS_KEY]
        atomic_write_json(out_path, r)
        # (test metrics are not printed; 2026-10-19)
        print(f'  d{depth}: [ep={r["epochs_run"]}, {r["runtime_s"]}s, n_test={r["n_test"]}] '
              f'wrote {out_path.name}', flush=True)

    summary_path = OUT_DIR / f'{tag}__summary.json'
    atomic_write_json(summary_path, summary)
    print(f'[done] {summary_path}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
