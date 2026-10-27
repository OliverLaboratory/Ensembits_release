"""Binding-site and binding-affinity prediction on MISATO from level-1 Ensembits tokens.

What it does
------------
  1. Loads the level-1 token cache and the level-1 codebook through
     `tokenize_misato()` (from `tokenize_misato_and_eval.py`; the cache is built
     by `_pretokenize_misato_dihedral.py`).
  2. Binding site (per residue, binary): `BindingConv1D` = frozen codebook
     embedding -> 3 x [Conv1d(k=5) + GELU + Dropout] -> Linear -> one logit per
     residue (no pooling); masked BCE; test AUROC and AP pooled over all test
     residues.
  3. Binding affinity (per protein, regression): `AffinityConv1D` = same conv
     stack -> masked mean + max pooling, optionally concatenated with a
     projection of the 167-bit MACCS fingerprint of the ligand
     (`--use-ligand-maccs`) -> MLP -> scalar; MSE loss; test R^2, Spearman and
     MSE.
  4. Both tasks run on each requested split (sequence / structure / random);
     results for all splits go into one JSON per (variant, P mode, ligand flag,
     tag, seed). An existing JSON is reused and only missing (split, task)
     entries are run.

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool alphabet carry functional
    information?} \\label{sec:downstreams}: "per-residue binding-site
    classification, and a complementary binding-affinity regression".
  * `appendix.tex`, \\section{Downstream task experiments}
    \\label{app:downstreams}, \\paragraph{Binding Site} (per-residue 1D-CNN
    head, no pooling, masked BCE, AUROC / AP pooled over test residues;
    Table \\label{tab:binding_structure}) and \\paragraph{Binding Affinity}
    (-log Kd/Ki, MACCS-167 ligand fingerprint + mean+max-pooled protein
    representation -> small MLP; R^2 / Spearman / MSE;
    Table \\label{tab:affinity_structure}).
  * `appendix.tex`, \\section{All Experiments} \\label{app:experiments},
    \\subsection{Binding Site} and \\subsection{Binding Affinity} (three splits,
    mean +- std over 10 seeds).

Origin
------
Copied from `submission_exp/src/tokenize_misato_binding_eval.py`. Changes:
sys.path setup only (ROOT now comes from `ensembits.paths`; the script's own
folder is put on sys.path so the sibling `tokenize_misato_and_eval` import
resolves), plus comments. Logic, defaults, file names and CLI flags are
unchanged.
Fixes 2026-10-16 (FIXED markers): `checkpoint` block in the result JSON, checked
when an existing JSON is completed (--force) (code_fix P17); --overwrite for the
seed-rescue re-run (P2); --allow-unfinished (P12).
Fix 2026-10-18 (benchmark_audit D4): binding-site labels v2 (PDBbind pocket aligned
to MISATO residues), sha256 recorded in the result JSON as `binding_labels_*`;
an existing JSON whose binding results lack it is not merged into.

Pre-run fixes 2026-10-19 (camera_ready/function_reaudit.md; FIXED markers): padded positions
are zero vectors in BindingConv1D / AffinityConv1D (Q23); binding / affinity items from
function_common (length mismatch = error, drops counted, every affinity mode on the
complexes with a finite label, a MACCS fingerprint and >= 10 residues); without
--use-ligand-maccs the run is the D5 protein-only control; affinity labels / MACCS read with
their builder sidecars, sha256 recorded; validation metrics in r['val'] (binding: + per-protein
AUROC); token-cache record and protocol tag; an existing JSON of the old protocol is not merged
into; atomic write; test metrics not printed.

FIXED(function input standardisation, authors' decision 2026-10-19): the frozen codebook is
standardised per dimension over its rows by default (tokenize_misato_and_eval
.codebook_input_std, the rule of every lookup feature of every method); the MACCS ligand input
is unchanged (binary, function_common.LIGAND_RULE). Every (split, task) block and the JSON
record `input_standardisation`; an existing JSON of another rule is not merged into.
`--no-input-standardisation` = raw codebook (old protocol, tag + "rawin").

Used by the reported model: yes. `downstream/_run_model_battery.py` runs it with
`--task both --splits sequence,structure,random --use-ligand-maccs` for seeds
0-9 x {P=full, P=1} and `--tag stab --warmup 5 --min-epochs 20 --patience 12
--cosine`.

Inputs / outputs
----------------
Read  : `output/<variant>/` (checkpoint), the token cache
        `data/cached_descriptors/ensembits_<tag>_tokens_misato.npz`,
        `data/annotations/misato_binding_site_v2.npz` (per-residue 0/1 labels,
        PDBbind pocket aligned to MISATO residues; built by
        `data_prep/build_misato_binding_site_labels.py`, D4 2026-10-18; the old
        `data/cached_descriptors/misato_binding_site.npz` is no longer read),
        `data/annotations/misato_affinity.json` (-log Kd/Ki per pid; + sidecar,
        data_prep/build_misato_affinity_labels.py),
        `data/cached_descriptors/misato_ligand_maccs.npz` (167-bit MACCS per
        pid; + sidecar, data_prep/build_misato_ligand_maccs.py; ported from the
        research repo 2026-10-19),
        `data/misato_splits_v2.json`
Write : `output/binding_misato/<tag><suffix>.json` (suffix: `_P1inf`,
        `_randEmb<d>`, `_lig`, `_<tag>`, `_seed<n>`)
        Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): <json stem>.<split>_<task>.preds.npz per
        block (binding: per-residue probabilities and labels; affinity: per-complex prediction
        and label; val and test), written next to
        the JSON (downstream/protein_preds.py) and recorded with its sha256 in
        `per_protein_preds` of that block; read only by downstream/protein_level_stats.py.
All paths are relative to REPO_ROOT (the research repo).

How to run
----------
  python downstream/tokenize_misato_binding_eval.py \\
      --variant dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
      --device cuda:0 --task both --splits sequence,structure,random \\
      --use-ligand-maccs --seed 0 \\
      --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine \\
      [--p-inference 1]
"""
from __future__ import annotations

import os
# Keep CPU thread pools small (several of these jobs run in parallel).
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, average_precision_score, r2_score)
from scipy.stats import spearmanr

# Codebase root and this script's folder (for the sibling import below) on
# sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

from tokenize_misato_and_eval import load_model_robust, tokenize_misato  # noqa: E402
from ensembits.featurize import checkpoint_provenance, check_recorded_checkpoint  # noqa: E402  code_fix P17
from tokenize_misato_and_eval import token_cache_path, codebook_input_std  # noqa: E402  (2026-10-19)
# Shared function-probe rules (function_reaudit pre-run fixes, 2026-10-19).
from function_common import (binding_site_items, affinity_items, items_record,  # noqa: E402
                             binding_metrics, affinity_metrics, load_affinity_labels,
                             load_ligand_maccs, protocol_record, token_cache_record,
                             function_result_problems, atomic_write_json, add_input_std_arg,
                             input_std_tag)
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import PREDS_KEY, items_preds, write_function_preds  # noqa: E402

# ---------- data paths (all under REPO_ROOT) ----------
# Per-residue binding-site labels: PDBbind v2020 refined-set pocket residues mapped onto the
# MISATO residue list. Paper: appendix.tex \label{app:downstreams} \paragraph{Binding Site}.
# Open(audit D.4): appendix.tex still says "within 5 Å"; the labels are the PDBbind v2020
#   refined-set pocket (within 10 Å), positive rate ~0.10. Text pass item.
# FIXED(benchmark_audit D4, 2026-10-18): old = data/cached_descriptors/misato_binding_site.npz
#   (ProteinShake: pocket author residue numbers matched against MISATO's sequential counter,
#   chains / insertion codes ignored; Jaccard with the 10 Å set median 0.20, 575 complexes
#   all-negative). New = ensembits.misato_binding_labels.MISATO_BINDING_SITE
#   (data/annotations/misato_binding_site_v2.npz, data_prep/build_misato_binding_site_labels.py:
#   sequence alignment to <pid>_protein.pdb + geometric checks), read through
#   load_binding_site_labels (refuses a file without the v2 sidecar); the result JSON records
#   binding_labels_record(), and binding results of an existing JSON without it are not reused.
from ensembits.misato_binding_labels import (MISATO_BINDING_SITE as BS_LABELS,  # noqa: E402
                                             load_binding_site_labels, binding_labels_record,
                                             binding_label_problem, has_binding_results)
# Affinity labels / MACCS: ensembits.paths.MISATO_AFFINITY / MISATO_LIGAND_MACCS, read through
# function_common.load_affinity_labels / load_ligand_maccs (2026-10-19; the names below are
# kept for reference).
AFF_LABELS = ROOT / 'data' / 'annotations' / 'misato_affinity.json'
# FIXED(benchmark_audit D3, 2026-10-18): old = data/misato_splits.json (leaky UniProt / first-CATH-H
#   grouping). New = ensembits.paths.MISATO_SPLITS (data/misato_splits_v2.json), read through
#   data_versions.load_misato_splits; the result JSON records misato_splits_record(), and an
#   existing result built on another split file is not merged into (move it aside or
#   --overwrite).
from ensembits.paths import MISATO_SPLITS as SPLITS  # noqa: E402
from ensembits.data_versions import (load_misato_splits, misato_splits_record,  # noqa: E402
                                     misato_split_problem)
OUT_DIR = ROOT / 'output' / 'binding_misato'
LIGAND_MACCS = ROOT / 'data' / 'cached_descriptors' / 'misato_ligand_maccs.npz'
LIGAND_DIM = 167   # MACCS keys
# Paper: appendix.tex \label{app:downstreams} \paragraph{Binding Affinity} — 167-bit MACCS structural-key fingerprint.


# Output tag of a checkpoint folder name (strips `final_model`).
def tag_of(v): return v.replace('final_model_','').replace('final_model','') or v


def load_split(name):
    """Return {train, val, test} sets of UPPERCASE pids.

    Same split file as EC / GO (ensembits.paths.MISATO_SPLITS, v2). All three reported
    splits have a 'val' list, so the 10 % fallback below is never used for them.
    """
    d = load_misato_splits(SPLITS)[name]
    out = {}
    for k in ('train', 'val', 'test'):
        if k in d:
            out[k] = {pid.upper() for pid in d[k]}
        else: out[k] = set()
    if 'val' not in d:
        # split doesn't have val — sample 10% from train as val
        rng = np.random.default_rng(42)
        train = sorted(out['train'])
        idx = rng.permutation(len(train))
        n_val = max(1, int(0.1 * len(train)))
        out['val'] = set(train[i] for i in idx[:n_val])
        out['train'] = set(train[i] for i in idx[n_val:])
    return out


# ── Models ────────────────────────────────────────────────────────────

class BindingConv1D(nn.Module):
    """Per-residue binary classifier.

    Frozen level-1 codebook embedding (or random trainable with `random_init`)
    -> `depth` x [Conv1d + GELU + Dropout] -> Linear(hidden, 1) per residue.
    Paper: appendix.tex \\label{app:downstreams} \\paragraph{Binding Site} —
    "a per-residue 1D-CNN head with the same conv stack (no pooling)".
    """
    def __init__(self, codebook, hidden=128, depth=3, kernel=5, dropout=0.1,
                 random_init=False, rand_emb_dim=128, zero_pad=True):
        super().__init__()
        self.zero_pad = zero_pad       # Q23 (2026-10-19), see forward
        M = codebook.shape[0]
        if random_init:
            d = rand_emb_dim
            self.emb = nn.Embedding(M, d)
            nn.init.normal_(self.emb.weight, std=0.02)
        else:
            d = codebook.shape[1]
            self.emb = nn.Embedding.from_pretrained(codebook.float(), freeze=True)
        layers = []; prev = d
        for _ in range(depth):
            layers += [nn.Conv1d(prev, hidden, kernel, padding=kernel//2),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, tokens, mask):
        # FIXED(function_reaudit Q23, 2026-10-19): padded positions are zero vectors (like the
        #   baselines' zero-padded features); old (zero_pad=False): codebook[0] of pad id 0.
        x = self.emb(tokens)
        if self.zero_pad:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        x = x.transpose(1, 2)                       # (B, d, L)
        x = self.conv(x).transpose(1, 2)            # (B, L, h)
        return self.head(x).squeeze(-1)             # (B, L) logits


class AffinityConv1D(nn.Module):
    """Per-protein scalar regressor (mean+max pool, optional ligand condition).

    Same conv stack as BindingConv1D, then masked mean + max pooling; with
    `ligand_dim > 0` the MACCS vector goes through Linear(167, 64) + GELU +
    Dropout and is concatenated to the pooled protein vector; an MLP maps the
    joint vector to the affinity.
    Paper: appendix.tex \\label{app:downstreams} \\paragraph{Binding Affinity}.
    """
    def __init__(self, codebook, hidden=128, depth=3, kernel=5, dropout=0.1,
                 random_init=False, rand_emb_dim=128,
                 ligand_dim=0, lig_proj=64, zero_pad=True):
        super().__init__()
        self.zero_pad = zero_pad       # Q23 (2026-10-19), see forward
        M = codebook.shape[0]
        if random_init:
            d = rand_emb_dim
            self.emb = nn.Embedding(M, d)
            nn.init.normal_(self.emb.weight, std=0.02)
        else:
            d = codebook.shape[1]
            self.emb = nn.Embedding.from_pretrained(codebook.float(), freeze=True)
        layers = []; prev = d
        for _ in range(depth):
            layers += [nn.Conv1d(prev, hidden, kernel, padding=kernel//2),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.ligand_dim = ligand_dim
        head_in = 2 * hidden
        if ligand_dim > 0:
            # FIXED(require_fix #18, 2026-10-15): appendix now says "projected to 64 dimensions".
            self.lig_proj = nn.Sequential(
                nn.Linear(ligand_dim, lig_proj), nn.GELU(), nn.Dropout(dropout))
            head_in += lig_proj
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, tokens, mask, ligand=None):
        # FIXED(function_reaudit Q23, 2026-10-19): padded positions are zero vectors (old
        #   with zero_pad=False: codebook[0] of pad id 0, mixed into the last residues).
        x = self.emb(tokens)
        if self.zero_pad:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        m = mask.unsqueeze(-1).float()
        avg = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        very_neg = torch.full_like(x, float('-inf'))
        x_max = torch.where(mask.unsqueeze(-1), x, very_neg).max(1).values
        x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
        feat = [avg, x_max]
        if self.ligand_dim > 0:
            assert ligand is not None, 'ligand tensor required when ligand_dim>0'
            feat.append(self.lig_proj(ligand.float()))
        return self.head(torch.cat(feat, dim=-1)).squeeze(-1)


# Pad a batch for the binding-site task: (token ids, mask, per-residue labels).
def collate_bs(batch, pad_id=0):
    toks = [b['x'] for b in batch]; ys = [b['y'] for b in batch]
    L_max = max(t.shape[0] for t in toks)
    T = torch.full((len(batch), L_max), pad_id, dtype=torch.long)
    Y = torch.zeros((len(batch), L_max), dtype=torch.float32)
    M = torch.zeros((len(batch), L_max), dtype=torch.bool)
    for i, (t, y) in enumerate(zip(toks, ys)):
        L = t.shape[0]
        T[i, :L] = torch.as_tensor(t, dtype=torch.long)
        Y[i, :L] = torch.as_tensor(y, dtype=torch.float32)
        M[i, :L] = True
    return T, M, Y


# Pad a batch for the affinity task: (token ids, mask, targets[, MACCS]).
def collate_aff(batch, pad_id=0):
    toks = [b['x'] for b in batch]
    Y = torch.tensor([b['y'] for b in batch], dtype=torch.float32)
    L_max = max(t.shape[0] for t in toks)
    T = torch.full((len(batch), L_max), pad_id, dtype=torch.long)
    M = torch.zeros((len(batch), L_max), dtype=torch.bool)
    for i, t in enumerate(toks):
        L = t.shape[0]
        T[i, :L] = torch.as_tensor(t, dtype=torch.long)
        M[i, :L] = True
    if 'lig' in batch[0]:
        L_arr = torch.stack([torch.as_tensor(b['lig'], dtype=torch.float32)
                              for b in batch])
        return T, M, Y, L_arr
    return T, M, Y


def _make_sched(opt, warmup, total, cosine):
    """Per-epoch LR multiplier: linear warmup over `warmup` epochs, then
    optional cosine decay to 0 over the rest. No-op when warmup=0 & cosine=False.

    Same schedule as `_make_sched` in tokenize_misato_and_eval.py. The paper's
    numbers use warmup=5 epochs then cosine (the battery's "stab" recipe)."""
    import math
    def lr_lambda(ep):
        if warmup > 0 and ep < warmup:
            return (ep + 1) / warmup
        if cosine and total > warmup:
            prog = (ep - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# Train the binding-site head (masked per-residue BCE) and score it on test.
# AdamW (lr 1e-3, wd 1e-4), batch 32, up to 40 epochs, early stopping on the
# validation per-residue BCE; best-validation weights are evaluated.
# FIXED(require_fix #19, 2026-10-15): appendix "Probe training" paragraph lists the recipe.
def train_binding(items_tr, items_va, items_te, codebook, device,
                   epochs=40, batch_size=32, lr=1e-3, wd=1e-4, patience=8,
                   random_init=False, rand_emb_dim=128, seed=0,
                   min_epochs=0, warmup=0, cosine=False, return_preds=False):
    # return_preds: also return the selected model's per-residue val / test probabilities and
    #   labels per item under the private key '_preds' (final_fix_list A4 (a), 2026-10-19).
    torch.manual_seed(seed)
    model = BindingConv1D(codebook, random_init=random_init,
                           rand_emb_dim=rand_emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _make_sched(opt, warmup, epochs, cosine)
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = len(items_tr); best_va = float('inf'); best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).tolist()
        for s in range(0, n, batch_size):
            batch = [items_tr[i] for i in perm[s:s+batch_size]]
            T, M, Y = collate_bs(batch)
            T, M, Y = T.to(device), M.to(device), Y.to(device)
            logits = model(T, M)
            # Masked BCE: padding residues do not count.
            # Paper: appendix.tex \label{app:downstreams} \paragraph{Binding Site} — "masked binary cross-entropy".
            loss = F.binary_cross_entropy_with_logits(logits, Y, reduction='none')
            loss = (loss * M.float()).sum() / M.float().sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        v_loss = 0.0; n_v = 0
        with torch.no_grad():
            for s in range(0, len(items_va), batch_size):
                T, M, Y = collate_bs(items_va[s:s+batch_size])
                T, M, Y = T.to(device), M.to(device), Y.to(device)
                lg = F.binary_cross_entropy_with_logits(model(T, M), Y, reduction='none')
                v_loss += (lg * M.float()).sum().item()
                n_v += M.float().sum().item()
        va = v_loss / max(n_v, 1)
        sched.step()
        if va < best_va - 1e-4:
            best_va = va; no_imp = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: no_imp += 1
        if ep + 1 >= min_epochs and no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    # Test eval: AUROC and AP over all test residues pooled together.
    # Paper: appendix.tex \label{app:downstreams} \paragraph{Binding Site} — "both pooled over all test-set residues".
    # FIXED(function_reaudit pre-run fix 5, 2026-10-19): function_common.binding_metrics (same
    #   pooled AUROC / AP, plus the per-protein AUROC), on test and, in r['val'], on the
    #   validation split with the selected weights.
    model.eval()
    def _scores(items):
        all_p, all_y = [], []
        with torch.no_grad():
            for s in range(0, len(items), batch_size):
                T, M, Y = collate_bs(items[s:s+batch_size])
                T, M = T.to(device), M.to(device)
                p = torch.sigmoid(model(T, M)).cpu().numpy()
                for i in range(p.shape[0]):
                    L = int(M[i].sum().item())
                    all_p.append(p[i, :L]); all_y.append(np.asarray(items[s+i]['y']))
        return all_p, all_y
    sc_te = _scores(items_te)
    mt = binding_metrics(*sc_te)
    out = {'auroc': mt['auroc'], 'ap': mt['ap'],
           'auroc_per_protein_mean': mt['auroc_per_protein_mean'],
           'auroc_per_protein_median': mt['auroc_per_protein_median'],
           'n_test_proteins_auroc': mt['n_proteins_auroc'],
           'n_test_proteins': len(items_te), 'n_test_residues': mt['n_residues'],
           'frac_pos': mt['frac_pos'], 'epochs_run': ep + 1, 'best_val_loss': best_va}
    sc_va = _scores(items_va)
    out['val'] = {**binding_metrics(*sc_va), 'best_val_loss': best_va}
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-item predictions of the selected model.
    if return_preds:
        out['_preds'] = {'val': items_preds(items_va, *sc_va), 'test': items_preds(items_te, *sc_te)}
    return out


# Train the affinity head (MSE) and score it on test with R^2, Spearman, MSE.
# AdamW (lr 1e-3, wd 1e-4), batch 32, up to 80 epochs (patience default 10;
# the reported runs pass --patience 12 with the stab recipe).
# Paper: appendix.tex \label{app:downstreams} \paragraph{Binding Affinity} — R^2, Spearman, MSE.
def train_affinity(items_tr, items_va, items_te, codebook, device,
                    epochs=80, batch_size=32, lr=1e-3, wd=1e-4, patience=10,
                    random_init=False, rand_emb_dim=128, seed=0,
                    ligand_dim=0, min_epochs=0, warmup=0, cosine=False, return_preds=False):
    # return_preds: also return the selected model's val / test predictions and labels per
    #   complex under the private key '_preds' (final_fix_list A4 (a), 2026-10-19).
    torch.manual_seed(seed)
    model = AffinityConv1D(codebook, random_init=random_init,
                            rand_emb_dim=rand_emb_dim,
                            ligand_dim=ligand_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _make_sched(opt, warmup, epochs, cosine)
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = len(items_tr); best_va = float('inf'); best_state = None; no_imp = 0
    use_lig = ligand_dim > 0
    def _step(batch):
        out = collate_aff(batch)
        if use_lig:
            T, M, Y, L_ = out
            return T.to(device), M.to(device), Y.to(device), L_.to(device)
        else:
            T, M, Y = out
            return T.to(device), M.to(device), Y.to(device), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).tolist()
        for s in range(0, n, batch_size):
            batch = [items_tr[i] for i in perm[s:s+batch_size]]
            T, M, Y, L_ = _step(batch)
            loss = F.mse_loss(model(T, M, ligand=L_), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        v_loss = 0.0; n_v = 0
        with torch.no_grad():
            for s in range(0, len(items_va), batch_size):
                T, M, Y, L_ = _step(items_va[s:s+batch_size])
                v_loss += F.mse_loss(model(T, M, ligand=L_), Y).item() * T.shape[0]
                n_v += T.shape[0]
        va = v_loss / max(n_v, 1)
        sched.step()
        if va < best_va - 1e-4:
            best_va = va; no_imp = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: no_imp += 1
        if ep + 1 >= min_epochs and no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    # FIXED(function_reaudit pre-run fix 5, 2026-10-19): the same R² / Spearman / MSE also on
    #   the validation split with the selected weights (r['val']).
    def _pred(items):
        preds, gts = [], []
        with torch.no_grad():
            for s in range(0, len(items), batch_size):
                T, M, Y, L_ = _step(items[s:s+batch_size])
                preds.append(model(T, M, ligand=L_).cpu().numpy())
                gts.append(Y.cpu().numpy())
        return np.concatenate(preds), np.concatenate(gts)
    py, yy = _pred(items_te)
    mt = affinity_metrics(py, yy)
    pv, yv = _pred(items_va)
    return {'r2': mt['r2'], 'spearman': mt['spearman'], 'mse': mt['mse'],
            'n_test': len(yy), 'epochs_run': ep + 1, 'best_val_loss': best_va,
            'ligand_dim': int(ligand_dim),
            'val': {**affinity_metrics(pv, yv), 'best_val_loss': best_va},
            # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-complex predictions.
            **({'_preds': {'val': items_preds(items_va, pv, yv),
                           'test': items_preds(items_te, py, yy)}} if return_preds else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--p-inference', type=int, default=None)
    ap.add_argument('--random-init-embedding', action='store_true')
    ap.add_argument('--rand-emb-dim', type=int, default=128)
    ap.add_argument('--task', choices=['binding', 'affinity', 'both'], default='both')
    ap.add_argument('--splits', default='sequence,structure,random',
                    help='comma-separated split names')
    # Paper: appendix.tex \label{app:experiments} — mean +- std over 10 seeds (battery: --seed 0..9).
    ap.add_argument('--seed', type=int, default=0)
    # The paper's affinity numbers all use the MACCS ligand-aware head.
    ap.add_argument('--use-ligand-maccs', action='store_true',
                     help='Condition the affinity head on MACCS ligand fingerprints '
                          '(no effect on binding-site task). Drops pids without MACCS.')
    # Downstream-head stabilization (opt-in; defaults reproduce the paper protocol)
    # NOTE: the defaults reproduce the original (pre-camera-ready) protocol. The
    # numbers reported in the paper were run with --warmup 5 --min-epochs 20
    # --patience 12 --cosine --tag stab (RECIPE in _run_model_battery.py).
    ap.add_argument('--patience', type=int, default=None,
                    help='Early-stop patience (default: 8 binding / 10 affinity).')
    ap.add_argument('--min-epochs', type=int, default=0,
                    help='Do not early-stop before this many epochs.')
    ap.add_argument('--warmup', type=int, default=0,
                    help='Linear LR warmup over this many epochs.')
    ap.add_argument('--cosine', action='store_true',
                    help='Cosine-decay LR to 0 after warmup.')
    ap.add_argument('--tag', default='',
                    help='Extra output-filename suffix (e.g. stab) to avoid '
                         'overwriting the default-protocol runs.')
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='accept a checkpoint whose training run has not finished (code_fix P12)')
    ap.add_argument('--overwrite', action='store_true',
                    help='recompute every requested (split, task) even if the output JSON '
                         'has it (used by the seed-rescue re-run, code_fix P2; the baseline '
                         'script conv1d_binding_baselines.py always recomputes)')
    ap.add_argument('--force', action='store_true',
                    help='reuse an existing output JSON even if it was written by another '
                         'checkpoint (code_fix P17)')
    # FIXED(function input standardisation, 2026-10-19): codebook standardised by default.
    add_input_std_arg(ap)
    args = ap.parse_args()
    input_std_tag(args)
    stab = dict(min_epochs=args.min_epochs, warmup=args.warmup, cosine=args.cosine)

    # Output file name; if it exists, only the missing (split, task) entries run.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ''
    if args.p_inference is not None: suffix += f'_P{args.p_inference}inf'
    if args.random_init_embedding: suffix += f'_randEmb{args.rand_emb_dim}'
    if args.use_ligand_maccs: suffix += '_lig'
    if args.tag: suffix += f'_{args.tag}'
    if args.seed != 0: suffix += f'_seed{args.seed}'
    out_path = OUT_DIR / f'{tag_of(args.variant)}{suffix}.json'
    # FIXED(code_fix P17, 2026-10-16): the checkpoint block of this run; an existing output
    #   whose results are reused must name the same checkpoint (refused otherwise, unless
    #   --force). FIXED(code_fix P2, 2026-10-16): --overwrite recomputes everything (the
    #   seed-rescue re-run; before, an existing complete file made the re-run a no-op).
    variant_dir = ROOT / 'output' / args.variant
    ckpt_block = checkpoint_provenance(variant_dir)
    split_rec = misato_splits_record(SPLITS)          # D3 (2026-10-18)
    if out_path.exists() and not args.overwrite:
        existing = json.load(open(out_path))
        check_recorded_checkpoint(existing.get('checkpoint'), ckpt_block, out_path.name,
                                  force=args.force)
        why = misato_split_problem(existing, SPLITS)    # D3: never merge across split files
        if why:
            print(f'[error] {out_path.name} exists and {why}; move it aside or --overwrite',
                  file=sys.stderr); return 1
        # D4: never merge across binding-site label files
        why = binding_label_problem(existing, BS_LABELS) if has_binding_results(existing) else None
        if why:
            print(f'[error] {out_path.name} exists and {why}; move it aside or --overwrite',
                  file=sys.stderr); return 1
        # function_reaudit (2026-10-19): never merge into a result of the old protocol, or one
        #   on other affinity labels / MACCS files
        why = function_result_problems(existing, 'affinity' if any(
            k.endswith('_affinity') for k in existing.get('results', {})) else None)
        if why:
            print(f'[error] {out_path.name} exists and {why[0]}; move it aside or --overwrite',
                  file=sys.stderr); return 1
        # Allow re-running if any requested split/task is missing
        needed = []
        for sp in args.splits.split(','):
            for t in (['binding','affinity'] if args.task=='both' else [args.task]):
                k = f'{sp}_{t}'
                if k not in existing.get('results', {}):
                    needed.append(k)
        if not needed:
            print(f'[skip] {out_path.name}'); return 0
        results = existing.get('results', {})
    else:
        results = {}

    # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv.
    # Level-1 tokens per pid + the level-1 codebook used as the token embedding.
    feats, codebook, M = tokenize_misato(variant_dir, device=args.device,
                                          p_inference=args.p_inference,
                                          allow_unfinished=args.allow_unfinished)
    # FIXED(function input standardisation, 2026-10-19): codebook standardised over its rows
    #   (default; the rule of every lookup feature of every method). Old: raw codebook.
    codebook, in_std = codebook_input_std(codebook, args.variant,
                                          enabled=not args.no_input_standardisation,
                                          random_init=args.random_init_embedding,
                                          rand_emb_dim=args.rand_emb_dim)
    print(f'[input-std] {in_std.base_record()["rule"]}', flush=True)
    in_dim = codebook.shape[1]
    print(f'[setup] variant={args.variant} M={M} d_emb={in_dim} '
          f'p_inf={args.p_inference} rand_init={args.random_init_embedding}',
          flush=True)

    # Load labels and MACCS fingerprints.
    # FIXED(function_reaudit pre-run fixes 2 / 3, 2026-10-19): the affinity labels and the MACCS
    #   file are read through function_common (their builder sidecars are checked) and both
    #   sha256 go into the result JSON; the MACCS file is read for every affinity run, because
    #   the affinity items are the complexes with a fingerprint whether or not it is fed to
    #   the head (D5: protein-only control on the same complexes).
    bs_lab = load_binding_site_labels(BS_LABELS)       # D4: v2 labels, sidecar checked
    aff_lab, aff_rec = load_affinity_labels()
    lig_lab, lig_rec = load_ligand_maccs()
    n_res = {p: len(v) for p, v in bs_lab.items()}     # residue count = binding-label length
    print(f'[setup] {len(aff_lab)} affinity labels, MACCS for {len(lig_lab)} pids', flush=True)
    method = f'Ensembits {args.variant}'
    aff_input = 'protein+ligand' if args.use_ligand_maccs else 'protein'

    for sp in args.splits.split(','):
        try:
            split = load_split(sp)
        except Exception as e:
            print(f'  skip split {sp}: {e}'); continue
        print(f'[split={sp}] train={len(split["train"])} val={len(split["val"])} '
              f'test={len(split["test"])}', flush=True)

        # ---- Binding ----
        # FIXED(function_reaudit pre-run fix 3, 2026-10-19): old = tokens and labels cut to the
        #   shorter of the two lengths, complexes < 10 residues skipped uncounted, complexes
        #   without tokens dropped. New = function_common.binding_site_items: a length
        #   mismatch or a missing token list is an error; drops are counted per split.
        if args.task in ('binding','both'):
            bucket, dropped = binding_site_items(split, bs_lab, feats, method)
            bucket, std_rec = in_std.apply(bucket)    # 2026-10-19 (lookup: tokens unchanged)
            print(f'  binding bucket: tr={len(bucket["train"])} va={len(bucket["val"])} te={len(bucket["test"])}',
                  flush=True)
            if bucket['train'] and bucket['test']:
                t0 = time.time()
                r = train_binding(bucket['train'], bucket['val'], bucket['test'],
                                    codebook, args.device,
                                    random_init=args.random_init_embedding,
                                    rand_emb_dim=args.rand_emb_dim, seed=args.seed,
                                    patience=args.patience if args.patience is not None else 8,
                                    return_preds=True, **stab)
                preds = r.pop('_preds', None)
                r['runtime_s'] = round(time.time() - t0, 1)
                r.update({'dropped': dropped, **items_record(bucket),
                          'input_standardisation': std_rec})
                # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar
                #   <out stem>.{block}.preds.npz of this block, recorded in its `per_protein_preds`.
                r[PREDS_KEY] = write_function_preds(
                    out_path, preds, r, task='binding', block=f'{sp}_binding', split=sp,
                    method=method, p_inference=args.p_inference, seed=args.seed,
                    checkpoint_best_pt_sha256=(ckpt_block or {}).get('best_pt_sha256'))
                results[f'{sp}_binding'] = r
                # (test metrics are not printed; 2026-10-19)
                print(f'  [{sp} binding] n_test={r["n_test_proteins"]} prots [{r["runtime_s"]}s]',
                      flush=True)

        # ---- Affinity ----
        # FIXED(function_reaudit pre-run fixes 3 / 4, 2026-10-19): function_common.affinity_items:
        #   complexes with a finite label, a MACCS fingerprint and >= 10 residues, for every
        #   input mode (old: the MACCS filter only with --use-ligand-maccs); drops counted.
        if args.task in ('affinity','both'):
            bucket, dropped = affinity_items(split, aff_lab, lig_lab, n_res, feats, method,
                                             with_ligand=args.use_ligand_maccs)
            bucket, std_rec = in_std.apply(bucket)    # 2026-10-19; MACCS unchanged
            print(f'  affinity bucket: tr={len(bucket["train"])} va={len(bucket["val"])} te={len(bucket["test"])}'
                  + (' [+MACCS]' if args.use_ligand_maccs else ' [protein only]'),
                  flush=True)
            if bucket['train'] and bucket['test']:
                t0 = time.time()
                r = train_affinity(bucket['train'], bucket['val'], bucket['test'],
                                    codebook, args.device,
                                    random_init=args.random_init_embedding,
                                    rand_emb_dim=args.rand_emb_dim, seed=args.seed,
                                    ligand_dim=LIGAND_DIM if args.use_ligand_maccs else 0,
                                    patience=args.patience if args.patience is not None else 10,
                                    return_preds=True, **stab)
                preds = r.pop('_preds', None)
                r['runtime_s'] = round(time.time() - t0, 1)
                r.update({'dropped': dropped, 'affinity_input': aff_input, **items_record(bucket),
                          'input_standardisation': std_rec})
                # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar
                #   <out stem>.{block}.preds.npz of this block, recorded in its `per_protein_preds`.
                r[PREDS_KEY] = write_function_preds(
                    out_path, preds, r, task='affinity', block=f'{sp}_affinity', split=sp,
                    method=method, p_inference=args.p_inference, seed=args.seed,
                    checkpoint_best_pt_sha256=(ckpt_block or {}).get('best_pt_sha256'))
                results[f'{sp}_affinity'] = r
                # (test metrics are not printed; 2026-10-19)
                print(f'  [{sp} affinity] n_test={r["n_test"]} [{r["runtime_s"]}s]', flush=True)

    # One JSON with all (split, task) results of this run.
    out = {'variant': args.variant, 'p_inference': args.p_inference,
           'random_init_embedding': args.random_init_embedding,
           'rand_emb_dim': args.rand_emb_dim, 'M': int(M),
           'd_emb': int(in_dim), 'seed': args.seed, 'results': results,
           'checkpoint': ckpt_block,     # FIXED(code_fix P17, 2026-10-16): new key (old: none)
           **split_rec,                  # D3 (2026-10-18): split file path + sha256
           **binding_labels_record(BS_LABELS),   # D4 (2026-10-18): label file path + sha256
           # function_reaudit 2026-10-19: affinity labels + MACCS sha256, token cache, protocol
           'affinity_labels': aff_rec, 'ligand_maccs': lig_rec,
           'affinity_input': aff_input,
           **protocol_record([token_cache_record(token_cache_path(variant_dir, args.p_inference))],
                             input_std=in_std.run_record(
                                 [(k, v['input_standardisation']) for k, v in sorted(results.items())
                                  if 'input_standardisation' in v]))}
    atomic_write_json(out_path, out)
    print(f'[done] {out_path.name}', flush=True)


if __name__ == '__main__':
    # FIXED(code_fix P19, 2026-10-16): propagate main()'s return code (refusals exit non-zero).
    sys.exit(main())
