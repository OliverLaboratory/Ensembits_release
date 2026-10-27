"""Binding site and binding affinity on MISATO for the baselines (Conv1d heads on fixed features).

What it does
------------
Per-residue features come from `load_features` of
`ec_classify_conv1d_misato.py` (3Di tokens: fixed 2-D centroid lookup built
here; `ligand_only`: no protein input). For each split in `--splits`:
  * binding site (`--task binding|both`): per-residue Conv1d head
    (`BindingConv1DCont`, 3 x Conv1d(k=5) + linear) with masked BCE, early
    stopping on val loss; test AUROC and AP over all test residues;
  * affinity (`--task affinity|both`): Conv1d stack + masked mean+max pooling,
    optionally concatenated with a projection of the 167-bit ligand MACCS
    fingerprint (`--use-ligand-maccs`, drops pids without MACCS), MLP -> MSE;
    test R2, Spearman, MSE. `ligand_only` fits an MLP on MACCS alone.
`--concat-aa` appends the 21-D aa one-hot to the feature. The learning-rate
schedule (`_make_sched`) and the label split loader (`load_split`) are imported
from `downstream/tokenize_misato_binding_eval.py`, so baselines and Ensembits
use the same protocol. Results of all splits go into one JSON (merged with an
existing file of the same name).

Paper
-----
  * `appendix.tex` \\label{app:downstreams}, \\paragraph{Binding Site}
    (Table \\label{tab:binding_structure}) and \\paragraph{Binding Affinity}
    (Table \\label{tab:affinity_structure}), baseline rows;
  * `appendix.tex` \\label{app:experiments}, \\subsection{Binding Site},
    \\subsection{Binding Affinity} (structure / sequence / random splits);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * baselines: `appendix.tex` \\label{app:baselines}.

Origin
------
Copied from `submission_exp/src/conv1d_binding_baselines.py`. Changed: `ROOT`
from `ensembits.paths.REPO_ROOT`; `ec_classify_conv1d_misato` is imported from
this folder and `tokenize_misato_binding_eval` from `downstream/` (both were in
`submission_exp/src`); comments and this docstring. Logic unchanged. (The
FileNotFoundError text still names the research-repo builder
`submission_exp/src/build_misato_ligand_features.py`, not part of this
codebase, code_fix C7.)
Fix 2026-10-18 (benchmark_audit D4): binding-site labels v2 (FIXED marker at BS_LABELS).
Pre-run fixes 2026-10-19 (camera_ready/function_reaudit.md; FIXED markers): provenance-checked
features of every split-file complex (aa from misato_real_bb); binding / affinity items from
function_common (length mismatch = error instead of cutting to the shorter length; drops
counted per split; every affinity mode on the same complexes: finite label, MACCS, >= 10
residues); affinity labels / MACCS read with their builder sidecars and recorded (sha256);
D5 controls: `ligand_only` (batch size 32 as the other affinity heads) and protein only
(no --use-ligand-maccs); validation metrics in r['val'] (binding: + per-protein AUROC);
protocol record; atomic write; test metrics not printed.

FIXED(function input standardisation, authors' decision 2026-10-19): the feature's input rule
from `load_features_checked` (lookup tables standardised over their rows; continuous features
over the residues of each split's training items of the task, after binding_site_items /
affinity_items); `--concat-aa`: each block by its own rule (aa = lookup); the MACCS ligand
input unchanged (binary; function_common.LIGAND_RULE) for every method incl. ligand_only.
Every (split, task) block and the JSON record `input_standardisation`.
`--no-input-standardisation` = raw (old protocol, tag + "rawin").

Used by the reported model: no (baseline rows of the binding / affinity
tables). Run by `downstream/baselines/run_misato_baselines.py` with
`--task both --use-ligand-maccs`.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/annotations/misato_binding_site_v2.npz (binding-site labels v2, D4; through
          ensembits.misato_binding_labels), data/annotations/misato_affinity.json (+ sidecar),
          data/cached_descriptors/misato_ligand_maccs.npz (+ sidecar; every affinity run),
          data/cached_descriptors/mini3di_tokens_misato.npz, mini3di_centroids.npy (3di_tokens),
          data/misato_splits_v2.json (via load_split), the feature cache of `load_features`
  writes  output/binding_misato/baseline_<feature>[_aa][_lig][_<tag>][_seed<s>].json
          Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): <json stem>.<split>_<task>.preds.npz per
          block (binding: per-residue probabilities and labels; affinity / ligand only:
          per-complex prediction and label; val and test), written next to
          the JSON (downstream/protein_preds.py) and recorded with its sha256 in
          `per_protein_preds` of that block; read only by downstream/protein_level_stats.py.

How to run
----------
    python downstream/baselines/conv1d_binding_baselines.py --feature esm3struct \\
        --task both --splits sequence,structure,random --use-ligand-maccs --seed 0 \\
        --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine --device cuda:0
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from scipy.stats import spearmanr

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))            # tokenize_misato_*_eval.py
sys.path.insert(0, str(Path(__file__).resolve().parent))    # sibling probes in downstream/baselines/
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

from ec_classify_conv1d_misato import load_features, load_features_checked  # noqa: E402  aa / esm3struct / ...
# Shared function-probe rules (function_reaudit pre-run fixes, 2026-10-19).
from function_common import (binding_site_items, affinity_items, items_record,  # noqa: E402
                             binding_metrics, affinity_metrics, load_affinity_labels,
                             load_ligand_maccs, protocol_record, check_length,
                             function_result_problems, atomic_write_json, add_input_std_arg,
                             input_std_tag, InputStandardisation)
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import PREDS_KEY, items_preds, write_function_preds  # noqa: E402
from tokenize_misato_binding_eval import (BindingConv1D, AffinityConv1D,  # noqa: E402
                                           collate_bs, collate_aff, load_split,
                                           train_binding, train_affinity,
                                           _make_sched, SPLITS)
from ensembits.data_versions import misato_splits_record, misato_split_problem  # noqa: E402  D3

# FIXED(benchmark_audit D4, 2026-10-18): old = data/cached_descriptors/misato_binding_site.npz
#   (pocket author numbers matched against MISATO's sequential counter, wrong residues). New =
#   v2 labels (data/annotations/misato_binding_site_v2.npz, PDBbind pocket aligned to MISATO
#   residues, data_prep/build_misato_binding_site_labels.py) read with load_binding_site_labels;
#   the output records binding_labels_record(), and an existing output whose binding results
#   lack it is not merged into.
from ensembits.misato_binding_labels import (MISATO_BINDING_SITE as BS_LABELS,  # noqa: E402
                                             load_binding_site_labels, binding_labels_record,
                                             binding_label_problem, has_binding_results)
AFF_LABELS = ROOT / 'data' / 'annotations' / 'misato_affinity.json'
TOKENS_3DI = ROOT / 'data' / 'cached_descriptors' / 'mini3di_tokens_misato.npz'
CENTROIDS = ROOT / 'data' / 'cached_descriptors' / 'mini3di_centroids.npy'
OUT_DIR = ROOT / 'output' / 'binding_misato'
LIGAND_MACCS = ROOT / 'data' / 'cached_descriptors' / 'misato_ligand_maccs.npz'
LIGAND_DIM = 167


# Continuous-feature versions of the heads
class BindingConv1DCont(nn.Module):
    def __init__(self, in_dim, hidden=128, depth=3, kernel=5, dropout=0.1):
        super().__init__()
        layers = []; prev = in_dim
        for _ in range(depth):
            layers += [nn.Conv1d(prev, hidden, kernel, padding=kernel//2),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, mask):
        x = x.transpose(1, 2); x = self.conv(x).transpose(1, 2)
        return self.head(x).squeeze(-1)


class AffinityConv1DCont(nn.Module):
    def __init__(self, in_dim, hidden=128, depth=3, kernel=5, dropout=0.1,
                 ligand_dim=0, lig_proj=64):
        super().__init__()
        layers = []; prev = in_dim
        for _ in range(depth):
            layers += [nn.Conv1d(prev, hidden, kernel, padding=kernel//2),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.ligand_dim = ligand_dim
        head_in = 2 * hidden
        if ligand_dim > 0:
            self.lig_proj = nn.Sequential(
                nn.Linear(ligand_dim, lig_proj), nn.GELU(), nn.Dropout(dropout))
            head_in += lig_proj
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, x, mask, ligand=None):
        x = x.transpose(1, 2); x = self.conv(x).transpose(1, 2)
        m = mask.unsqueeze(-1).float()
        avg = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        very_neg = torch.full_like(x, float('-inf'))
        x_max = torch.where(mask.unsqueeze(-1), x, very_neg).max(1).values
        x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
        feat = [avg, x_max]
        if self.ligand_dim > 0:
            assert ligand is not None
            feat.append(self.lig_proj(ligand.float()))
        return self.head(torch.cat(feat, dim=-1)).squeeze(-1)


class LigandOnlyMLP(nn.Module):
    """Affinity floor: MLP on MACCS with no protein input."""
    def __init__(self, ligand_dim=167, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ligand_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, ligand):
        return self.net(ligand.float()).squeeze(-1)


def collate_cont_bs(batch):
    xs = [b['x'] for b in batch]; ys = [b['y'] for b in batch]
    L_max = max(x.shape[0] for x in xs); D = xs[0].shape[1]
    X = torch.zeros((len(batch), L_max, D), dtype=torch.float32)
    Y = torch.zeros((len(batch), L_max), dtype=torch.float32)
    M = torch.zeros((len(batch), L_max), dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]
        X[i, :L] = torch.as_tensor(x); Y[i, :L] = torch.as_tensor(y); M[i, :L] = True
    return X, M, Y


def collate_cont_aff(batch):
    xs = [b['x'] for b in batch]
    Y = torch.tensor([b['y'] for b in batch], dtype=torch.float32)
    L_max = max(x.shape[0] for x in xs); D = xs[0].shape[1]
    X = torch.zeros((len(batch), L_max, D), dtype=torch.float32)
    M = torch.zeros((len(batch), L_max), dtype=torch.bool)
    for i, x in enumerate(xs):
        L = x.shape[0]
        X[i, :L] = torch.as_tensor(x); M[i, :L] = True
    if 'lig' in batch[0]:
        L_arr = torch.stack([torch.as_tensor(b['lig'], dtype=torch.float32)
                              for b in batch])
        return X, M, Y, L_arr
    return X, M, Y


def train_binding_cont(items_tr, items_va, items_te, in_dim, device,
                        epochs=40, batch_size=32, lr=1e-3, wd=1e-4, patience=8,
                        seed=0, min_epochs=0, warmup=0, cosine=False, return_preds=False):
    # return_preds: also return the selected model's per-residue val / test probabilities and
    #   labels per item under the private key '_preds' (final_fix_list A4 (a), 2026-10-19).
    torch.manual_seed(seed)
    model = BindingConv1DCont(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _make_sched(opt, warmup, epochs, cosine)
    g = torch.Generator(device='cpu').manual_seed(seed)
    best_va = float('inf'); best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(items_tr), generator=g).tolist()
        for s in range(0, len(items_tr), batch_size):
            batch = [items_tr[i] for i in perm[s:s+batch_size]]
            X, M, Y = collate_cont_bs(batch)
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(X, M), Y, reduction='none')
            loss = (loss * M.float()).sum() / M.float().sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        v_loss = 0.0; n_v = 0
        with torch.no_grad():
            for s in range(0, len(items_va), batch_size):
                X, M, Y = collate_cont_bs(items_va[s:s+batch_size])
                X, M, Y = X.to(device), M.to(device), Y.to(device)
                lg = F.binary_cross_entropy_with_logits(model(X, M), Y, reduction='none')
                v_loss += (lg * M.float()).sum().item(); n_v += M.float().sum().item()
        va = v_loss / max(n_v, 1)
        sched.step()
        if va < best_va - 1e-4: best_va, no_imp = va, 0
        else: no_imp += 1
        if best_va == va: best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep + 1 >= min_epochs and no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    # FIXED(function_reaudit pre-run fix 5, 2026-10-19): function_common.binding_metrics (same
    #   pooled AUROC / AP, plus the per-protein AUROC) on test and, in r['val'], on validation.
    def _scores(items):
        all_p, all_y = [], []
        with torch.no_grad():
            for s in range(0, len(items), batch_size):
                X, M, Y = collate_cont_bs(items[s:s+batch_size])
                X, M = X.to(device), M.to(device)
                p = torch.sigmoid(model(X, M)).cpu().numpy()
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


def train_affinity_cont(items_tr, items_va, items_te, in_dim, device,
                         epochs=80, batch_size=32, lr=1e-3, wd=1e-4, patience=10,
                         seed=0, ligand_dim=0, min_epochs=0, warmup=0, cosine=False,
                         return_preds=False):
    # return_preds: per-complex val / test predictions under '_preds' (A4 (a), 2026-10-19).
    torch.manual_seed(seed)
    model = AffinityConv1DCont(in_dim, ligand_dim=ligand_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _make_sched(opt, warmup, epochs, cosine)
    g = torch.Generator(device='cpu').manual_seed(seed)
    best_va = float('inf'); best_state = None; no_imp = 0
    use_lig = ligand_dim > 0
    def _step(batch):
        out = collate_cont_aff(batch)
        if use_lig:
            X, M, Y, L_ = out
            return X.to(device), M.to(device), Y.to(device), L_.to(device)
        X, M, Y = out
        return X.to(device), M.to(device), Y.to(device), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(items_tr), generator=g).tolist()
        for s in range(0, len(items_tr), batch_size):
            batch = [items_tr[i] for i in perm[s:s+batch_size]]
            X, M, Y, L_ = _step(batch)
            loss = F.mse_loss(model(X, M, ligand=L_), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        v_loss = 0.0; n_v = 0
        with torch.no_grad():
            for s in range(0, len(items_va), batch_size):
                X, M, Y, L_ = _step(items_va[s:s+batch_size])
                v_loss += F.mse_loss(model(X, M, ligand=L_), Y).item() * X.shape[0]
                n_v += X.shape[0]
        va = v_loss / max(n_v, 1)
        sched.step()
        if va < best_va - 1e-4: best_va, no_imp = va, 0
        else: no_imp += 1
        if best_va == va: best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep + 1 >= min_epochs and no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    # FIXED(function_reaudit pre-run fix 5, 2026-10-19): also R² / Spearman / MSE on validation.
    def _pred(items):
        preds, gts = [], []
        with torch.no_grad():
            for s in range(0, len(items), batch_size):
                X, M, Y, L_ = _step(items[s:s+batch_size])
                preds.append(model(X, M, ligand=L_).cpu().numpy())
                gts.append(Y.cpu().numpy())
        return np.concatenate(preds), np.concatenate(gts)
    py, yy = _pred(items_te)
    mt = affinity_metrics(py, yy)
    pv, yv = _pred(items_va)
    return {'r2': mt['r2'], 'spearman': mt['spearman'], 'mse': mt['mse'], 'n_test': len(yy),
            'epochs_run': ep + 1, 'best_val_loss': best_va,
            'ligand_dim': int(ligand_dim),
            'val': {**affinity_metrics(pv, yv), 'best_val_loss': best_va},
            # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-complex predictions.
            **({'_preds': {'val': items_preds(items_va, pv, yv),
                           'test': items_preds(items_te, py, yy)}} if return_preds else {})}


# FIXED(function_reaudit pre-run fix 4, 2026-10-19): default batch size 32, as every other
#   affinity head (old: 64); the recipe (epochs 80, AdamW lr 1e-3 / wd 1e-4, warmup / cosine /
#   min-epochs / patience from the CLI, early stopping on val MSE) is the protein heads'.
def train_ligand_only_affinity(items_tr, items_va, items_te, device,
                                epochs=80, batch_size=32, lr=1e-3, wd=1e-4,
                                patience=10, seed=0, min_epochs=0, warmup=0, cosine=False,
                                return_preds=False):
    """MLP on MACCS only — affinity floor with no protein input (D5 control row)."""
    torch.manual_seed(seed)
    model = LigandOnlyMLP(ligand_dim=LIGAND_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = _make_sched(opt, warmup, epochs, cosine)
    g = torch.Generator(device='cpu').manual_seed(seed)
    best_va = float('inf'); best_state = None; no_imp = 0
    def _stack(batch):
        L_ = torch.stack([torch.as_tensor(b['lig'], dtype=torch.float32)
                           for b in batch])
        Y = torch.tensor([b['y'] for b in batch], dtype=torch.float32)
        return L_.to(device), Y.to(device)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(items_tr), generator=g).tolist()
        for s in range(0, len(items_tr), batch_size):
            batch = [items_tr[i] for i in perm[s:s+batch_size]]
            L_, Y = _stack(batch)
            loss = F.mse_loss(model(L_), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        v_loss = 0.0; n_v = 0
        with torch.no_grad():
            for s in range(0, len(items_va), batch_size):
                L_, Y = _stack(items_va[s:s+batch_size])
                v_loss += F.mse_loss(model(L_), Y).item() * L_.shape[0]
                n_v += L_.shape[0]
        va = v_loss / max(n_v, 1)
        sched.step()
        if va < best_va - 1e-4: best_va, no_imp = va, 0
        else: no_imp += 1
        if best_va == va: best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep + 1 >= min_epochs and no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    # FIXED(function_reaudit pre-run fix 5, 2026-10-19): also R² / Spearman / MSE on validation.
    def _pred(items):
        preds, gts = [], []
        with torch.no_grad():
            for s in range(0, len(items), batch_size):
                L_, Y = _stack(items[s:s+batch_size])
                preds.append(model(L_).cpu().numpy()); gts.append(Y.cpu().numpy())
        return np.concatenate(preds), np.concatenate(gts)
    py, yy = _pred(items_te)
    mt = affinity_metrics(py, yy)
    pv, yv = _pred(items_va)
    return {'r2': mt['r2'], 'spearman': mt['spearman'], 'mse': mt['mse'], 'n_test': len(yy),
            'epochs_run': ep + 1, 'best_val_loss': best_va,
            'ligand_dim': LIGAND_DIM,
            'val': {**affinity_metrics(pv, yv), 'best_val_loss': best_va},
            # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-complex predictions.
            **({'_preds': {'val': items_preds(items_va, pv, yv),
                           'test': items_preds(items_te, py, yy)}} if return_preds else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', required=True,
                    choices=['aa','esm2','mini3di','protprofile_K5',
                             'protprofile_K8','3di_tokens',
                             'vote_3di','pca_eig','random',
                             'aminoaseed','protoken','esm3struct',
                             'aug_rot6d','ensembits_ifold',
                             'ensembits_ifold_L0',
                             'big_divafs','big_divonly_p10',
                             'ligand_only',
                             'esm3struct_pooled','aminoaseed_pooled','3di_pooled',
                             'protoken_pooled',
                             # static tokenizers retrained on OUR corpus (rebuttal)
                             'aminoaseed_ourdata','vanillavq_ourdata',
                             'esm3style_ourdata',
                             # retrained on BioEmu-CATH + MISATO (camera-ready P-17/P-26/P-42)
                             'aminoaseed_retrained','esm3struct_retrained'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--task', choices=['binding','affinity','both'], default='both')
    ap.add_argument('--splits', default='sequence,structure,random')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--use-ligand-maccs', action='store_true',
                     help='Condition the affinity head on MACCS ligand fingerprints. '
                          'Ignored for binding-site task. (Since 2026-10-19 every affinity run '
                          'uses only complexes with MACCS; without this flag = protein-only '
                          'control, D5.)')
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--concat-aa', action='store_true',
                     help='Concatenate per-residue AA one-hot (21-D) to the chosen '
                          'protein feature before the conv1d head.')
    ap.add_argument('--allow-unverified-cache', action='store_true',
                    help='accept a feature cache without a current provenance sidecar (the '
                         'result is marked unverified; compilers refuse it)')
    add_input_std_arg(ap)              # FIXED(function input standardisation, 2026-10-19)
    args = ap.parse_args()
    input_std_tag(args)
    do_std = not args.no_input_standardisation

    if args.feature == 'ligand_only':
        # Force ligand mode and affinity-only.
        args.use_ligand_maccs = True
        if args.task == 'binding':
            print('ligand_only: binding task is meaningless without protein input. '
                  'Forcing --task affinity.', flush=True)
        args.task = 'affinity'

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_tag = '' if args.seed == 0 else f'_seed{args.seed}'
    lig_tag = '_lig' if (args.use_ligand_maccs and args.feature != 'ligand_only') else ''
    aa_tag = '_aa' if args.concat_aa else ''
    tag_seg = f'_{args.tag}' if args.tag else ''
    out_path = OUT_DIR / f'baseline_{args.feature}{aa_tag}{lig_tag}{tag_seg}{seed_tag}.json'
    # FIXED(benchmark_audit D3, 2026-10-18): an existing result built on another split file is
    #   not merged into (old: its results were kept and extended); the output records the
    #   split file (misato_splits_record).
    split_rec = misato_splits_record(SPLITS)
    if out_path.exists():
        existing = json.load(open(out_path)); results = existing.get('results', {})
        why = misato_split_problem(existing, SPLITS)
        if why:
            print(f'[error] {out_path.name} exists and {why}; move it aside', file=sys.stderr)
            sys.exit(1)
        # D4: never merge across binding-site label files
        why = binding_label_problem(existing, BS_LABELS) if has_binding_results(existing) else None
        if why:
            print(f'[error] {out_path.name} exists and {why}; move it aside', file=sys.stderr)
            sys.exit(1)
        # function_reaudit (2026-10-19): never merge into a result of the old protocol, or one
        #   on other affinity labels / MACCS files
        why = function_result_problems(existing, 'affinity' if any(
            k.endswith('_affinity') for k in results) else None)
        if why:
            print(f'[error] {out_path.name} exists and {why[0]}; move it aside', file=sys.stderr)
            sys.exit(1)
    else: results = {}

    # FIXED(function_reaudit pre-run fixes 2 / 3, 2026-10-19): affinity labels and MACCS through
    #   function_common (builder sidecars checked, sha256 recorded); MACCS read for every
    #   affinity run (the items are the complexes with a fingerprint in every input mode).
    bs_lab = load_binding_site_labels(BS_LABELS)       # D4: v2 labels, sidecar checked
    aff_lab, aff_rec = load_affinity_labels()
    lig_lab, lig_rec = load_ligand_maccs()
    n_res = {p: len(v) for p, v in bs_lab.items()}     # residue count = binding-label length
    print(f'[setup] {len(aff_lab)} affinity labels, MACCS for {len(lig_lab)} pids', flush=True)

    # ligand_only: no protein features (D5 control: MACCS -> MLP).
    # FIXED(function_reaudit pre-run fixes 1 / 2, 2026-10-19): protein features of every
    #   split-file complex with a provenance-checked cache (load_features_checked; old:
    #   load_features without checks, aa on 3,903 complexes, 3di_tokens built here).
    if args.feature == 'ligand_only':
        feats, in_dim = None, 0
        feat_recs = [{**lig_rec, 'file': Path(lig_rec['path']).name, 'kind': 'ligand_maccs',
                      'verified': True}]          # its sidecar was checked by load_ligand_maccs
        # no per-residue input; MACCS unchanged (2026-10-19)
        in_std = (InputStandardisation('ligand_only', []) if do_std
                  else InputStandardisation.disabled('ligand_only'))
    else:
        feats, in_dim, feat_recs, in_std = load_features_checked(
            args.feature, args.allow_unverified_cache, standardise=do_std)
        print(f'[setup] {args.feature}: in_dim={in_dim}, n_proteins={len(feats)}',
              flush=True)

    # Optional: concatenate per-residue AA one-hot (21-D, 20 AAs + X) to the
    # protein feature.
    # FIXED(function_reaudit, 2026-10-19): the aa one-hot is the misato_real_bb sequence (same
    #   residue list as every cache) and a length mismatch is an error (old: the pairs-pickle
    #   sequence, both cut to min(L) from the N-terminus).
    if args.concat_aa and args.feature not in ('ligand_only', 'aa'):
        aa_feats, aa_dim, aa_recs, aa_std = load_features_checked(
            'aa', args.allow_unverified_cache, standardise=do_std)
        new_feats = {}
        for pid, f in feats.items():
            check_length(pid, len(f), len(aa_feats[pid]), args.feature, 'concat-aa')
            new_feats[pid] = np.concatenate([f, aa_feats[pid]], axis=-1).astype(np.float32)
        feats = new_feats
        in_dim = in_dim + aa_dim
        in_std = in_std.concat(aa_std)   # 2026-10-19: each block by its own rule
        feat_recs = feat_recs + aa_recs
        print(f'[setup] +concat_aa: in_dim={in_dim}, n_proteins={len(feats)}',
              flush=True)

    aff_input = ('ligand' if args.feature == 'ligand_only' else
                 'protein+ligand' if args.use_ligand_maccs else 'protein')
    for sp in args.splits.split(','):
        try: split = load_split(sp)
        except Exception as e: print(f'  skip {sp}: {e}'); continue
        # FIXED(function_reaudit pre-run fix 3, 2026-10-19): function_common.binding_site_items
        #   (old: features and labels cut to min(L), < 10 residues skipped uncounted).
        if args.task in ('binding','both'):
            bucket, dropped = binding_site_items(split, bs_lab, feats, args.feature)
            bucket, std_rec = in_std.apply(bucket)    # 2026-10-19: training residues only
            print(f'  [{sp} binding] tr={len(bucket["train"])} va={len(bucket["val"])} te={len(bucket["test"])}',
                  flush=True)
            if bucket['train'] and bucket['test']:
                t0 = time.time()
                r = train_binding_cont(bucket['train'], bucket['val'], bucket['test'],
                                        in_dim, args.device, seed=args.seed,
                                        patience=args.patience if args.patience is not None else 8,
                                        min_epochs=args.min_epochs, warmup=args.warmup,
                                        cosine=args.cosine, return_preds=True)
                preds = r.pop('_preds', None)
                r['runtime_s'] = round(time.time() - t0, 1)
                r.update({'dropped': dropped, **items_record(bucket),
                          'input_standardisation': std_rec})
                # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar
                #   <out stem>.{block}.preds.npz of this block, recorded in its `per_protein_preds`.
                r[PREDS_KEY] = write_function_preds(
                    out_path, preds, r, task='binding', block=f'{sp}_binding', split=sp,
                    method=args.feature, seed=args.seed)
                results[f'{sp}_binding'] = r
                print(f'  [{sp} binding] done [{r["runtime_s"]}s]', flush=True)   # no test metric printed
        # FIXED(function_reaudit pre-run fixes 3 / 4, 2026-10-19): function_common.affinity_items:
        #   the same complexes (finite label, MACCS, >= 10 residues) for protein + ligand,
        #   protein-only and ligand-only runs; drops counted.
        if args.task in ('affinity','both'):
            bucket, dropped = affinity_items(split, aff_lab, lig_lab, n_res, feats, args.feature,
                                             with_ligand=args.use_ligand_maccs)
            bucket, std_rec = in_std.apply(bucket)    # 2026-10-19; MACCS unchanged
            print(f'  [{sp} affinity] tr={len(bucket["train"])} va={len(bucket["val"])} te={len(bucket["test"])}'
                  + f' [{aff_input}]', flush=True)
            if bucket['train'] and bucket['test']:
                t0 = time.time()
                if args.feature == 'ligand_only':
                    r = train_ligand_only_affinity(bucket['train'], bucket['val'],
                                                    bucket['test'], args.device,
                                                    seed=args.seed,
                                                    patience=args.patience if args.patience is not None else 10,
                                                    min_epochs=args.min_epochs, warmup=args.warmup,
                                                    cosine=args.cosine, return_preds=True)
                else:
                    r = train_affinity_cont(bucket['train'], bucket['val'], bucket['test'],
                                             in_dim, args.device, seed=args.seed,
                                             ligand_dim=LIGAND_DIM if args.use_ligand_maccs else 0,
                                             patience=args.patience if args.patience is not None else 10,
                                             min_epochs=args.min_epochs, warmup=args.warmup,
                                             cosine=args.cosine, return_preds=True)
                r['runtime_s'] = round(time.time() - t0, 1)
                preds = r.pop('_preds', None)
                r.update({'dropped': dropped, 'affinity_input': aff_input, **items_record(bucket),
                          'input_standardisation': std_rec})
                # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar
                #   <out stem>.{block}.preds.npz of this block, recorded in its `per_protein_preds`.
                r[PREDS_KEY] = write_function_preds(
                    out_path, preds, r, task='affinity', block=f'{sp}_affinity', split=sp,
                    method=args.feature, seed=args.seed)
                results[f'{sp}_affinity'] = r
                print(f'  [{sp} affinity] done [{r["runtime_s"]}s]', flush=True)   # no test metric printed

    out = {'feature': args.feature, 'in_dim': in_dim, 'seed': args.seed,
           'results': results, **split_rec,       # D3: split file path + sha256
           **binding_labels_record(BS_LABELS),    # D4: label file path + sha256
           # function_reaudit 2026-10-19: affinity labels + MACCS sha256, inputs, protocol
           'affinity_labels': aff_rec, 'ligand_maccs': lig_rec, 'affinity_input': aff_input,
           'concat_aa': bool(args.concat_aa),
           **protocol_record(feat_recs, input_std=in_std.run_record(
               [(k, v['input_standardisation']) for k, v in sorted(results.items())
                if 'input_standardisation' in v]))}
    atomic_write_json(out_path, out)
    print(f'[done] {out_path.name}', flush=True)


if __name__ == '__main__': main()
