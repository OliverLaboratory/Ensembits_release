"""t-SNE of the level-1 codebook, coloured by how often each code is used on the training corpus.

What it does
------------
  1. Loads an Ensembits checkpoint and takes its level-1 codebook (M_1 = 2048
     vectors of dimension 128).
  2. Counts, for every code, how many residues of the model's TRAINING corpus
     (BioEmu-CATH + MISATO, the descriptor cache written by
     `data_prep/build_bioemu_misato_descriptor.py`) are assigned to it: the corpus
     descriptor file of the checkpoint's dihedral convention is tokenized row by
     row (P = 10 frames per residue). `--split all` (default) counts every system
     of the corpus; `--split train|val|test` uses the residue rows of splits.json.
  3. Runs t-SNE (cosine metric, PCA init) on the codebook and plots one point per
     code, coloured by log10(1 + usage), unused codes in grey, the top-N most used
     codes annotated.
  4. Writes the figure (PNG + PDF) and, with `--counts-cache`, the per-code counts
     with their provenance (checkpoint hash, convention, corpus, split).

Paper
-----
  * `appendix.tex`, \\subsection{Codebook structure} (\\label{sec:codebook_tsne})
    and Figure fig:codebook_tsne (`img/tSNE_codes.pdf`): utilisation, top-to-mean
    usage ratio, p99/p50, number of residue-token assignments and ensembles. The
    printed summary gives these numbers for the text.

Origin
------
Copied from `experiments/shared_states_v2/plot_codebook_tsne.py`. Changes, each
marked FIXED(code_fix P20, 2026-10-16):
  * usage counts over the model's training corpus (`--corpus-dir`, default
    REPO_ROOT/output/bioemu_cath_misato_e0/cached_desc: layout.json, splits.json and
    the convention-named descriptor file) instead of tokenizing mdCATH-div
    on the fly plus an old MISATO token file (`tokenize_corpus` and the
    `--src` / `--misato-tokens-npz` options were removed);
  * the corpus convention (layout.json) and chain-break cutoff must match the
    checkpoint's; the counts cache records and checks the checkpoint hash;
  * model dir default `ensembits.paths.CKPT_DIR`; config keys `k` and
    `descriptor_mode` read with fallbacks; the import of `_resolve_descriptor_mode`
    (not in this codebase) was dropped.
`get_codebook` and the plot are unchanged.

Used by the reported model: yes (Figure fig:codebook_tsne, after the retrain).

Inputs / outputs
----------------
  reads   <corpus-dir>/layout.json, splits.json, esm3desc_dihedral[_iupac]_K16_P10.npy
          <model-dir>/config.json, best.pt, stats.npz
  writes  --out (PNG) and the same name with .pdf; --counts-cache JSON
          (default --out REPO_ROOT/experiments/results/search_demo/figures/codebook_tsne.png)

How to run
----------
    python downstream/analyses/plot_codebook_tsne.py --model-dir ckpt/<retrained> \\
        --counts-cache output/codebook_tsne/counts.json
    # the current checkpoint's (legacy) corpus:
    #   --corpus-dir output/reviewer_3_special_corpus_MISATO/cached_desc
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _common.py
from ensembits.paths import REPO_ROOT as PROJECT_ROOT, CKPT_DIR  # noqa: E402
from ensembits.featurize import load_model  # noqa: E402
from ensembits.descriptor import with_convention, resolve_chain_break_cutoff  # noqa: E402
from _common import resolve_model_dir, ckpt_hash, cfg_get, checkpoint_settings  # noqa: E402

DEFAULT_MODEL_DIR = CKPT_DIR
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "output" / "bioemu_cath_misato_e0" / "cached_desc"
DEFAULT_OUT_PNG = PROJECT_ROOT / "experiments" / "results" / "search_demo" / "figures" / "codebook_tsne.png"


def get_codebook(ens, level: int = 0) -> np.ndarray:
    """Pull codebook weights for level `level` from an Ensembits model.
    Works for v1 (single VQ) and v2 (RVQ stack)."""
    state = ens.model.state_dict()
    keys = [k for k in state if "codebook.weight" in k]
    if not keys:
        raise RuntimeError(f"No codebook.weight in state dict; keys: {list(state.keys())[:10]}")
    # v2 RVQ: vq_levels.{i}.codebook.weight; v1: vq.codebook.weight
    matched = [k for k in keys if f".{level}.codebook" in k or (level == 0 and "vq.codebook" in k)]
    key = matched[0] if matched else keys[level]
    return state[key].detach().cpu().numpy().astype(np.float32)


def open_corpus(corpus_dir: Path, conv: str, cutoff):
    """(layout rows, settings, mmap descriptor) of a training corpus, checked against
    the checkpoint's convention and cutoff."""
    lj = json.loads((corpus_dir / "layout.json").read_text())
    rows = lj["rows"] if isinstance(lj, dict) else lj
    settings = {k: v for k, v in lj.items() if k != "rows"} if isinstance(lj, dict) else {}
    # Corpora written before code_fix §1 have no convention field: legacy_plus180.
    corpus_conv = settings.get("dihedral_convention", "legacy_plus180")
    if corpus_conv != conv:
        raise SystemExit(f"{corpus_dir} is a {corpus_conv!r} corpus, the checkpoint expects {conv!r}")
    if "chain_break_cutoff" in settings:
        a = resolve_chain_break_cutoff(settings["chain_break_cutoff"], conv)
        if (a is None) != (cutoff is None) or (a is not None and abs(float(a) - float(cutoff)) > 1e-6):
            raise SystemExit(f"{corpus_dir}: corpus chain_break_cutoff={a}, checkpoint {cutoff}")
    desc_path = corpus_dir / with_convention("esm3desc_dihedral_K16_P10.npy", conv)
    if not desc_path.exists():
        raise SystemExit(f"descriptor file {desc_path} missing")
    desc = np.load(desc_path, mmap_mode="r")
    total = sum(int(r["L"]) for r in rows)
    if desc.shape[0] != total:
        raise SystemExit(f"{desc_path} has {desc.shape[0]} rows, layout.json {total}")
    return rows, settings, desc, desc_path


def count_corpus_usage(ens, corpus_dir: Path, split: str, max_rows: int, device: str,
                       bs: int = 16384) -> dict:
    """Per-code level-1 usage over the training corpus rows of `split`.

    FIXED(code_fix P20, 2026-10-16): replaces `tokenize_corpus` (mdCATH-div on the
    fly + an old MISATO token file) with the model's own training corpus.
    """
    conv, cutoff = checkpoint_settings(ens)
    rows, settings, desc, desc_path = open_corpus(corpus_dir, conv, cutoff)
    total = desc.shape[0]
    if split == "all":
        idx = np.arange(total)
    else:
        sp = json.loads((corpus_dir / "splits.json").read_text())
        idx = np.sort(np.asarray(sp[split], dtype=np.int64))
    if max_rows:
        idx = idx[:max_rows]
    corpus_of_row = np.repeat(np.array([r.get("corpus", "?") for r in rows]),
                              [int(r["L"]) for r in rows])
    M0 = int(ens.config["codebook_sizes"][0]) if "codebook_sizes" in ens.config \
        else int(ens.config["codebook_size"])
    mu = ens.mean.astype(np.float32); std = ens.std.astype(np.float32)
    counts = np.zeros(M0, dtype=np.int64)
    per_corpus = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            ii = idx[s:s + bs]
            # contiguous ranges read as slices (fast for --split all)
            x = desc[ii[0]:ii[-1] + 1] if ii[-1] - ii[0] + 1 == len(ii) else desc[ii]
            x = (np.asarray(x, dtype=np.float32) - mu) / std
            tok = ens.model.encode(torch.as_tensor(x, device=device)).cpu().numpy()
            counts += np.bincount(tok, minlength=M0)
            for c in np.unique(corpus_of_row[ii]):
                per_corpus[str(c)] = per_corpus.get(str(c), 0) + int((corpus_of_row[ii] == c).sum())
            if (s // bs) % 50 == 0:
                print(f"  {s + len(ii):,}/{len(idx):,} rows ({time.time() - t0:.0f}s)", flush=True)
    starts = np.array([int(r["start"]) if "start" in r else -1 for r in rows])
    if (starts >= 0).all():
        sel = np.zeros(total, dtype=bool); sel[idx] = True
        n_sys = {}
        for r, st in zip(rows, starts):
            if sel[st]:
                n_sys[r.get("corpus", "?")] = n_sys.get(r.get("corpus", "?"), 0) + 1
    else:
        n_sys = None
    return {"counts": counts, "residues_per_corpus": per_corpus, "systems_per_corpus": n_sys,
            "descriptor_file": desc_path.name, "corpus_settings": settings, "n_rows": int(len(idx))}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR),
                    help="training corpus dir with layout.json, splits.json and the descriptor file")
    # B.8 fix: default 'train' (was 'all' which included test; train matches the reported figure)
    ap.add_argument("--split", choices=("all", "train", "val", "test"), default="train")
    ap.add_argument("--out", default=str(DEFAULT_OUT_PNG))
    ap.add_argument("--top-n-label", type=int, default=12,
                     help="Annotate this many of the most-used tokens with their id")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0,
                     help="If >0, count only the first N residue rows (smoke test)")
    ap.add_argument("--counts-cache", default=None,
                     help="If set, load/save token counts JSON here")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_png = Path(args.out)
    out_pdf = out_png.with_suffix(".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model_dir(args.model_dir)
    ens = load_model(model_dir, device=args.device)
    cfg = ens.config
    conv, cutoff = checkpoint_settings(ens)
    sha = ckpt_hash(model_dir)
    cb = get_codebook(ens, level=0)
    M, d = cb.shape
    print(f"Codebook L0: {cb.shape}, descriptor={cfg_get(cfg, 'descriptor_mode', 'esm3desc_dihedral')}, "
          f"P={cfg['num_prototypes']}, k={cfg_get(cfg, 'k', 16)}, convention={conv}, sha={sha}")

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = PROJECT_ROOT / corpus_dir
    prov = {"model_dir": model_dir.name, "ckpt_sha256_12": sha, "dihedral_convention": conv,
            "chain_break_cutoff": cutoff, "corpus_dir": str(args.corpus_dir), "split": args.split,
            "max_rows": args.max_rows}
    counts_cache = Path(args.counts_cache) if args.counts_cache else None
    if counts_cache and counts_cache.exists():
        cc = json.loads(counts_cache.read_text())
        # FIXED(code_fix P20, 2026-10-16): refuse counts of another checkpoint / corpus.
        # B.8 fix: also check corpus_dir in cache (was omitted)
        for k in ("ckpt_sha256_12", "dihedral_convention", "split", "max_rows", "corpus_dir"):
            if cc.get(k) != prov[k]:
                raise SystemExit(f"{counts_cache}: {k}={cc.get(k)!r} != {prov[k]!r}; delete it")
        counts = np.array(cc["counts"], dtype=np.int64)
        summary = cc
        print(f"  loaded counts from {counts_cache}")
    else:
        print(f"Tokenizing the training corpus {corpus_dir} ({args.split}) for per-code usage...")
        summary = count_corpus_usage(ens, corpus_dir, args.split, args.max_rows, args.device)
        counts = summary["counts"]
        summary = {**prov, **{k: v for k, v in summary.items() if k != "counts"}}
        if counts_cache:
            counts_cache.parent.mkdir(parents=True, exist_ok=True)
            counts_cache.write_text(json.dumps({**summary, "M": int(M), "counts": counts.tolist()},
                                               indent=2))
            print(f"  saved counts to {counts_cache}")

    used = counts > 0
    if used.any():
        cu = counts[used]
        print(f"usage: {int(counts.sum()):,} assignments, utilisation {used.sum()}/{M} = "
              f"{used.mean():.2%}, top/mean = {cu.max() / cu.mean():.2f}, "
              f"p99/p50 = {np.percentile(cu, 99) / np.percentile(cu, 50):.2f}, "
              f"residues per corpus {summary.get('residues_per_corpus')}, "
              f"systems per corpus {summary.get('systems_per_corpus')}")

    # t-SNE
    from sklearn.manifold import TSNE
    print(f"Running t-SNE (perplexity={args.perplexity}, seed={args.seed})...")
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                random_state=args.seed, metric="cosine").fit_transform(cb)
    print(f"  t-SNE done in {time.time() - t0:.0f}s")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=140)
    log_use = np.log10(counts + 1)
    sc = ax.scatter(emb[~used, 0], emb[~used, 1], c="lightgray", s=8,
                      label=f"unused ({(~used).sum()})", zorder=1)
    sc2 = ax.scatter(emb[used, 0], emb[used, 1], c=log_use[used], s=14,
                      cmap="viridis", zorder=2)

    if args.top_n_label > 0:
        top_idx = np.argsort(-counts)[:args.top_n_label]
        for tk in top_idx:
            ax.annotate(str(tk), emb[tk], fontsize=8, weight="bold",
                          color="crimson", ha="center", va="center")

    cbar = plt.colorbar(sc2, ax=ax)
    cbar.set_label(r"$\log_{10}(1 + \mathrm{usage\ count})$")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    util = float((counts > 0).sum()) / M
    # B.8 fix: add λ, split and model dir to the title
    lam = cfg.get('args', {}).get('consistency_mse_weight', '?')
    ax.set_title(
        f"$L_1$ codebook t-SNE  ($M = {M}$, $d = {d}$, $\\lambda = {lam}$)\n"
        f"model: {model_dir.name}, split={args.split}\n"
        f"utilization {used.sum()}/{M} = {util:.1%}",
        fontsize=10)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"\nWrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
