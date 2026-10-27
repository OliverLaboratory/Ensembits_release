"""Per-residue export of the tokenizer's training corpus: level-1 token, latent distance d_z and s_1.

What it does
------------
  1. Opens the model's training corpus (BioEmu-CATH + MISATO descriptor cache written by
     `data_prep/build_bioemu_misato_descriptor.py`: layout.json, splits.json and the
     convention-named descriptor file; default REPO_ROOT/output/bioemu_cath_misato_e0/
     cached_desc) and keeps the residue rows of one split (`--split train`, default: the
     rows the tokenizer was trained on).
  2. s_1 of every kept residue, with the definition of the token ANOVA
     (`token_anova.per_residue_pca_features`, imported unchanged): the leading singular
     value of the (P = 10, 3) C-alpha trajectory of the residue after Kabsch-aligning every
     frame on the 10 A C-alpha ball around it in frame 0. The C-alpha come from the same
     real-backbone files and the same ten frames the descriptor was built from
     (layout.json `bioemu_bb` / `misato_bb`, `bb_10[:, :, 1]`), so tokens and s_1 describe
     the same frames. A residue with fewer than 4 C-alpha in its ball keeps the
     placeholder s_1 = 0 and is flagged (`s1_valid = False`), as in the ANOVA. The s_1
     table does not depend on the checkpoint and is cached once
     (`<out-root>/s1_<split>__<layout sha12>.npz` + provenance JSON).
  3. For every checkpoint in `--variants`: the encoder latent z = encoder(x) of each row
     (P = 10 frames, the checkpoint's standardisation), the level-1 code
     c = argmin_i ||z - C_i|| (the code `model.encode` returns; checked on every batch)
     and d_z = ||z - C_c||, the Euclidean distance between the pre-quantisation latent and
     the assigned codebook vector (the "encoder-latent distance to the codebook centroid"
     of `appendix.tex` app:viz). The descriptor file is read once for all variants.
  4. Writes, per variant, `<out-root>/<variant>/train_residues.npz` with one row per
     residue: row (global descriptor row), token, d_z, s1, s1_valid, corpus_id
     (0 = bioemu_cath, 1 = misato), system_id, res (0-based residue index in the system),
     plus the string tables `systems`, `pids`, `corpora`; and a provenance JSON
     (checkpoint sha, corpus files, split, row counts, per-code usage summary).

Paper
-----
  * `2_results.tex` \\subsection{Inspecting the learned alphabet} (\\label{val:pics}),
    Figures fig:token1699, fig:token1063, and `appendix.tex` \\section{Token
    visualizations} (\\label{app:viz}), Figure fig:5x5tokens: the exemplar ranking by
    d_z and the token motion amplitude <s_1>. Definition of s_1: `appendix.tex`
    \\label{app:token_anova} \\paragraph{Setup.}.
  * Authors' decision A11 (b) of `camera_ready/final_fix_list.md` (2026-10-19): exemplars
    from the training corpus, regenerated with the retrained model.

Origin
------
New file (2026-10-19, final_fix_list A11). Replaces the research cache
`experiments/results/shared_states/token_pca_residues_aligned.npz` (mdCATH-div, old
192-D flagship, legacy +180 deg descriptor) read by the research renderers
`experiments/shared_states_v2/render_token_knn_pymol.py` / `render_token_grid_pymol.py`.

Used by the reported model: yes (input of `render_token_exemplars.py`).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/bioemu_cath_misato_e0/cached_desc/{layout.json, splits.json,
          esm3desc_dihedral_iupac_K16_P10.npy}
          data/cached_descriptors/{bioemu_cath_real_bb, misato_real_bb_kabsch}/<system>.npz
          <variant>/{config.json, best.pt, stats.npz}
  writes  output/token_exemplars/s1_train__<sha12>.npz (+ .provenance.json)
          output/token_exemplars/<variant>/train_residues.npz (+ .provenance.json)

How to run
----------
    python downstream/analyses/token_exemplar_export.py \\
        --variants output/ensembits_bm_lam10_rv \\
        --device cuda:0 --gpu-mem-gb 4 --workers 48
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT  # noqa: E402

DEFAULT_CORPUS_DIR = REPO_ROOT / "output" / "bioemu_cath_misato_e0" / "cached_desc"
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "token_exemplars"
# FIXED(recon2 A.7, 2026-10-25): default switched to the chosen λ = 10 _rv run.
DEFAULT_VARIANTS = [REPO_ROOT / "output" / "ensembits_bm_lam10_rv"]
CORPUS_IDS = {"bioemu_cath": 0, "misato": 1}
S1_DEFINITION = ("per_residue_pca_features(aligned=True, ball_radius=10.0) on bb_10[:, :, 1] "
                 "(the descriptor's 10 frames); placeholder s1 = 0 when < 4 CA in the ball")


def file_sha256(path, n=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def open_layout(corpus_dir: Path):
    """(rows, settings) of layout.json; each row gets its global start (cumulative L)."""
    lj = json.loads((corpus_dir / "layout.json").read_text())
    rows = lj["rows"]
    settings = {k: v for k, v in lj.items() if k != "rows"}
    cur = 0
    for r in rows:
        if "start" in r and int(r["start"]) != cur:
            raise SystemExit(f"layout.json: row {r['system']} start {r['start']} != cumulative {cur}")
        r["start"] = cur
        cur += int(r["L"])
    return rows, settings, cur


def system_bb_path(row: dict, settings: dict) -> Path:
    """Real-backbone file of one layout row (the file the descriptor was built from)."""
    if row["corpus"] == "bioemu_cath":
        return Path(settings["bioemu_bb"]) / f"{row['system']}.npz"
    if row["corpus"] == "misato":
        return Path(settings["misato_bb"]) / f"{row['pid']}.npz"
    raise ValueError(f"unknown corpus {row['corpus']!r}")


def _s1_one(args):
    """(row index, s1 (L,), s1_valid (L,)) of one system."""
    i, path, L = args
    from token_anova import per_residue_pca_features
    bb = np.load(path, allow_pickle=False)["bb_10"]
    if bb.shape[1] != L or bb.shape[0] != 10:
        raise RuntimeError(f"{path}: bb_10 shape {bb.shape}, layout L = {L}")
    ca = np.ascontiguousarray(bb[:, :, 1, :], dtype=np.float32)
    feats = per_residue_pca_features(ca, aligned=True)
    # placeholder rule of token_anova (fewer than 4 C-alpha within 10 A in frame 0)
    d = np.linalg.norm(ca[0][:, None, :] - ca[0][None, :, :], axis=-1)
    valid = (d <= 10.0).sum(1) >= 4
    return i, feats[:, 0].astype(np.float32), valid


def s1_table(rows, settings, sel_rows: np.ndarray, out_root: Path, split: str, workers: int,
             layout_sha: str):
    """s_1 and validity of every residue of the systems touched by `sel_rows` (cached)."""
    path = out_root / f"s1_{split}__{layout_sha[:12]}.npz"
    prov_path = path.with_name(path.name + ".provenance.json")
    if path.exists() and prov_path.exists():
        prov = json.loads(prov_path.read_text())
        if prov.get("layout_sha256") == layout_sha and prov.get("definition") == S1_DEFINITION \
                and prov.get("sha256") == file_sha256(path):
            z = np.load(path)
            print(f"[s1] reuse {path}", flush=True)
            return z["s1"], z["s1_valid"], {"file": str(path), **prov}
        raise SystemExit(f"{path} does not match its provenance; delete it")
    starts = np.array([r["start"] for r in rows]); Ls = np.array([int(r["L"]) for r in rows])
    sys_of_row = np.searchsorted(starts, sel_rows, side="right") - 1
    systems = np.unique(sys_of_row)
    jobs = [(int(i), str(system_bb_path(rows[i], settings)), int(Ls[i])) for i in systems]
    # large systems first for load balance
    jobs.sort(key=lambda j: -j[2])
    s1_full, ok_full = {}, {}
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(max(1, workers)) as pool:
        for n, (i, s1, ok) in enumerate(pool.imap_unordered(_s1_one, jobs, chunksize=4)):
            s1_full[i] = s1; ok_full[i] = ok
            if (n + 1) % 2000 == 0:
                print(f"[s1] {n + 1}/{len(jobs)} systems ({time.time() - t0:.0f}s)", flush=True)
    s1 = np.zeros(len(sel_rows), np.float32); ok = np.zeros(len(sel_rows), bool)
    res = sel_rows - starts[sys_of_row]
    for i in systems:
        m = sys_of_row == i
        s1[m] = s1_full[i][res[m]]; ok[m] = ok_full[i][res[m]]
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp.npz")
    np.savez(tmp, s1=s1, s1_valid=ok, rows=sel_rows)
    os.replace(tmp, path)
    prov = {"layout_sha256": layout_sha, "definition": S1_DEFINITION, "split": split,
            "n_rows": int(len(sel_rows)), "n_systems": int(len(systems)),
            "n_placeholder": int((~ok).sum()), "seconds": round(time.time() - t0, 1),
            "sha256": file_sha256(path), "builder": "downstream/analyses/token_exemplar_export.py"}
    prov_path.write_text(json.dumps(prov, indent=1))
    print(f"[s1] {len(sel_rows):,} residues of {len(systems):,} systems in "
          f"{time.time() - t0:.0f}s -> {path}", flush=True)
    return s1, ok, {"file": str(path), **prov}


def row_windows(idx: np.ndarray, window: int):
    """Split sorted row indices into windows of the global row space: yields
    (a, b, take) with take = the selected rows in [a, b) as offsets from a. One read of
    desc[a:b] per window (non-selected rows in the window are read and dropped)."""
    if len(idx) == 0:
        return
    lo = int(idx[0])
    hi = int(idx[-1]) + 1
    for a in range(lo, hi, window):
        b = min(a + window, hi)
        i0, i1 = np.searchsorted(idx, [a, b])
        if i1 > i0:
            yield a, b, idx[i0:i1] - a


def encode_latent(ens, x, torch):
    """(token (N,), d_z (N,)) of standardised descriptor rows x (N, P, 256).

    token = argmin ||z - C|| over the level-1 codebook (the VQ rule of
    ensembits/tokenizer/vqvae.py VectorQuantizer.forward); d_z = that minimum distance.
    """
    z = ens.model.encoder(x)
    cb = ens.model.vq_levels[0].codebook.weight
    d = torch.cdist(z.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
    dz, tok = d.min(dim=-1)
    return tok, dz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", default=[str(v) for v in DEFAULT_VARIANTS])
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    ap.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0,
                    help="cap of this process's CUDA memory (torch per-process fraction)")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=32, help="processes for the s_1 table")
    ap.add_argument("--max-rows", type=int, default=0, help="smoke test: first N rows only")
    args = ap.parse_args(argv)

    import torch
    from ensembits.featurize import load_model
    from ensembits.descriptor import with_convention
    from _common import resolve_model_dir, ckpt_hash, checkpoint_settings

    corpus_dir = Path(args.corpus_dir)
    out_root = Path(args.out_root)
    rows, settings, total = open_layout(corpus_dir)
    layout_sha = file_sha256(corpus_dir / "layout.json")
    if args.split == "all":
        sel = np.arange(total, dtype=np.int64)
    else:
        sp = json.loads((corpus_dir / "splits.json").read_text())
        sel = np.sort(np.asarray(sp[args.split], dtype=np.int64))
    if args.max_rows:
        sel = sel[:args.max_rows]
    split_tag = args.split if not args.max_rows else f"{args.split}_first{args.max_rows}"
    print(f"[export] corpus {corpus_dir}: {total:,} rows, split {args.split}: {len(sel):,}",
          flush=True)

    if args.device.startswith("cuda"):
        dev = torch.device(args.device)
        frac = args.gpu_mem_gb * (1 << 30) / torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, frac), dev)

    # checkpoints (all must expect the corpus convention)
    models = []
    conv_corpus = settings.get("dihedral_convention", "legacy_plus180")
    for v in args.variants:
        md = resolve_model_dir(v)
        ens = load_model(md, device=args.device)
        conv, cutoff = checkpoint_settings(ens)
        if conv != conv_corpus:
            raise SystemExit(f"{md}: convention {conv!r}, corpus {conv_corpus!r}")
        if cutoff is not None and abs(float(cutoff) - float(settings["chain_break_cutoff"])) > 1e-6:
            raise SystemExit(f"{md}: chain_break_cutoff {cutoff}, corpus {settings['chain_break_cutoff']}")
        models.append((md, ens, ckpt_hash(md)))
        print(f"[export] {md.name}: sha {models[-1][2]}, conv {conv}", flush=True)
    desc_path = corpus_dir / with_convention("esm3desc_dihedral_K16_P10.npy", conv_corpus)
    desc = np.load(desc_path, mmap_mode="r")
    if desc.shape[0] != total:
        raise SystemExit(f"{desc_path}: {desc.shape[0]} rows, layout {total}")

    # residue identity columns
    starts = np.array([r["start"] for r in rows])
    sys_id = (np.searchsorted(starts, sel, side="right") - 1).astype(np.int32)
    res = (sel - starts[sys_id]).astype(np.int32)
    corpora = sorted(CORPUS_IDS, key=CORPUS_IDS.get)
    corpus_id = np.array([CORPUS_IDS[rows[i]["corpus"]] for i in range(len(rows))], np.int8)[sys_id]

    s1, s1_ok, s1_rec = s1_table(rows, settings, sel, out_root, split_tag, args.workers, layout_sha)

    # tokens and d_z of every variant, one pass over the descriptor file
    n = len(sel)
    toks = [np.zeros(n, np.int16) for _ in models]
    dzs = [np.zeros(n, np.float32) for _ in models]
    pos = 0
    t0 = time.time()
    n_checked = 0
    with torch.no_grad():
        for a, b, take in row_windows(sel, args.batch):
            x = np.asarray(desc[a:b], dtype=np.float32)[take]
            nb = len(take)
            for k, (md, ens, _) in enumerate(models):
                xs = (x - ens.mean.astype(np.float32)) / ens.std.astype(np.float32)
                xt = torch.as_tensor(xs, device=args.device)
                tok, dz = encode_latent(ens, xt, torch)
                if n_checked < 20:          # the code equals model.encode on the first batches
                    ref = ens.model.encode(xt)
                    if not torch.equal(ref, tok):
                        raise SystemExit(f"{md.name}: argmin code != model.encode")
                toks[k][pos:pos + nb] = tok.cpu().numpy().astype(np.int16)
                dzs[k][pos:pos + nb] = dz.cpu().numpy()
            n_checked += 1
            pos += nb
            if n_checked % 50 == 0:
                el = time.time() - t0
                print(f"[encode] {pos:,}/{n:,} rows ({el:.0f}s, eta {el / pos * (n - pos):.0f}s)",
                      flush=True)
    assert pos == n

    systems = np.array([r["system"] for r in rows])
    pids = np.array([r["pid"] for r in rows])
    for k, (md, ens, sha) in enumerate(models):
        out_dir = out_root / md.name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{split_tag}_residues.npz"
        tmp = path.with_name("." + path.name + ".tmp.npz")
        np.savez(tmp, row=sel, token=toks[k], d_z=dzs[k], s1=s1, s1_valid=s1_ok,
                 corpus_id=corpus_id, system_id=sys_id, res=res,
                 systems=systems, pids=pids, corpora=np.array(corpora))
        os.replace(tmp, path)
        M = int(ens.config["codebook_sizes"][0])
        counts = np.bincount(toks[k].astype(np.int64), minlength=M)
        prov = {
            "variant": md.name, "model_dir": str(md), "ckpt_sha12": sha,
            "lambda_consistency_mse_weight": (ens.config.get("args") or {}).get(
                "consistency_mse_weight", ens.config.get("consistency_mse_weight")),
            "dihedral_convention": conv_corpus, "corpus_dir": str(corpus_dir),
            "descriptor_file": desc_path.name, "layout_sha256": layout_sha,
            "split": args.split, "max_rows": args.max_rows, "n_rows": n,
            "n_systems": int(len(np.unique(sys_id))),
            "residues_per_corpus": {c: int((corpus_id == CORPUS_IDS[c]).sum()) for c in corpora},
            "token_rule": "argmin_i ||encoder(x) - C_i|| on the level-1 codebook "
                          "(checked equal to model.encode on the first 20 batches)",
            "d_z": "||encoder(x) - C_token|| (Euclidean, latent dim 128)",
            "s1": s1_rec, "n_s1_placeholder": int((~s1_ok).sum()),
            "usage": {"codes_used": int((counts > 0).sum()), "M": M,
                      "top_over_mean": float(counts.max() / counts[counts > 0].mean())},
            "seconds_encode": round(time.time() - t0, 1),
            "sha256": file_sha256(path),
            "builder": "downstream/analyses/token_exemplar_export.py",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.with_name(path.name + ".provenance.json").write_text(json.dumps(prov, indent=1))
        print(f"[export] {md.name}: {n:,} rows, {prov['usage']['codes_used']} codes used -> {path}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
