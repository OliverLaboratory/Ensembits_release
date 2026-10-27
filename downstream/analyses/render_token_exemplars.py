"""Token exemplar figures: two single-token panels and a 5 x 5 token grid, rendered with PyMOL.

What it does
------------
Reads the per-residue export of the training corpus written by
`token_exemplar_export.py` (`output/token_exemplars/<variant>/train_residues.npz`: level-1
token, d_z, s_1, corpus, system, residue) and, for one checkpoint:

  1. Token statistics over the training split: usage u(c) (training residues assigned to
     code c), <s_1>(c) (mean s_1 of those residues, s_1 placeholders excluded) and the
     background (all training residues).
  2. Exemplar walk of a token (`exemplars`): its training residues in increasing d_z; a
     residue is accepted when (a) its s_1 is not a placeholder, (b) it is at least
     `--min-terminal-distance` residues from the end of its chain piece, (c) its all-atom
     trajectory is available, (d) its pid (CATH domain / PDB entry) is not already used and
     (e) the sequence identity of its chain to every accepted chain is below
     `--max-seq-identity` (`token_exemplars_lib.seq_identity`). The walk stops at
     `--n-eval` accepted residues (or `--max-walk` candidates). Panels show the first ones,
     left to right by increasing d_z.
  3. Geometry of an exemplar: the sixteen canonical neighbours of the anchor (the anchor
     and the 15 residues most often in its descriptor kNN lists over the ten descriptor
     frames; `token_exemplars_lib.canonical_knn`), their frame-averaged C-alpha distances
     d to the anchor and sequence offsets |i - j| (same chain piece only).
  4. Compactness of a token, from its `--n-eval` exemplars E:
         kappa(c) = mean_E(dbar) + std_E(dbar) + 0.05 * mean_rank std_E(|i-j| profile)
     dbar = mean d over the 15 non-self canonical neighbours of one exemplar; |i-j| profile =
     the 15 sorted offsets, capped at 50 (a neighbour on another chain piece counts as 50).
     Lower kappa = tighter neighbourhoods that recur across the exemplars. (Same formula as
     the research renderer's pick_token, with the paper's neighbour set and the d_z-ranked
     exemplars instead of a random sample.)
  5. Token choice (the rule is written to selection.json):
       * stationary panel (analogue of fig:token1699): codes with u >= --min-count and
         <s_1> below the median <s_1> of those codes; the --top-n-pool most used; lowest
         kappa;
       * flexible panel (analogue of fig:token1063): same with <s_1> >= the upper quartile;
       * grid (fig:5x5tokens): the --grid-top-used most used codes with u >= --min-count
         (and --n-eval exemplars), without the two single-panel codes; --n-rows targets
         evenly spaced between the 5th and 95th percentile of their <s_1>; per target the
         lowest-kappa code within half a step (else the nearest unused <s_1>); rows by
         <s_1>. (Stratifying first keeps the rows spread over <s_1>: a compactness
         pre-filter keeps mostly stationary codes, because kappa grows with motion.)
  6. Rendering (`render_panels`): for each exemplar, `--n-frames` frames are loaded with the
     all-atom loaders (`--frames descriptor`: frames of the ten descriptor frames, evenly
     spaced over their stored order; `--frames even`: evenly spaced over the whole
     trajectory); every frame is Kabsch-superposed on the first using the C-alpha of all
     residues except the 3-mers (j-1, j, j+1) of the sixteen canonical neighbours, the same
     rigid transform applied to all atoms; the panel is then expressed in the anchor's
     backbone frame (frame 1), so all panels share the anchor's orientation. PyMOL draws the
     residues within --context-radius of the anchor (frame 1) as a grey transparent
     cartoon, the 3-mers of every frame as cartoons coloured by frame (plasma) and the
     anchor C-alpha as a red sphere. One PyMOL session per figure (per grid row): the view
     is oriented on the union of the panels' 3-mers, each panel zoomed on its own 3-mers.
     Dashed-bond artefacts of the old renders are removed: cartoon_gap_cutoff 0 (no dashes
     across residue-number gaps) and a chain id per chain piece (no cartoon across chain
     breaks or between MISATO chains).
  7. Writes, in --out-dir (default output/figures/tokens/<variant>/):
       token_stationary_<id>.{png,pdf}, token_flexible_<id>.{png,pdf}, 5x5tokens.{png,pdf},
       each with <name>.args.json (all arguments, the code hash, checkpoint, selection rule,
       panels with pid / residue / d_z / s_1 / neighbours / frames / alignment RMSD);
       selection.json (scores of every evaluated code); caption_numbers.json (token id,
       usage, d_z, <s_1> of exemplars vs token vs background, |i - j| statistics);
       work/ (PDB, PML and per-panel PNG files).

Paper
-----
  * `2_results.tex` \\subsection{Inspecting the learned alphabet} (\\label{val:pics}),
    Figures fig:token1699 (stationary token) and fig:token1063 (flexible token).
  * `appendix.tex` \\section{Token visualizations} (\\label{app:viz}) \\subsection{Tokens},
    Figure fig:5x5tokens: d_z ranking, sixteen canonical kNN with no sequence gap, 3-mers,
    Kabsch on all C-alpha minus the 3-mers, five overlaid frames, rows by <s_1>, columns by
    d_z.
  * Authors' decision A11 (b) of `camera_ready/final_fix_list.md` (2026-10-19): exemplars
    from the training corpus (BioEmu-CATH + MISATO), regenerated with the retrained model.
  Text to adjust in the paper (not edited here): the exemplars come from BioEmu-CATH and
  MISATO (not mdCATH); with `--frames descriptor` (default) the overlaid frames are five
  of the ten descriptor frames (not "five evenly-spaced frames"); the selection rule of
  step 5; the token ids and numbers of caption_numbers.json.

Origin
------
New file (2026-10-19, final_fix_list A11). Ports, with the paper's settings, the research
scripts `experiments/shared_states_v2/render_token_knn_pymol.py` (pick_token,
select_occurrences, prep_panel_allatom, build_session_pml_allatom, compose_figure) and
`experiments/shared_states_v2/render_token_grid_pymol.py` (pick_tokens_spanning_s1,
compose_grid, _row_crop_box). Differences: training-corpus exemplars from the camera-ready
export (was mdCATH-div / MISATO with the 192-D flagship and the legacy descriptor); K = 16
canonical kNN with no sequence gap on C-alpha (was k = 3, gap 3, 3Di virtual centres);
alignment on all C-alpha minus the 3-mers (was a 10 A anchor ball); deterministic
d_z-ranked exemplars for kappa (was a random sample); distinct proteins by pid and
sequence identity (was pid only); cross-panel orientation by the anchor's backbone frame
(was Kabsch on anchor + 3 NN); <s_1> = mean (the grid used the 90th percentile); the dashed
bond fixes of step 6; every argument logged next to each figure.

Used by the reported model: yes (figures fig:token1699, fig:token1063, fig:5x5tokens).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/token_exemplars/<variant>/train_residues.npz (+ .provenance.json)
          output/bioemu_cath_misato_e0/cached_desc/layout.json
          data/cached_descriptors/{bioemu_cath_real_bb, bioemu_cath_extras, misato_real_bb_kabsch}
          data/misato/MD.hdf5, data/reviewer_3_special_corpus/bioemu_cath/extracted
  writes  output/figures/tokens/<variant>/ (see step 7)

How to run
----------
    ENSEMBITS_PYMOL=<pymol executable> python downstream/analyses/render_token_exemplars.py \\
        --variant output/ensembits_bm_lam1
    # selection only (no PyMOL): --no-render
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT  # noqa: E402
import token_exemplars_lib as TL  # noqa: E402

DEFAULT_EXPORT_ROOT = REPO_ROOT / "output" / "token_exemplars"
DEFAULT_FIG_ROOT = REPO_ROOT / "output" / "figures" / "tokens"
DEFAULT_CORPUS_DIR = REPO_ROOT / "output" / "bioemu_cath_misato_e0" / "cached_desc"
SEQ_CAP = 50          # |i - j| cap (and value for a neighbour on another chain piece) in kappa

SELECTION_RULE = (
    "Exemplars of a code: its training residues in increasing d_z, accepted if s_1 is not a "
    "placeholder, >= {min_term} residues from the end of the chain piece, all-atom trajectory "
    "available, pid not yet used and chain sequence identity < {max_id:.2f} to every accepted "
    "chain; first {n_eval} accepted (walk <= {max_walk}). kappa(c) = mean(dbar) + std(dbar) + "
    "0.05 * mean_rank std(sorted |i-j| of the 15 non-self canonical neighbours, capped at "
    "{cap}, other chain = {cap}) over those exemplars; dbar = mean frame-averaged CA distance "
    "of the 15 non-self canonical neighbours to the anchor. Eligible codes: training usage >= "
    "{min_count}. Stationary panel: eligible codes with <s1> < median(<s1> of eligible), the "
    "{top_pool} most used, lowest kappa. Flexible panel: eligible codes with <s1> >= upper "
    "quartile, the {top_pool} most used, lowest kappa. Grid: the {grid_top} most used eligible "
    "codes with {n_eval} exemplars, without the two single-panel codes; {n_rows} targets evenly spaced between the "
    "5th and 95th percentile of their <s1>; per target the lowest-kappa code within half a "
    "step (else the nearest <s1>); rows ordered by <s1>, columns by d_z."
)


def code_sha(n=12) -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(TL.__file__)):
        h.update(p.read_bytes())
    return h.hexdigest()[:n]


# ----------------------------------------------------------------------------- data

class Corpus:
    """The export of one variant plus lazy system / kNN caches."""

    def __init__(self, export_path: Path, corpus_dir: Path, need_misato_keys=True):
        z = np.load(export_path, allow_pickle=False)
        self.export_path = export_path
        self.prov = json.loads(Path(str(export_path) + ".provenance.json").read_text())
        self.token = z["token"].astype(np.int64)
        self.d_z = z["d_z"].astype(np.float64)
        self.s1 = z["s1"].astype(np.float64)
        self.s1_valid = z["s1_valid"].astype(bool)
        self.corpus_id = z["corpus_id"].astype(np.int64)
        self.system_id = z["system_id"].astype(np.int64)
        self.res = z["res"].astype(np.int64)
        self.systems = [str(s) for s in z["systems"]]
        self.pids = [str(s) for s in z["pids"]]
        self.corpora = [str(s) for s in z["corpora"]]
        lj = json.loads((corpus_dir / "layout.json").read_text())
        self.rows = lj["rows"]
        self.settings = {k: v for k, v in lj.items() if k != "rows"}
        if [r["system"] for r in self.rows] != self.systems:
            raise SystemExit("export systems != layout.json systems")
        self.M = int(self.prov.get("usage", {}).get("M", 2048))
        self.counts = np.bincount(self.token, minlength=self.M)
        s1v = np.where(self.s1_valid, self.s1, 0.0)
        n_valid = np.bincount(self.token, weights=self.s1_valid.astype(float), minlength=self.M)
        self.mean_s1 = np.full(self.M, np.nan)
        ok = n_valid > 0
        self.mean_s1[ok] = np.bincount(self.token, weights=s1v, minlength=self.M)[ok] / n_valid[ok]
        order = np.lexsort((self.d_z, self.token))       # by token, then d_z
        self.order = order
        self.tok_start = np.concatenate([[0], np.cumsum(self.counts)])
        self._sys = {}
        self._knn = {}
        self._seg = {}
        self.misato_keys = None
        if need_misato_keys and TL.MISATO_H5.exists():
            import h5py
            with h5py.File(TL.MISATO_H5, "r", swmr=True) as h:
                self.misato_keys = set(h.keys())

    def residues_of(self, c: int) -> np.ndarray:
        return self.order[self.tok_start[c]:self.tok_start[c + 1]]

    def system(self, sid: int) -> TL.System:
        if sid not in self._sys:
            self._sys[sid] = TL.load_system(self.rows[sid], self.settings)
        return self._sys[sid]

    def seg(self, sid: int) -> np.ndarray:
        if sid not in self._seg:
            s = self.system(sid)
            self._seg[sid] = TL.segments(s.bb[0], s.chain_index)
        return self._seg[sid]

    def knn(self, sid: int) -> np.ndarray:
        if sid not in self._knn:
            self._knn[sid] = TL.knn_edges_all_frames(self.system(sid).bb)
        return self._knn[sid]

    def background(self) -> dict:
        v = self.s1[self.s1_valid]
        return {"n_residues": int(len(self.s1)), "n_valid_s1": int(len(v)),
                "mean_s1": float(v.mean()), "median_s1": float(np.median(v)),
                "mean_d_z": float(self.d_z.mean()), "median_d_z": float(np.median(self.d_z))}


# ----------------------------------------------------------------------------- exemplars

def exemplar_geometry(C: Corpus, sid: int, r: int) -> dict:
    s = C.system(sid)
    seg = C.seg(sid)
    nbrs, cnt, md = TL.canonical_knn(C.knn(sid), s.bb[:, :, 1], r, TL.K_DESCRIPTOR)
    others = [(int(j), int(c), float(d)) for j, c, d in zip(nbrs, cnt, md) if int(j) != r]
    same = [seg[j] == seg[r] for j, _, _ in others]
    off = [abs(j - r) for j, _, _ in others]
    prof = sorted(min(o, SEQ_CAP) if sm else SEQ_CAP for o, sm in zip(off, same))
    same_off = [o for o, sm in zip(off, same) if sm]
    return {
        "nbrs": [int(j) for j in nbrs], "nbr_counts": [int(c) for c in cnt],
        "nbr_mean_dist": [round(float(d), 3) for d in md],
        "dbar": float(np.mean([d for _, _, d in others])),
        "seq_profile": prof,
        "n_same_piece": int(sum(same)), "n_other_piece": int(len(same) - sum(same)),
        "n_offset_le_15": int(sum(1 for o in same_off if o <= 15)),
        "offsets_same_piece": sorted(int(o) for o in same_off),
    }


def exemplars(C: Corpus, c: int, args, n: int | None = None) -> tuple[list, dict]:
    """First n accepted exemplars of code c (d_z order) and the walk statistics."""
    n = n or args.n_eval
    out, used_pid, seqs = [], set(), []
    stats = {"walked": 0, "rej_s1": 0, "rej_term": 0, "rej_avail": 0, "rej_pid": 0, "rej_identity": 0}
    for i in C.residues_of(c)[:args.max_walk]:
        stats["walked"] += 1
        sid = int(C.system_id[i]); r = int(C.res[i]); pid = C.pids[sid]
        if not C.s1_valid[i]:
            stats["rej_s1"] += 1; continue
        if pid in used_pid:
            stats["rej_pid"] += 1; continue
        corpus = C.corpora[C.corpus_id[i]]
        if not TL.allatom_available(corpus, C.systems[sid], pid, C.misato_keys):
            stats["rej_avail"] += 1; continue
        if TL.dist_to_segment_end(C.seg(sid))[r] < args.min_terminal_distance:
            stats["rej_term"] += 1; continue
        sq = TL.chain_sequence(C.system(sid), r)
        idn = [TL.seq_identity(sq, s) for s in seqs]
        if idn and max(idn) >= args.max_seq_identity:
            stats["rej_identity"] += 1; continue
        e = {"row": int(i), "system_id": sid, "system": C.systems[sid], "pid": pid, "corpus": corpus,
             "res": r, "d_z": float(C.d_z[i]), "s1": float(C.s1[i]),
             "max_identity_to_previous": float(max(idn)) if idn else None}
        e.update(exemplar_geometry(C, sid, r))
        out.append(e); used_pid.add(pid); seqs.append(sq)
        if len(out) >= n:
            break
    return out, stats


def kappa(ex: list) -> dict:
    d = np.array([e["dbar"] for e in ex])
    prof = np.array([e["seq_profile"] for e in ex], float)
    seq_std = float(prof.std(axis=0).mean()) if len(ex) > 1 else 0.0
    return {"kappa": float(d.mean() + d.std() + 0.05 * seq_std), "mean_dbar": float(d.mean()),
            "std_dbar": float(d.std()), "seq_std": seq_std}


def score_code(C: Corpus, c: int, args, cache: dict) -> dict | None:
    if c in cache:
        return cache[c]
    ex, st = exemplars(C, c, args)
    rec = None
    if len(ex) >= args.n_eval:
        rec = {"code": int(c), "usage": int(C.counts[c]), "mean_s1": float(C.mean_s1[c]),
               **kappa(ex), "walk": st, "exemplars": ex}
    cache[c] = rec
    return rec


def select_tokens(C: Corpus, args) -> dict:
    eligible = np.where(C.counts >= args.min_count)[0]
    eligible = eligible[np.isfinite(C.mean_s1[eligible])]
    ms = C.mean_s1[eligible]
    med, q3 = float(np.median(ms)), float(np.quantile(ms, 0.75))
    cache: dict = {}
    t0 = time.time()

    def pick_pool(mask, label):
        pool = eligible[mask]
        pool = pool[np.argsort(-C.counts[pool], kind="stable")][:args.top_n_pool]
        recs = [score_code(C, int(c), args, cache) for c in pool]
        recs = [r for r in recs if r is not None]
        if not recs:
            raise SystemExit(f"no {label} code with {args.n_eval} exemplars")
        best = min(recs, key=lambda r: r["kappa"])
        print(f"[select] {label}: code {best['code']} (usage {best['usage']}, <s1> "
              f"{best['mean_s1']:.2f}, kappa {best['kappa']:.2f}) of {len(recs)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return best, [r["code"] for r in recs]

    stat, stat_pool = pick_pool(ms < med, "stationary")
    flex, flex_pool = pick_pool(ms >= q3, "flexible")

    top = eligible[np.argsort(-C.counts[eligible], kind="stable")][:args.grid_top_used]
    recs = []
    for k, c in enumerate(top):
        r = score_code(C, int(c), args, cache)
        if r is not None:
            recs.append(r)
        if (k + 1) % 50 == 0:
            print(f"[select] grid: scored {k + 1}/{len(top)} ({time.time() - t0:.0f}s)", flush=True)
    # the codes of the two single-token figures are not repeated in the grid
    pool = [r for r in recs if r["code"] not in (stat["code"], flex["code"])]
    s = np.array([r["mean_s1"] for r in pool])
    lo, hi = float(np.percentile(s, 5)), float(np.percentile(s, 95))
    targets = np.linspace(lo, hi, args.n_rows)
    half = (hi - lo) / (2 * (args.n_rows - 1)) if args.n_rows > 1 else (hi - lo) / 2
    rows, used = [], set()
    for t in targets:
        band = [r for r in pool if abs(r["mean_s1"] - t) <= half and r["code"] not in used]
        if band:
            best = min(band, key=lambda r: r["kappa"]); how = "lowest kappa in band"
        else:
            best = min((r for r in pool if r["code"] not in used),
                       key=lambda r: abs(r["mean_s1"] - t)); how = "nearest <s1> (empty band)"
        used.add(best["code"])
        rows.append({**best, "target_s1": float(t), "chosen_by": how,
                     "n_in_band": len(band)})
    rows.sort(key=lambda r: r["mean_s1"])
    print(f"[select] grid rows: {[r['code'] for r in rows]} ({time.time() - t0:.0f}s)", flush=True)
    return {
        "rule": SELECTION_RULE.format(
            min_term=args.min_terminal_distance, max_id=args.max_seq_identity, n_eval=args.n_eval,
            max_walk=args.max_walk, cap=SEQ_CAP, min_count=args.min_count, top_pool=args.top_n_pool,
            grid_top=args.grid_top_used, n_rows=args.n_rows),
        "n_eligible": int(len(eligible)), "median_mean_s1": med, "q3_mean_s1": q3,
        "stationary": stat, "stationary_pool": stat_pool,
        "flexible": flex, "flexible_pool": flex_pool,
        "grid": rows, "grid_pool": [r["code"] for r in pool],
        "grid_targets": [float(t) for t in targets], "grid_half_step": float(half),
        "grid_s1_range": [lo, hi],
        "scores": sorted([{k: v for k, v in r.items() if k != "exemplars"}
                          for r in cache.values() if r is not None], key=lambda r: r["kappa"]),
        "n_scored": len(cache),
    }


# ----------------------------------------------------------------------------- panels

def prep_panel(C: Corpus, e: dict, args) -> dict:
    """All-atom frames of one exemplar, aligned and in the anchor's frame."""
    s = C.system(e["system_id"])
    seg = C.seg(e["system_id"])
    r = e["res"]
    frames, pos = TL.trajectory_frames(s, args.frames, args.n_frames)
    aa = TL.load_allatom(s, frames)
    ca = aa.ca()
    if pos is not None:                      # the loader must reproduce the descriptor frames
        dev = float(np.abs(ca - s.bb[pos][:, :, 1]).max())
        if dev > 1e-3:
            raise RuntimeError(f"{e['system']}: all-atom CA differ from bb_10 by {dev:.4f} A")
    tm = TL.threemer_residues(e["nbrs"], seg)
    align_idx = [i for i in range(s.L) if i not in set(tm)]
    trs = TL.align_to_first(ca, align_idx)
    coords = np.stack([TL.apply_transform(aa.coords[p], trs[p]) for p in range(len(trs))])
    ca_al = coords[:, aa.ca_atom, :]
    rms = [TL.rmsd(ca_al[p, align_idx], ca_al[0, align_idx]) for p in range(len(trs))]
    rms_tm = [TL.rmsd(ca_al[p, tm], ca_al[0, tm]) for p in range(len(trs))]
    # anchor frame (frame 1 after alignment): local coordinates
    names = aa.atom_name
    at_r = np.where(aa.atom_res == r)[0]
    get = lambda nm: coords[0, [a for a in at_r if names[a] == nm][0]]  # noqa: E731
    rot, org = TL.anchor_frame(get("N"), get("CA"), get("C"))
    coords = (coords - org) @ rot
    ca_al = coords[:, aa.ca_atom, :]
    d0 = np.linalg.norm(ca_al[0] - ca_al[0, r], axis=-1)
    bg = sorted(set(np.where(d0 <= args.context_radius)[0].tolist()) | set(tm) | {r})
    return {"exemplar": e, "coords": coords, "aa": aa, "seg": seg, "threemers": tm, "bg": bg,
            "frames": frames, "frame_positions": pos, "align_n_ca": len(align_idx),
            "align_rmsd": [round(x, 3) for x in rms], "threemer_rmsd_after_align": [round(x, 3) for x in rms_tm],
            "n_states": coords.shape[0]}


def _sel(ids: dict, residues) -> str:
    by_ch = {}
    for i in residues:
        ch, rs = ids[int(i)]
        by_ch.setdefault(ch, []).append(str(rs))
    return " or ".join(f"(chain {ch} and resi {'+'.join(v)})" for ch, v in by_ch.items())


def build_pml(panels, pdb_paths, png_paths, args) -> str:
    import matplotlib
    plasma = matplotlib.colormaps["plasma"]
    n_states = panels[0]["n_states"]
    L = ["bg_color white", "set ray_opaque_background, 1", "set ray_shadows, 0",
         "set ray_trace_mode, 1", "set antialias, 2", "set cartoon_fancy_helices, 1",
         "set cartoon_round_helices, 1", "set cartoon_smooth_loops, 1",
         # FIXED(final_fix_list A11, 2026-10-19): no dashed cartoon across residue gaps
         #   (the old renders drew dashes between the separate 3-mers and the cut background)
         "set cartoon_gap_cutoff, 0",
         f"viewport {args.ray_size}, {args.ray_size}", ""]
    for p in range(n_states):
        rgb = plasma(p / max(n_states - 1, 1))[:3]
        L.append(f"set_color st_col_{p}, [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]")
    for pi, (pn, pdb) in enumerate(zip(panels, pdb_paths)):
        ids = pn["ids"]
        fg = _sel(ids, pn["threemers"])
        anc = _sel(ids, [pn["exemplar"]["res"]])
        obj = f"prot{pi}"
        L += [f"# panel {pi}: {pn['exemplar']['pid']} res {pn['exemplar']['res']}",
              f"load {pdb}, {obj}", f"dss {obj}, 1",
              f"create p{pi}_bg, {obj}, 1, 1", f"color gray70, p{pi}_bg",
              f"set cartoon_transparency, {args.bg_transparency}, p{pi}_bg"]
        for s in range(n_states):
            L += [f"create p{pi}_fg_s{s + 1}, {obj} and ({fg}), {s + 1}, 1",
                  f"color st_col_{s}, p{pi}_fg_s{s + 1}",
                  f"set cartoon_transparency, {args.fg_transparency}, p{pi}_fg_s{s + 1}"]
        L += [f"create p{pi}_anchor, {obj} and ({anc}) and name CA, 1, 1",
              f"set sphere_scale, 1.05, p{pi}_anchor", f"color red, p{pi}_anchor",
              f"delete {obj}", ""]
    allfg = " or ".join(f"p{pi}_fg_s1" for pi in range(len(panels)))
    L += ["hide everything", f"show cartoon, {allfg}", f"orient ({allfg})",
          f"turn x, {args.turn_x}", f"turn y, {args.turn_y}", ""]
    for pi, png in enumerate(png_paths):
        sel = " or ".join([f"p{pi}_fg_s{s + 1}" for s in range(n_states)] + [f"p{pi}_anchor"])
        L += ["hide everything", f"show cartoon, p{pi}_bg"]
        L += [f"show cartoon, p{pi}_fg_s{s + 1}" for s in range(n_states)]
        L += [f"show spheres, p{pi}_anchor", f"zoom ({sel}), {args.zoom_buffer}",
              f"ray {args.ray_size}, {args.ray_size}", f"png {png}, dpi=200", ""]
    L.append("quit")
    return "\n".join(L) + "\n"


def render_panels(C: Corpus, exs: list, tag: str, work: Path, args) -> list:
    panels = [prep_panel(C, e, args) for e in exs]
    pdbs, pngs = [], []
    for pi, pn in enumerate(panels):
        e = pn["exemplar"]
        base = work / f"{tag}_p{pi}_{e['pid']}_r{e['res']}"
        pn["ids"] = TL.write_multistate_pdb(base.with_suffix(".pdb"), pn["coords"], pn["aa"].atom_res,
                                            pn["aa"].atom_name, pn["aa"].resname, pn["seg"], pn["bg"])
        pdbs.append(base.with_suffix(".pdb")); pngs.append(base.with_suffix(".png"))
    pml = work / f"{tag}_session.pml"
    pml.write_text(build_pml(panels, pdbs, pngs, args))
    proc = subprocess.run([args.pymol_bin, "-cq", str(pml)], capture_output=True, text=True)
    if proc.returncode != 0 or not all(p.exists() for p in pngs):
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise RuntimeError(f"PyMOL failed on {pml}")
    for pn, png in zip(panels, pngs):
        pn["png"] = png
    return panels


def panel_record(pn) -> dict:
    e = {k: v for k, v in pn["exemplar"].items()}
    e.update({"frames_loaded": [list(f) if isinstance(f, tuple) else f for f in pn["frames"]],
              "descriptor_frame_positions": pn["frame_positions"],
              "threemer_residues": pn["threemers"], "n_background_residues": len(pn["bg"]),
              "align_n_ca": pn["align_n_ca"], "align_rmsd_to_frame1": pn["align_rmsd"],
              "threemer_rmsd_to_frame1": pn["threemer_rmsd_after_align"], "png": str(pn["png"])})
    return e


# ----------------------------------------------------------------------------- composition

def _label(e) -> str:
    src = "BioEmu" if e["corpus"] == "bioemu_cath" else "MISATO"
    return f"{e['pid']} · res {e['res']} ({src})   $d_z$={e['d_z']:.2f}"


def compose_single(token: int, panels, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    n = len(panels); n_states = panels[0]["n_states"]
    fig = plt.figure(figsize=(5.0 * n, 5.4), dpi=160)
    gs = fig.add_gridspec(1, n, wspace=0.04)
    for c, pn in enumerate(panels):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(imread(pn["png"])); ax.axis("off")
        ax.set_title(_label(pn["exemplar"]), fontsize=11)
    fig.suptitle(title, fontsize=18, y=0.98)
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps["plasma"], norm=plt.Normalize(1, n_states))
    fig.subplots_adjust(right=0.94)
    cax = fig.add_axes([0.955, 0.20, 0.010, 0.55])
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(1, n_states + 1))
    cb.set_label("frame", fontsize=10); cb.ax.tick_params(labelsize=8)
    fig.savefig(out_png, bbox_inches="tight"); fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _row_crop_box(imgs, bg_threshold=0.97, pad=12):
    """Union non-background bounding box of a row of renders (copied from the research
    render_token_grid_pymol._row_crop_box, unchanged)."""
    H, W = imgs[0].shape[:2]
    y0_u, y1_u, x0_u, x1_u = H, 0, W, 0
    for img in imgs:
        rgb = img[..., :3] if img.shape[-1] >= 3 else img
        non_bg = rgb.mean(axis=-1) < bg_threshold
        if not non_bg.any():
            continue
        ra = non_bg.any(axis=1); ca = non_bg.any(axis=0)
        y0 = int(np.argmax(ra)); y1 = int(H - np.argmax(ra[::-1]))
        x0 = int(np.argmax(ca)); x1 = int(W - np.argmax(ca[::-1]))
        y0_u, x0_u = min(y0_u, y0), min(x0_u, x0); y1_u, x1_u = max(y1_u, y1), max(x1_u, x1)
    if y1_u <= y0_u or x1_u <= x0_u:
        return 0, H, 0, W
    return max(0, y0_u - pad), min(H, y1_u + pad), max(0, x0_u - pad), min(W, x1_u + pad)


def compose_grid(rows, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    n_rows = len(rows); n_cols = max(len(r["panels"]) for r in rows)
    n_states = rows[0]["panels"][0]["n_states"]
    imgs, aspects = [], []
    for row in rows:
        raw = [imread(pn["png"]) for pn in row["panels"]]
        y0, y1, x0, x1 = _row_crop_box(raw)
        cr = [im[y0:y1, x0:x1] for im in raw]
        imgs.append(cr); aspects.append(cr[0].shape[0] / cr[0].shape[1])
    pw = 4.5
    heights = [max(pw * a, 1.0) for a in aspects]
    fig = plt.figure(figsize=(pw * n_cols + 0.9, sum(heights) + 0.2), dpi=180)
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.04],
                          height_ratios=heights, wspace=0.02, hspace=0.18)
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps["plasma"], norm=plt.Normalize(1, n_states))
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c]); ax.axis("off")
            if c < len(imgs[r]):
                ax.imshow(imgs[r][c])
                e = row["panels"][c]["exemplar"]
                ax.set_title(f"{e['pid']}  res {e['res']}   $d_z$={e['d_z']:.2f}", fontsize=17)
            if c == 0:
                ax.text(-0.04, 0.5, f"Token {row['code']}\n$\\langle s_1\\rangle$={row['mean_s1']:.2f}",
                        transform=ax.transAxes, ha="right", va="center", fontsize=20)
        cax = fig.add_subplot(gs[r, n_cols])
        cb = fig.colorbar(sm, cax=cax, ticks=np.arange(1, n_states + 1))
        cb.set_label("frame", fontsize=14); cb.ax.tick_params(labelsize=12)
    fig.savefig(out_png, bbox_inches="tight"); fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- captions

def offset_stats(exs) -> dict:
    same = [o for e in exs for o in e["offsets_same_piece"]]
    tot = sum(e["n_same_piece"] + e["n_other_piece"] for e in exs)
    return {"n_neighbours": tot, "n_same_chain_piece": len(same),
            "n_other_chain_piece": int(sum(e["n_other_piece"] for e in exs)),
            "frac_offset_le_15_of_all": (sum(1 for o in same if o <= 15) / tot) if tot else None,
            "median_offset_same_piece": float(np.median(same)) if same else None,
            "max_offset_same_piece": int(max(same)) if same else None,
            "per_exemplar_n_offset_le_15_of_15": [e["n_offset_le_15"] for e in exs]}


def token_caption(C: Corpus, rec: dict, exs: list, bg: dict) -> dict:
    c = rec["code"]
    rank = int((C.counts > C.counts[c]).sum()) + 1
    s1_all = C.s1[C.residues_of(c)]; ok = C.s1_valid[C.residues_of(c)]
    vals = C.mean_s1[(C.counts >= 1) & np.isfinite(C.mean_s1)]
    pct = float((vals < C.mean_s1[c]).mean())
    return {
        "token": int(c), "usage": int(C.counts[c]),
        "usage_frac": float(C.counts[c] / C.counts.sum()), "usage_rank": rank,
        "mean_s1_token": float(s1_all[ok].mean()), "median_s1_token": float(np.median(s1_all[ok])),
        "mean_s1_token_over_background": float(s1_all[ok].mean() / bg["mean_s1"]),
        "token_mean_s1_percentile_among_codes": pct,
        "kappa": rec.get("kappa"), "mean_dbar": rec.get("mean_dbar"),
        "exemplars": [{"pid": e["pid"], "corpus": e["corpus"], "res": e["res"],
                       "d_z": round(e["d_z"], 3), "s1": round(e["s1"], 3),
                       "n_offset_le_15_of_15": e["n_offset_le_15"],
                       "n_other_chain_piece": e["n_other_piece"]} for e in exs],
        "exemplar_mean_s1": float(np.mean([e["s1"] for e in exs])),
        "exemplar_d_z": [round(e["d_z"], 3) for e in exs],
        "offsets": offset_stats(exs),
        "corpus_of_exemplars": sorted({e["corpus"] for e in exs}),
    }


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True, help="checkpoint dir (its name keys the export)")
    ap.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    ap.add_argument("--export-file", default="train_residues.npz")
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    ap.add_argument("--out-dir", default=None, help="default output/figures/tokens/<variant>")
    ap.add_argument("--pymol-bin", default=os.environ.get("ENSEMBITS_PYMOL") or shutil.which("pymol") or "pymol")
    ap.add_argument("--no-render", action="store_true", help="selection and caption numbers only")
    # selection
    ap.add_argument("--min-count", type=int, default=500)
    ap.add_argument("--top-n-pool", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=5)
    ap.add_argument("--n-panels", type=int, default=3)
    ap.add_argument("--max-walk", type=int, default=3000)
    ap.add_argument("--min-terminal-distance", type=int, default=5)
    ap.add_argument("--max-seq-identity", type=float, default=0.30)
    ap.add_argument("--grid-top-used", type=int, default=200)
    ap.add_argument("--n-rows", type=int, default=5)
    ap.add_argument("--n-cols", type=int, default=5)
    ap.add_argument("--stationary-token", type=int, default=-1, help="override the rule")
    ap.add_argument("--flexible-token", type=int, default=-1, help="override the rule")
    # rendering
    ap.add_argument("--frames", choices=("descriptor", "even"), default="descriptor")
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--context-radius", type=float, default=18.0)
    ap.add_argument("--bg-transparency", type=float, default=0.55)
    ap.add_argument("--fg-transparency", type=float, default=0.45)
    ap.add_argument("--zoom-buffer", type=float, default=5.0)
    ap.add_argument("--turn-x", type=float, default=-8.0)
    ap.add_argument("--turn-y", type=float, default=12.0)
    ap.add_argument("--ray-size", type=int, default=1000)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import resolve_model_dir
    vdir = resolve_model_dir(args.variant)
    export = Path(args.export_root) / vdir.name / args.export_file
    out = Path(args.out_dir) if args.out_dir else DEFAULT_FIG_ROOT / vdir.name
    work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    C = Corpus(export, Path(args.corpus_dir))
    # B.9 fix: refuse if the export's checkpoint sha doesn't match the model dir's best.pt
    export_ckpt_sha = C.prov.get("ckpt_sha12") or C.prov.get("ckpt_sha256", "")[:12]
    if export_ckpt_sha:
        import hashlib
        best_pt = vdir / "best.pt"
        if best_pt.exists():
            h = hashlib.sha256()
            with open(best_pt, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            cur_sha = h.hexdigest()[:12]
            if cur_sha != export_ckpt_sha:
                raise SystemExit(f"[render] export checkpoint {export_ckpt_sha} != "
                                 f"best.pt {cur_sha} — re-export")
    print(f"[render] {vdir.name}: {len(C.token):,} training residues, {int((C.counts > 0).sum())} codes "
          f"used ({time.time() - t0:.0f}s)", flush=True)
    bg = C.background()
    sel = select_tokens(C, args)
    common = {"variant": vdir.name, "model_dir": str(vdir), "ckpt_sha12": C.prov.get("ckpt_sha12"),
              "export": str(export), "export_sha256": C.prov.get("sha256"), "code_sha12": code_sha(),
              "args": vars(args), "selection_rule": sel["rule"], "background": bg,
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "selection.json").write_text(json.dumps({**common, **{k: v for k, v in sel.items()}},
                                                   indent=1, default=str))

    captions = {**common, "figures": {}}
    for kind, forced in (("stationary", args.stationary_token), ("flexible", args.flexible_token)):
        rec = sel[kind]
        if forced >= 0:
            rec = score_code(C, forced, args, {}) or {"code": forced}
        exs = rec["exemplars"][:args.n_panels]
        name = f"token_{kind}_{rec['code']}"
        cap = token_caption(C, rec, exs, bg)
        captions["figures"][name] = {"kind": kind, **cap}
        if not args.no_render:
            panels = render_panels(C, exs, name, work, args)
            compose_single(rec["code"], panels, out / f"{name}.png", f"Token {rec['code']}")
            (out / f"{name}.args.json").write_text(json.dumps(
                {**common, "figure": name, "token": rec["code"], "kind": kind, "forced": forced >= 0,
                 "panels": [panel_record(p) for p in panels], "caption_numbers": cap},
                indent=1, default=str))
            print(f"[render] {name} ({time.time() - t0:.0f}s)", flush=True)

    grid_rows = []
    gcap = []
    for ri, rec in enumerate(sel["grid"]):
        exs = rec["exemplars"][:args.n_cols]
        gcap.append(token_caption(C, rec, exs, bg))
        if not args.no_render:
            panels = render_panels(C, exs, f"grid_r{ri}_t{rec['code']}", work, args)
            grid_rows.append({"code": rec["code"], "mean_s1": rec["mean_s1"], "panels": panels})
    captions["figures"]["5x5tokens"] = {"kind": "grid", "rows": gcap}
    if not args.no_render:
        compose_grid(grid_rows, out / "5x5tokens.png")
        (out / "5x5tokens.args.json").write_text(json.dumps(
            {**common, "figure": "5x5tokens",
             "rows": [{"token": r["code"], "mean_s1": r["mean_s1"],
                       "panels": [panel_record(p) for p in r["panels"]]} for r in grid_rows],
             "caption_numbers": gcap}, indent=1, default=str))
        print(f"[render] 5x5tokens ({time.time() - t0:.0f}s)", flush=True)
    st = captions["figures"][f"token_stationary_{sel['stationary']['code'] if args.stationary_token < 0 else args.stationary_token}"]
    fl = captions["figures"][f"token_flexible_{sel['flexible']['code'] if args.flexible_token < 0 else args.flexible_token}"]
    captions["flexible_over_stationary_mean_s1"] = fl["mean_s1_token"] / st["mean_s1_token"]
    (out / "caption_numbers.json").write_text(json.dumps(captions, indent=1, default=str))
    print(f"[render] done -> {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
