"""Tokenize the verified fold-matched pairs and the chance-floor population with every tokenizer.

What it does
------------
One code path for two populations of BioEmu-CATH systems (`--population`):

  pairs : every system of a VALID pair in `verification_divergent.json` and
          `verification_similar.json`;
  floor : every system of the standard real-backbone cache
          `data/cached_descriptors/bioemu_cath_real_bb/` (its layout_rows.json), or a
          seeded random subset of `--floor-limit` systems. The token frequencies of
          this population give each tokenizer's chance floor p = sum_t f(t)^2 in
          `bioemu_token_compare.py`, so all tokenizers' floors come from the SAME
          proteins and the same kind of input as the pair tokens.

Inputs per system (identical for both populations):

  * single representative structure = the medoid frame of run 1 (the first
    `run*_protein.cmprsd.xtc`), as in the original: the frame whose C-alpha
    coordinates are closest (sum of squared deviations) to the run-1 mean. Its REAL
    backbone N, CA, C, O is read from the trajectory, for the residues that have all
    four atoms (the residue rule of `data_prep/build_bioemu_real_bb.py`). Cached in
    `output/fold_pairs/medoids/<system>.npz` (bb (L,4,3), ca, seq, frame, run).
  * standard 10 frames = `bb_10` of `data/cached_descriptors/bioemu_cath_real_bb/<system>.npz`
    (concatenate all runs, subsample to <= 2000 frames, FPS on Kabsch C-alpha RMSD,
    k = 10). If the file is missing (pairs only), it is computed with
    `data_prep/build_bioemu_real_bb.py`'s own `_process` (same frame selection, seed 0)
    into `output/fold_pairs/real_bb_fallback/`.

Tokenizers (`--tokenizers`, default all):

  3di         mini3di on the medoid's real N, CA, C; C-beta = mini3di's own virtual
              (tetrahedral) C-beta built from the real N/CA/C
              (`ensembits.baselines._real_or_ideal_bb.cb_from_bb_or_real(bb, None)`,
              i.e. `VirtualCenterEncoder._approximate_cb_position`), the convention
              mini3di substitutes when C-beta is missing; split at chain breaks;
              terminal residues get mini3di's invalid state 2 (`.filled()`).
  aminoaseed  `data_prep/baselines/build_aminoaseed_tokens.tokenize_protein(model, ca, device, bb=medoid)`.
  esm3struct  `data_prep/baselines/build_esm3struct_tokens.tokenize_protein(enc, ca, device, bb=medoid)`.
  protoken    `ensembits.baselines.ProToken.tokenize(ca, seq, bb=medoid)`, chain by chain
              (as `data_prep/baselines/build_protoken_tokens.py`), sequence from the
              topology (`--protoken-seq polyala` restores the original poly-Ala input).
              A chain whose token count differs from its length is a failure (no zero fill).
  ens_P1      Ensembits, P = 1: the medoid's real N/CA/C.
  ens_Pfull   Ensembits, P = 10: the bb_10 frames' real N/CA/C.
Ensembits is loaded with `ensembits.featurize.load_model(--model-dir)` (default the
reported checkpoint, `ensembits.paths.CKPT_DIR`) and tokenized with
`ensembits.featurize.tokenize_ensemble`, which applies the checkpoint's dihedral
convention and descriptor standardisation.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Step 4 of `rebuttal/fold_dynamics_mining.md` (tokenization).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/bioemu_token_tokenize.py` and
largely rewritten under the approved fixes (FIXED code_fix C15 markers):
  * every baseline now reads the medoid frame's real backbone (the original passed
    the medoid C-alpha only, so AminoAseed / ESM3-struct / ProToken rebuilt N/C/O with
    ideal geometry, and 3Di was computed in the comparison script from C-alpha with a
    reconstructed backbone);
  * Ensembits P = full uses the standard bb_10 frames (the original ran its own FPS
    inside run 1 only);
  * Ensembits is loaded with `ensembits.featurize.load_model` / `tokenize_ensemble`
    (the original called `build_esm3desc_dihedral._descriptor` and
    `tokenize_misato_and_eval.load_model_robust` directly);
  * the floor population (new) and system keys (both deposits) were added; tokens are
    written one file per tokenizer and population.
Kept: the medoid rule (run 1, closest to the run mean).
Fixes 2026-10-16 (FIXED markers): Ensembits token files get a provenance sidecar
and a `checkpoint` block in the manifest, checked on reuse (--force) (code_fix P17);
--allow-unfinished (P12).

Used by the reported model: yes (the Ensembits rows use the reported checkpoint).
FIXED(code_fix E0 / FP-A8, 2026-10-23): run with λ=10 checkpoint completed.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/fold_pairs/verification_{divergent,similar}.json      (pairs)
          data/cached_descriptors/bioemu_cath_real_bb/{layout_rows.json,<system>.npz}
          data/reviewer_3_special_corpus/bioemu_cath/extracted/<deposit>/<system>/
              topology.pdb, trajs/run*_protein.cmprsd.xtc
          the tokenizer weights (ckpt/, StructTokenBench, ESM3 via the HF cache, ProToken)
  writes  output/fold_pairs/medoids/<system>.npz
          output/fold_pairs/real_bb_fallback/<system>.npz              (only if bb_10 is missing)
          output/fold_pairs/tokens/<population>/<tokenizer>.npz        system -> (L,) int64
          output/fold_pairs/tokens/<population>/manifest.json
          output/fold_pairs/tokens/<population>/ens_*.npz.provenance.json   (checkpoint block)

How to run
----------
    python downstream/fold_pairs/bioemu_token_tokenize.py --population pairs --device cuda:0
    python downstream/fold_pairs/bioemu_token_tokenize.py --population floor --device cuda:0 [--floor-limit N]
    (add --tokenizers protoken to run ProToken alone, e.g. in another environment;
     HF_HUB_OFFLINE=1 with a cached ESM3; ProToken needs TensorFlow + JAX)
"""
from __future__ import annotations
import os, sys, glob, json, argparse, time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fp_common as C   # noqa: E402
CODE_ROOT = C.CODE_ROOT
from ensembits.paths import CKPT_DIR, PROTOKEN_SRC   # noqa: E402

TOKENIZERS = ('3di', 'aminoaseed', 'esm3struct', 'protoken', 'ens_P1', 'ens_Pfull')
FPS_MAX_FRAMES = 2000      # build_bioemu_real_bb.py --max-frames default
FLOOR_SEED = 0


# ---------------------------------------------------------------- systems
def pair_systems(ver_paths):
    """Systems of the VALID pairs of the given verification JSONs."""
    systems = set()
    for vp in ver_paths:
        obj = json.load(open(vp))
        recs = obj['pairs'] if isinstance(obj, dict) else obj
        for r in recs:
            if r['valid']:
                systems.update([r['A'], r['B']])
    return sorted(systems)


def floor_systems(real_bb_dir, limit=0, seed=FLOOR_SEED):
    """Systems of the standard bb_10 cache; a seeded random subset when limit > 0."""
    # FIXED(code_fix C15, 2026-10-15): old: no floor population; the comparison took the
    #   chance floor from caches of other datasets (mdCATH for the baselines, MISATO for
    #   Ensembits). New: every system of bioemu_cath_real_bb (or a seeded subset) is
    #   tokenized here with the same inputs as the pair systems (medoid frame for the
    #   single-structure tokenizers, bb_10 for Ensembits P=full).
    lay = Path(real_bb_dir) / 'layout_rows.json'
    if lay.exists():
        systems = sorted(r['system'] for r in json.load(open(lay)))
    else:
        systems = sorted(p.stem for p in Path(real_bb_dir).glob('*.npz'))
    if not systems:
        raise FileNotFoundError(
            f'no systems in {real_bb_dir}; build it first with '
            f'data_prep/build_bioemu_real_bb.py --out {real_bb_dir}')
    if limit and limit < len(systems):
        rng = np.random.RandomState(seed)
        systems = sorted(rng.choice(systems, size=limit, replace=False).tolist())
    return systems


# ---------------------------------------------------------------- structures
def run1_backbone(system):
    """Real backbone of every frame of run 1: (F, L, 4, 3) float32 [N, CA, C, O] (A), seq, run file."""
    import mdtraj as md
    S = C.sysdir_of(system)
    if S is None:
        raise FileNotFoundError(f'no BioEmu-CATH directory for {system}')
    top = md.load_topology(str(S / 'topology.pdb'))
    xf = sorted(glob.glob(str(S / 'trajs' / 'run*_protein.cmprsd.xtc')))[0]
    t = md.load(xf, top=top)
    n_idx, ca_idx, c_idx, o_idx, res_ids = C.backbone_indices(top)
    xyz = t.xyz * 10.0
    bb = np.stack([xyz[:, n_idx], xyz[:, ca_idx], xyz[:, c_idx], xyz[:, o_idx]], axis=2).astype(np.float32)
    seq = ''.join(C.AA3TO1.get(top.residue(r).name, 'X') for r in res_ids)
    return bb, seq, os.path.basename(xf)


def medoid(system, medoid_dir=C.MEDOID_DIR):
    """Medoid frame of run 1 with its real backbone; cached in medoid_dir/<system>.npz."""
    p = Path(medoid_dir) / f'{system}.npz'
    if p.exists():
        d = np.load(p)
        return {'bb': d['bb'], 'ca': d['ca'], 'seq': str(d['seq']), 'frame': int(d['frame']),
                'run': str(d['run'])}
    bb, seq, run = run1_backbone(system)
    ca = bb[:, :, 1, :]
    # Medoid: the frame closest to the run-1 mean C-alpha structure.
    # FIXED(code_fix C15, 2026-10-15): the frames are superposed first
    #   (fold_dynamics_miner.internal_align). The trajectories are not pre-aligned
    #   (cath2_2dhkA01 run 1: raw C-alpha RMSD to frame 0 of 20-50 A, 2-5 A after
    #   superposition), so the original unaligned rule did not pick a structural medoid
    #   (frame 0 instead of frame 37 there). The original analysis had the same issue.
    import fold_dynamics_miner as _M
    ca_al = _M.internal_align(ca.astype(np.float64))
    med = int(np.argmin(((ca_al - ca_al.mean(0)) ** 2).sum((1, 2))))
    rep = {'bb': bb[med], 'ca': bb[med, :, 1, :].copy(), 'seq': seq, 'frame': med, 'run': run}
    Path(medoid_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, bb=rep['bb'], ca=rep['ca'], seq=np.array(seq), frame=np.array(med),
                        run=np.array(run))
    return rep


def _medoid_worker(system):
    try:
        medoid(system)
        return system, None
    except Exception as e:
        return system, f'{type(e).__name__}: {e}'


def load_bb10(system, real_bb_dir=C.REAL_BB, fallback_dir=C.REAL_BB_FALLBACK):
    """(10, L, 4, 3) float32 bb_10 of the standard cache; computed with
    build_bioemu_real_bb._process (same frame selection) if the system is missing."""
    # FIXED(code_fix C15, 2026-10-15): old: Ensembits P=full used 10 frames picked by a
    #   local FPS within run 1 only (seeded at the run-1 medoid). New: the standard 10
    #   frames (all runs, <= 2000 frames, Kabsch-RMSD FPS, seed 0) of
    #   data_prep/build_bioemu_real_bb.py, read from its cache or computed by its _process.
    for d in (Path(real_bb_dir), Path(fallback_dir)):
        p = d / f'{system}.npz'
        if p.exists():
            return np.load(p)['bb_10'].astype(np.float32), str(d.name)
    sys.path.insert(0, str(CODE_ROOT / 'data_prep'))
    import build_bioemu_real_bb as BB          # the standard frame selection, not reimplemented
    S = C.sysdir_of(system)
    if S is None:
        raise FileNotFoundError(f'no BioEmu-CATH directory for {system}')
    _, _, err, _ = BB._process(str(S), str(fallback_dir), BB.K_DEFAULT, FPS_MAX_FRAMES)
    if err:
        raise RuntimeError(f'build_bioemu_real_bb._process failed for {system}: {err}')
    return np.load(Path(fallback_dir) / f'{system}.npz')['bb_10'].astype(np.float32), Path(fallback_dir).name


# ---------------------------------------------------------------- tokenizers
def make_tokenizer(name, device, model_dir, protoken_seq, allow_unfinished=False):
    """Returns (fn(rep) -> (L,) int64, description)."""
    from ensembits.baselines._real_or_ideal_bb import split_chains_by_peptide_bond
    # FIXED(code_fix C15, 2026-10-15): old: the baselines received the medoid C-alpha only
    #   (AminoAseed / ESM3-struct / ProToken rebuilt N/C(/O) with ideal geometry; 3Di was
    #   computed in the comparison script from C-alpha with a reconstructed backbone).
    #   New: every tokenizer reads the medoid frame's real backbone from the trajectory.
    if name == '3di':
        import mini3di
        from ensembits.baselines._real_or_ideal_bb import cb_from_bb_or_real
        enc = mini3di.Encoder()

        def f(rep):
            bb = rep['bb'].astype(np.float32)
            n, ca, c = bb[:, 0], bb[:, 1], bb[:, 2]
            cb = cb_from_bb_or_real(bb, None)        # mini3di virtual C-beta from the real N/CA/C
            parts = [np.asarray(enc.encode_atoms(ca[s:e], cb[s:e], n[s:e], c[s:e]).filled(), dtype=np.int64)
                     for s, e in split_chains_by_peptide_bond(bb)]
            return np.concatenate(parts)
        return f, 'mini3di, medoid real N/CA/C, mini3di virtual C-beta'
    if name == 'aminoaseed':
        sys.path.insert(0, str(CODE_ROOT / 'data_prep' / 'baselines'))
        import build_aminoaseed_tokens as BA
        model = BA.build_aminoaseed_model(device)
        return (lambda rep: np.asarray(BA.tokenize_protein(model, rep['bb'][:, 1].copy(), device, bb=rep['bb']),
                                       dtype=np.int64),
                'AminoAseed, medoid real N/CA/C')
    if name == 'esm3struct':
        sys.path.insert(0, str(CODE_ROOT / 'data_prep' / 'baselines'))
        import build_esm3struct_tokens as BE
        from esm.pretrained import ESM3_structure_encoder_v0
        enc = ESM3_structure_encoder_v0(device=device)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return (lambda rep: np.asarray(BE.tokenize_protein(enc, rep['bb'][:, 1].copy(), device, bb=rep['bb']),
                                       dtype=np.int64),
                'ESM3 structure encoder, medoid real N/CA/C')
    if name == 'protoken':
        from ensembits.baselines import ProToken
        # device='' leaves CUDA_VISIBLE_DEVICES alone (set it outside if needed).
        tok = ProToken(repo_path=PROTOKEN_SRC, device='')

        def f(rep):
            bb = rep['bb'].astype(np.float32); ca = bb[:, 1]; L = bb.shape[0]
            # FIXED(code_fix C15, 2026-10-15): old: tokenize(medoid CA) with ideal N/C/O and a
            #   poly-Ala sequence. New: the real medoid backbone and, by default, the
            #   topology sequence (as data_prep/baselines/build_protoken_tokens.py does).
            seq = rep['seq'] if protoken_seq == 'real' else 'A' * L
            out = np.empty(L, dtype=np.int64)
            for s, e in split_chains_by_peptide_bond(bb):
                idx = np.asarray(tok.tokenize(ca[s:e], seq[s:e], bb=bb[s:e]), dtype=np.int64)
                if idx.shape[0] != e - s:
                    raise ValueError(f'ProToken returned {idx.shape[0]} tokens for a {e - s}-residue chain')
                out[s:e] = idx
            return out
        return f, f'ProToken, medoid real N/CA/C/O, sequence={protoken_seq}'
    if name in ('ens_P1', 'ens_Pfull'):
        # FIXED(code_fix C15, 2026-10-15): old: load_model_robust + build_esm3desc_dihedral.
        #   _descriptor called directly, standardised by hand. New: ensembits.featurize.
        #   load_model / tokenize_ensemble, which apply the checkpoint's dihedral convention,
        #   chain-break cutoff and stored mean/std.
        from ensembits.featurize import load_model, tokenize_ensemble
        ens = load_model(model_dir, device=device, allow_unfinished=allow_unfinished)
        if name == 'ens_P1':
            return (lambda rep: tokenize_ensemble(ens, None, bb_all=rep['bb'][None, :, :3].astype(np.float32)),
                    f'Ensembits P=1 (medoid real N/CA/C), model={Path(model_dir).name}, '
                    f'dihedral_convention={ens.dihedral_convention}')
        return (lambda rep: tokenize_ensemble(ens, None, bb_all=rep['bb10'][:, :, :3].astype(np.float32)),
                f'Ensembits P=10 (bb_10 real N/CA/C), model={Path(model_dir).name}, '
                f'dihedral_convention={ens.dihedral_convention}')
    raise ValueError(f'unknown tokenizer {name!r}')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--population', choices=('pairs', 'floor'), required=True)
    ap.add_argument('--tokenizers', default=','.join(TOKENIZERS),
                    help=f'comma list out of {",".join(TOKENIZERS)}')
    ap.add_argument('--verification', nargs='+',
                    default=[str(C.verification_path(m)) for m in C.SELECTIONS],
                    help='verification JSONs whose valid pairs are tokenized (population pairs)')
    ap.add_argument('--real-bb-dir', default=str(C.REAL_BB),
                    help='standard bb_10 cache (data_prep/build_bioemu_real_bb.py output)')
    ap.add_argument('--floor-limit', type=int, default=0,
                    help='floor population: seeded random subset of this many systems (0 = all)')
    ap.add_argument('--model-dir', default=str(CKPT_DIR), help='Ensembits checkpoint (default: the reported one)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--protoken-seq', choices=('real', 'polyala'), default='real')
    ap.add_argument('--workers', type=int, default=8, help='processes for the medoid step (CPU)')
    ap.add_argument('--out-dir', default=str(C.TOKENS_DIR))
    ap.add_argument('--overwrite', action='store_true', help='redo tokenizers whose file exists')
    ap.add_argument('--force', action='store_true',
                    help='keep an existing Ensembits token file even if its provenance names '
                         'another checkpoint (code_fix P17)')
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='accept a checkpoint whose training run has not finished (code_fix P12)')
    args = ap.parse_args()
    toks = [t for t in args.tokenizers.split(',') if t]
    bad = [t for t in toks if t not in TOKENIZERS]
    if bad:
        ap.error(f'unknown tokenizers {bad}')

    if args.population == 'pairs':
        systems = pair_systems(args.verification)
        src = [os.path.basename(v) for v in args.verification]
    else:
        systems = floor_systems(args.real_bb_dir, args.floor_limit)
        src = f'{Path(args.real_bb_dir).name} (limit={args.floor_limit}, seed={FLOOR_SEED})'
    out_dir = Path(args.out_dir) / args.population
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[fold-pairs-tok] population={args.population}: {len(systems)} systems from {src}', flush=True)

    # 1. Single representative structures (CPU, cached on disk).
    from multiprocessing import Pool
    t0 = time.time(); fail_med = {}
    with Pool(max(1, args.workers)) as p:
        for system, err in p.imap_unordered(_medoid_worker, systems):
            if err:
                fail_med[system] = err; print(f'  medoid FAIL {system}: {err}', flush=True)
    print(f'  medoids: {len(systems) - len(fail_med)} ok, {len(fail_med)} failed ({time.time()-t0:.0f}s)', flush=True)

    # 2. Representative + bb_10 per system, with residue-set checks.
    reps, skipped = {}, dict(fail_med)
    need_bb10 = 'ens_Pfull' in toks
    for system in systems:
        if system in skipped:
            continue
        try:
            rep = medoid(system)
            if need_bb10:
                rep['bb10'], rep['bb10_src'] = load_bb10(system, args.real_bb_dir)
                if rep['bb10'].shape[1] != rep['bb'].shape[0]:
                    raise ValueError(f'bb_10 has L={rep["bb10"].shape[1]}, medoid L={rep["bb"].shape[0]}')
            ca_npz = C.CA_CACHE / f'{system}.npz'
            if args.population == 'pairs' and ca_npz.exists():
                Lca = np.load(ca_npz)['ca'].shape[1]
                if Lca != rep['bb'].shape[0]:
                    raise ValueError(f'C-alpha cache has L={Lca}, complete-backbone L={rep["bb"].shape[0]}')
            reps[system] = rep
        except Exception as e:
            skipped[system] = f'{type(e).__name__}: {e}'
            print(f'  SKIP {system}: {skipped[system]}', flush=True)

    # 3. Tokenize, one tokenizer at a time (one model on the GPU at a time).
    man_p = out_dir / 'manifest.json'
    manifest = json.load(open(man_p)) if man_p.exists() else {}
    manifest.update({'population': args.population, 'source': src, 'n_systems': len(systems),
                     'single_structure': 'medoid frame of run 1 (closest to the run-1 mean C-alpha), real backbone',
                     'multi_frame': 'bb_10 of data/cached_descriptors/bioemu_cath_real_bb (FPS, k=10)',
                     'skipped_systems': skipped})
    manifest.setdefault('tokenizers', {})
    # FIXED(code_fix P17, 2026-10-16): old: an existing Ensembits token file was kept by name.
    #   New: the Ensembits files get <file>.provenance.json and a "checkpoint" block in the
    #   manifest; an existing one is reused only if it names the --model-dir checkpoint
    #   (refused otherwise, unless --force; --overwrite rebuilds it).
    ens_prov = None
    if any(t.startswith('ens_') for t in toks):
        from ensembits.featurize import (checkpoint_provenance, check_provenance_sidecar,
                                         check_checkpoint_finished)
        check_checkpoint_finished(args.model_dir, allow_unfinished=args.allow_unfinished)
        ens_prov = checkpoint_provenance(args.model_dir)
    for name in toks:
        outp = out_dir / f'{name}.npz'
        if outp.exists() and not args.overwrite:
            if name.startswith('ens_'):
                check_provenance_sidecar(outp, ens_prov, force=args.force)
            print(f'  [{name}] exists, skipped ({outp})', flush=True)
            continue
        t0 = time.time()
        fn, desc = make_tokenizer(name, args.device, args.model_dir, args.protoken_seq,
                                  allow_unfinished=args.allow_unfinished)
        out, failed = {}, {}
        for i, (system, rep) in enumerate(sorted(reps.items())):
            try:
                tok = np.asarray(fn(rep), dtype=np.int64)
                if tok.shape != (rep['bb'].shape[0],):
                    raise ValueError(f'got {tok.shape} tokens for L={rep["bb"].shape[0]}')
                out[system] = tok
            except Exception as e:
                failed[system] = f'{type(e).__name__}: {e}'
                if len(failed) <= 5:
                    print(f'  [{name}] FAIL {system}: {failed[system]}', flush=True)
            if (i + 1) % 100 == 0:
                print(f'  [{name}] {i+1}/{len(reps)} ({time.time()-t0:.0f}s)', flush=True)
        np.savez(outp, **out)
        manifest['tokenizers'][name] = {'input': desc, 'n_ok': len(out), 'failed': failed}
        if name.startswith('ens_'):
            from ensembits.featurize import write_provenance_sidecar
            write_provenance_sidecar(outp, ens_prov, population=args.population, tokenizer=name,
                                     written_by='downstream/fold_pairs/bioemu_token_tokenize.py')
            manifest['tokenizers'][name]['checkpoint'] = ens_prov      # FIXED(code_fix P17, 2026-10-16): new key (old: none)
        json.dump(manifest, open(man_p, 'w'), indent=1)
        print(f'  [{name}] {len(out)} ok, {len(failed)} failed -> {outp} ({time.time()-t0:.0f}s)', flush=True)
        del fn
    print(f'[fold-pairs-tok] done -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
