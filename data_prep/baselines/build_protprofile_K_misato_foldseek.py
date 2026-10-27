"""ProtProfileMD_K for MISATO (or mdCATH): per-residue 3Di profile over K frames, via foldseek + mmseqs.

What it does
------------
Runs the upstream ProtProfileMD recipe (`profile_from_trajectory_pdbs_foldseek`)
per protein and per chain: writes the first K real-backbone frames of
`misato_real_bb/<pid>.npz` (`bb_8`, `cb_8`, `seq`) as PDB files, runs
`foldseek createdb`, builds a fake all-vs-first alignment with awk +
`foldseek tsv2db`, and runs `mmseqs result2profile` on the 3Di database with
pseudo-counts and masking off. The profile (L, 20) is the per-residue position
probability matrix over the 20 3Di states (rows sum to 1). Chains shorter than
4 residues get an all-zero profile. Parallel over pids with a process pool.
The probes interleave each bin with its 2-D mini3di centroid (60-D feature).

Paper
-----
`appendix.tex` \\label{app:baselines}, *ProtProfileMD_K* (K = 8 on MISATO,
K = 10 on mdCATH; construction of the ProtProfileMD paper). Results:
  * `2_results.tex` \\label{result:rmsf}, Table \\label{tab:rmsf} (baseline rows,
    mdCATH-div and MISATO columns);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * `appendix.tex` \\label{app:downstreams} (Tables \\label{tab:ec_d1_structure},
    \\label{tab:go_structure}, \\label{tab:binding_structure},
    \\label{tab:affinity_structure}) and \\label{app:experiments} (all splits).

Origin
------
Copied from `submission_exp/src/build_protprofile_K_misato_foldseek.py`.
Changed: the foldseek / mmseqs executables come from
`ensembits.paths.FOLDSEEK_BIN` / `MMSEQS_BIN` (PATH, or `ENSEMBITS_FOLDSEEK` /
`ENSEMBITS_MMSEQS`; were absolute paths in a home folder); `ROOT` from
`ensembits.paths.REPO_ROOT`; new flag `--allow-old-parser` and a MISATO parser
check for `--dataset misato` (code_fix C5); comments and this docstring. Logic
unchanged. `_split_chains` duplicates
`ensembits.baselines._real_or_ideal_bb.split_chains_by_peptide_bond` (kept; unused
since 2026-10-18).
2026-10-16: exit 1 on an existing output unless `--overwrite` (code_fix P15; the
backbones are read from the real-backbone cache directly, so no ideal-geometry
fallback exists here); `_split_chains` threshold 2.5 Å (code_fix P5).
2026-10-16 (preretrain_audit_r2 Q29): mdCATH sequences X -> H before the PDBs are
written (histidines were UNK). Phase 2: `--dataset bioemu` (bioemu_cath_real_bb bb_10 +
bioemu_cath_extras cb_10 / seq -> protprofile_K10_bioemu.npz).
2026-10-16 (preretrain_audit_r2 F1 / F7): `<output>.provenance.json` (the RMSF probes
refuse a cache without it); a protein of the RMSF residue set that fails makes the
build exit 1 (`--max-fail-frac`).
2026-10-18 (authors' decisions, baseline fidelity audit, option A): the chains are the
true chains of `misato_chain_breaks.json` (MISATO; a `chain_index` is a molecule that can
hold several chains; cut at boundary / unresolved breaks, whole across missing-residue
gaps) or the whole protein (mdCATH, BioEmu), never the 2.5 Å C–N rule
(`_chain_pieces.py`; Foldseek uses the pieces only, no residue numbering); the sidecar
records `chain_split` and the chain-break table sha256.

Used by the reported model: no (ProtProfileMD_K baseline rows).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/misato_real_bb/<pid>.npz    (--dataset misato; bb_8, cb_8, seq)
          data/cached_descriptors/mdcath_real_bb/<pid>.npz    (--dataset mdcath; bb_10, cb_10, seq)
          data/cached_descriptors/bioemu_cath_real_bb/<system>.npz (bb_10) and
          data/cached_descriptors/bioemu_cath_extras/<system>.npz (cb_10, seq)   (--dataset bioemu)
  writes  data/cached_descriptors/protprofile_K<K>_<dataset>.npz   pid -> (L, 20) float32
  (temporary PDB / database files in the system temp folder, removed per chain)

How to run
----------
    python data_prep/baselines/build_protprofile_K_misato_foldseek.py --K 8 --workers 16 \\
        [--dataset misato] [--allow-old-parser] [--overwrite]
    python data_prep/baselines/build_protprofile_K_misato_foldseek.py --K 10 --dataset bioemu
    (needs foldseek and mmseqs on PATH)
"""
from __future__ import annotations
import argparse, os, pickle, shutil, subprocess, sys, time, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))   # rmsf_common (sidecars, tolerances)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _chain_pieces (2026-10-18)
BUILDER_TAG = 'v2026-10-18'   # recorded in the cache sidecars
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.paths import FOLDSEEK_BIN, MMSEQS_BIN   # noqa: E402  executables (PATH or env override)
from ensembits.data_versions import require_misato_parser   # noqa: E402
CACHE = ROOT / 'data' / 'cached_descriptors'
# Filled in main() per --dataset.
BB = CACHE / 'misato_real_bb'
BB_KEY = 'bb_8'
CB_KEY = 'cb_8'
DATASET = 'misato'
# Phase 2 (2026-10-16): BioEmu-CATH systems take bb_10 from bioemu_cath_real_bb and Cβ /
# sequence from bioemu_cath_extras (data_prep/build_bioemu_extras.py).
BIOEMU_EXTRAS = CACHE / 'bioemu_cath_extras'
FOLDSEEK = FOLDSEEK_BIN   # was an absolute path in a home folder
MMSEQS = MMSEQS_BIN

THREE = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS',
         'I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN',
         'R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR','X':'UNK'}


# FIXED(code_fix P5, 2026-10-16): default threshold 2.5 Å (was 5.0 Å), the Ensembits
#   chain-break cutoff; same value as split_chains_by_peptide_bond. Real backbones only.
# 2026-10-18: no longer used by _process_pid (pieces = true chains, _chain_pieces.py); kept
#   for reference and its test.
def _split_chains(bb, threshold=2.5):
    L = bb.shape[0]
    if L < 2: return [(0, L)]
    breaks = (np.linalg.norm(bb[:-1, 2] - bb[1:, 0], axis=-1) > threshold)
    bounds = [0] + (np.where(breaks)[0] + 1).tolist() + [L]
    return list(zip(bounds[:-1], bounds[1:]))


def _write_pdb_lines(bb, cb, seq):
    L = bb.shape[0]
    lines = []
    ai = 1
    for ri in range(L):
        aa1 = seq[ri] if ri < len(seq) else 'X'
        aa3 = THREE.get(aa1, 'UNK')
        for nm, slot in [('N', 0), ('CA', 1), ('C', 2)]:
            x, y, z = bb[ri, slot]
            lines.append(
                f"ATOM  {ai:5d}  {nm:<3s} {aa3:>3s} A{ri+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {nm[0]:>2s}\n")
            ai += 1
        if cb is not None and not np.isnan(cb[ri]).any():
            x, y, z = cb[ri]
            lines.append(
                f"ATOM  {ai:5d}  CB  {aa3:>3s} A{ri+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            ai += 1
        x, y, z = bb[ri, 3]
        lines.append(
            f"ATOM  {ai:5d}  O   {aa3:>3s} A{ri+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n")
        ai += 1
    lines.append("END\n")
    return ''.join(lines)


# Paper: appendix.tex app:baselines ProtProfileMD_K — "fraction of frames assigned to
# each state (the construction of ProtProfileMD)"; here computed by the upstream
# foldseek + mmseqs pipeline, not by counting tokens in Python.
def _profile_chain(bb_K, cb_K, seq, K):
    """Run upstream foldseek+mmseqs on a single chain's K-frame trajectory.

    bb_K: (K, L_chain, 4, 3); cb_K: (K, L_chain, 3); seq: str of length L_chain.
    Returns (L_chain, 20) float32 PPM.

    For very short chains (L < 4) foldseek's profiling fails (mini3di
    needs ≥ 3 residues to compute conformational descriptors and
    terminals are masked, leaving < 1 valid residue). We return all
    zeros in that case — equivalent to "no signal", consistent with
    foldseek emitting no profile rows.
    """
    L = bb_K.shape[1]
    if L < 4:
        return np.zeros((L, 20), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, 'inputs')
        os.makedirs(input_dir)
        for k in range(K):
            with open(os.path.join(input_dir, f'structure_{k+1:05d}.pdb'), 'w') as f:
                f.write(_write_pdb_lines(bb_K[k], cb_K[k], seq))

        subprocess.run([FOLDSEEK, 'createdb', input_dir,
                         os.path.join(tmpdir, 'inputdb'),
                         '--threads', '1'],
                        check=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        with open(os.path.join(tmpdir, 'fake_aln.tsv'), 'w') as f:
            subprocess.run(['awk',
                            '''{ len = $3 - 2; print "0\t"$1"\t0\t1.00\t0\t0\t"(len-1)"\t"len"\t0\t"(len-1)"\t"len"\t"len"M"; }''',
                             os.path.join(tmpdir, 'inputdb.index')],
                            stdout=f, check=True)

        # NB: foldseek tsv2db doesn't accept --threads; it's single-threaded
        # by nature anyway (file conversion).
        subprocess.run([FOLDSEEK, 'tsv2db',
                         os.path.join(tmpdir, 'fake_aln.tsv'),
                         os.path.join(tmpdir, 'fake_aln_db'),
                         '--output-dbtype', '5'],
                        check=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        params = ['--pca', '0', '--pcb', '0',
                   '--profile-output-mode', '1',
                   '--mask-profile', '0',
                   '--comp-bias-corr', '0',
                   '--e-profile', 'inf', '-e', 'inf',
                   '--gap-open', '11', '--gap-extend', '1',
                   '--threads', '1']
        subprocess.run([MMSEQS, 'result2profile',
                         os.path.join(tmpdir, 'inputdb_ss'),
                         os.path.join(tmpdir, 'inputdb_ss'),
                         os.path.join(tmpdir, 'fake_aln_db'),
                         os.path.join(tmpdir, 'profile.tsv')]
                        + params,
                        check=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        # Parse the profile.tsv: skip 2 header lines, drop trailing blank column.
        # Format is space-separated with trailing space → trailing empty column.
        rows = []
        with open(os.path.join(tmpdir, 'profile.tsv')) as f:
            for line in f.readlines()[2:]:
                vals = line.strip().split()
                if len(vals) == 20:
                    rows.append([float(v) for v in vals])
        prof = np.asarray(rows, dtype=np.float32)
        if prof.shape != (L, 20):
            raise RuntimeError(f'profile shape {prof.shape} != expected ({L}, 20)')
        return prof


def _process_pid(pid, K):
    bb_path = BB / f'{pid}.npz'
    if not bb_path.exists():
        return pid, None, 'no_bb'
    try:
        d = np.load(bb_path, allow_pickle=False)
        bb_full = d[BB_KEY].astype(np.float32)
        if DATASET == 'bioemu':
            ex = np.load(BIOEMU_EXTRAS / f'{pid}.npz', allow_pickle=False)
            cb_full = ex['cb_10'].astype(np.float32)
            seq = str(ex['seq'])
        else:
            cb_full = d[CB_KEY].astype(np.float32)
            seq = str(d['seq'])
        if DATASET == 'mdcath':
            # FIXED(preretrain_audit_r2 Q29, 2026-10-16): old = the cache's 'X' (CHARMM
            #   HSD/HSE/HSP) was written as UNK into the PDBs; new = X -> H.
            #   (A sequence that already contains H is refused: the mapping assumes the cache
            #   writes every histidine as X; rmsf_common.normalize_mdcath_real_bb_seq.)
            if 'H' in seq:
                return pid, None, 'mdcath seq already contains H (X -> H mapping unsafe)'
            seq = seq.replace('X', 'H')
        if K > bb_full.shape[0]:
            return pid, None, f'K={K} > {BB_KEY} frames {bb_full.shape[0]}'
        bb_K = bb_full[:K]; cb_K = cb_full[:K]
        L_total = bb_K.shape[1]
        # Per-chain
        # FIXED(baseline fidelity audit, 2026-10-18, option A): one piece per true chain
        #   (MISATO: misato_chain_breaks.json, cut at boundary / unresolved breaks, whole
        #   across missing gaps; mdCATH / BioEmu: the whole protein). Old (code_fix P5):
        #   _split_chains, C–N > 2.5 Å on frame 0.
        from _chain_pieces import chain_pieces
        chunks, _ = chain_pieces(DATASET, pid, L_total)   # Foldseek: no numbering
        chain_profiles = []
        any_failed = False
        for s, e in chunks:
            try:
                prof = _profile_chain(bb_K[:, s:e], cb_K[:, s:e], seq[s:e], K)
            except subprocess.CalledProcessError as exc:
                # Real foldseek/mmseqs invocation error — surface it so we
                # don't silently corrupt the cache. Tiny chains hit the
                # `L < 4` guard inside _profile_chain and never reach here.
                any_failed = True
                stderr = (exc.stderr.decode('utf-8', errors='replace')[:200]
                          if isinstance(exc.stderr, bytes) else str(exc.stderr)[:200])
                return pid, None, f'subprocess_fail_{exc.returncode}: {stderr}'
            except RuntimeError as exc:
                any_failed = True
                return pid, None, f'runtime_fail: {str(exc)[:80]}'
            chain_profiles.append(prof)
        full = np.concatenate(chain_profiles, axis=0)
        if full.shape != (L_total, 20):
            return pid, None, f'concat shape {full.shape}'
        return pid, full, 'ok'
    except Exception as exc:
        return pid, None, f'err_{type(exc).__name__}_{str(exc)[:40]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, required=True, choices=[5, 8, 10])
    ap.add_argument('--dataset', choices=['misato', 'mdcath', 'bioemu'], default='misato')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--allow-old-parser', action='store_true',
                    help='accept MISATO input written before the code_fix §4 parser '
                         'fix (only to reproduce old results)')
    # FIXED(code_fix P15, 2026-10-16): old: an existing output was overwritten silently.
    ap.add_argument('--overwrite', action='store_true',
                    help='replace an existing output (default: exit 1 if it exists)')
    # FIXED(preretrain_audit_r2 F7, 2026-10-16): new: failures inside the RMSF set fail.
    ap.add_argument('--max-fail-frac', type=float, default=0.0,
                    help='allowed fraction of RMSF-set proteins that fail (default 0)')
    args = ap.parse_args()

    global BB, BB_KEY, CB_KEY, DATASET
    DATASET = args.dataset
    if args.dataset == 'bioemu':
        BB = CACHE / 'bioemu_cath_real_bb'; BB_KEY = 'bb_10'; CB_KEY = 'cb_10'
    elif args.dataset == 'misato':
        BB = CACHE / 'misato_real_bb'; BB_KEY = 'bb_8'; CB_KEY = 'cb_8'
        # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
        require_misato_parser(BB, allow_old=args.allow_old_parser)
    else:
        BB = CACHE / 'mdcath_real_bb'; BB_KEY = 'bb_10'; CB_KEY = 'cb_10'
    OUT = CACHE / f'protprofile_K{args.K}_{args.dataset}.npz'
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs
    refuse_existing_outputs([OUT], args.overwrite)   # code_fix P15
    pids = sorted(p.stem for p in BB.glob('*.npz'))
    if args.limit: pids = pids[:args.limit]
    print(f'building {OUT.name} from foldseek+mmseqs on {len(pids)} pids, '
          f'K={args.K}, workers={args.workers}', flush=True)

    profiles = {}
    counts = {}
    failed = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_pid, p, args.K): p for p in pids}
        for i, fut in enumerate(as_completed(futures)):
            pid, prof, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == 'ok':
                profiles[pid] = prof
            else:
                failed[pid] = status
                if counts.get(status, 0) <= 3:
                    print(f'  {pid}: {status}', flush=True)
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(pids) - i - 1) / max(rate, 1e-6)
                print(f'  [{i+1}/{len(pids)}] {time.time()-t0:.0f}s '
                      f'ok={counts.get("ok", 0)} ETA={eta:.0f}s', flush=True)

    print(f'\nDONE in {time.time()-t0:.0f}s — {counts}', flush=True)
    from rmsf_common import (SEQUENCE_SOURCES, canonical_dataset, split_record,
                             check_builder_failures, dataset_input_record, write_cache_provenance)
    if not check_builder_failures(failed, pids, args.dataset, args.max_fail_frac):
        print('[error] failures inside the RMSF residue set; nothing written', flush=True)
        return 1
    np.savez(OUT, **profiles)
    # FIXED(preretrain_audit_r2 F1, 2026-10-16): provenance sidecar (old: none).
    write_cache_provenance(OUT, dataset=args.dataset, builder=f'{Path(__file__).name} {BUILDER_TAG}',
                           inputs=dataset_input_record(args.dataset), n_proteins=len(profiles),
                           frames=f'first {args.K} of {BB_KEY}',
                           **split_record(args.dataset),
                           sequence_source=SEQUENCE_SOURCES[canonical_dataset(args.dataset)],
                           failed=failed, foldseek=str(FOLDSEEK), mmseqs=str(MMSEQS))
    print(f'wrote {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB, + .provenance.json)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
