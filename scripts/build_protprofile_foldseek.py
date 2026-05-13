"""Build bit-equivalent ProtProfileMD targets via foldseek + mmseqs.

This is the canonical builder for the ``protprofile_K{K}_{dataset}.npz``
caches. The simpler ``mean(one_hot(mini3di_tokens))`` formulation in
``baselines/mini3di.protprofile_k`` is *algorithmically* equivalent but
**not bit-equivalent** because mmseqs applies Henikoff-style sequence
weighting + substitution-matrix smoothing internally (even with
``--pca 0 --pcb 0``). For ProtProfileMD comparison we want bit-equivalence,
so we shell out to upstream's pipeline.

External binaries required:

- foldseek 11.79cd10b static binary
- mmseqs   17.b804f release-static binary

Both must be on ``$PATH`` or passed explicitly via ``--foldseek-bin`` /
``--mmseqs-bin``.

Output: ``.npz`` keyed by pid; each value is ``(L, 20)`` float32 PPM
(rows sum to 1).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines._backbone import split_chains_by_peptide_bond     # noqa: E402

_THREE = {"A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
          "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
          "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR","X":"UNK"}


def _write_pdb_str(bb: np.ndarray, cb: np.ndarray | None, seq: str) -> str:
    L = bb.shape[0]
    lines = []
    ai = 1
    for ri in range(L):
        aa3 = _THREE.get(seq[ri] if ri < len(seq) else "X", "UNK")
        for nm, slot in [("N", 0), ("CA", 1), ("C", 2)]:
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
    return "".join(lines)


def _profile_chain(bb_K, cb_K, seq, K, foldseek_bin, mmseqs_bin):
    """Run foldseek + mmseqs on one chain's K-frame trajectory.

    Returns ``(L_chain, 20)`` float32 PPM. Tiny chains (L < 4) get an
    all-zero PPM (foldseek/mmseqs can't profile them, matching upstream).
    """
    L = bb_K.shape[1]
    if L < 4:
        return np.zeros((L, 20), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        os.makedirs(input_dir)
        for k in range(K):
            with open(os.path.join(input_dir, f"structure_{k+1:05d}.pdb"), "w") as f:
                f.write(_write_pdb_str(bb_K[k], cb_K[k] if cb_K is not None else None, seq))

        subprocess.run([foldseek_bin, "createdb", input_dir,
                         os.path.join(tmpdir, "inputdb"), "--threads", "1"],
                        check=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        with open(os.path.join(tmpdir, "fake_aln.tsv"), "w") as f:
            subprocess.run(["awk",
                            '{ len = $3 - 2; print "0\t"$1"\t0\t1.00\t0\t0\t"(len-1)"\t"len"\t0\t"(len-1)"\t"len"\t"len"M"; }',
                             os.path.join(tmpdir, "inputdb.index")],
                            stdout=f, check=True)

        # foldseek tsv2db is single-threaded; --threads not accepted.
        subprocess.run([foldseek_bin, "tsv2db",
                         os.path.join(tmpdir, "fake_aln.tsv"),
                         os.path.join(tmpdir, "fake_aln_db"),
                         "--output-dbtype", "5"],
                        check=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        params = ["--pca", "0", "--pcb", "0",
                  "--profile-output-mode", "1",
                  "--mask-profile", "0",
                  "--comp-bias-corr", "0",
                  "--e-profile", "inf", "-e", "inf",
                  "--gap-open", "11", "--gap-extend", "1",
                  "--threads", "1"]
        subprocess.run([mmseqs_bin, "result2profile",
                         os.path.join(tmpdir, "inputdb_ss"),
                         os.path.join(tmpdir, "inputdb_ss"),
                         os.path.join(tmpdir, "fake_aln_db"),
                         os.path.join(tmpdir, "profile.tsv")] + params,
                        check=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        rows = []
        with open(os.path.join(tmpdir, "profile.tsv")) as f:
            for line in f.readlines()[2:]:
                vals = line.strip().split()
                if len(vals) == 20:
                    rows.append([float(v) for v in vals])
        prof = np.asarray(rows, dtype=np.float32)
        if prof.shape != (L, 20):
            raise RuntimeError(f"profile shape {prof.shape} != expected ({L}, 20)")
        return prof


def _process_pid(args):
    pid, bb_path_str, K, foldseek_bin, mmseqs_bin = args
    bb_path = Path(bb_path_str)
    if not bb_path.exists():
        return pid, None, "no_bb"
    try:
        d = np.load(bb_path, allow_pickle=False)
        bb_key = f"bb_{K}" if f"bb_{K}" in d.files else (
            "bb_8" if "bb_8" in d.files else "bb_10")
        cb_key = f"cb_{K}" if f"cb_{K}" in d.files else (
            "cb_8" if "cb_8" in d.files else "cb_10")
        bb_full = d[bb_key].astype(np.float32)
        cb_full = d[cb_key].astype(np.float32)
        seq = str(d["seq"])
        if K > bb_full.shape[0]:
            return pid, None, f"K={K} > frames {bb_full.shape[0]}"
        bb_K = bb_full[:K]
        cb_K = cb_full[:K]

        chunks = split_chains_by_peptide_bond(bb_K[0])
        profiles = []
        for s, e in chunks:
            profiles.append(_profile_chain(
                bb_K[:, s:e], cb_K[:, s:e], seq[s:e], K, foldseek_bin, mmseqs_bin))
        full = np.concatenate(profiles, axis=0)
        if full.shape != (bb_K.shape[1], 20):
            return pid, None, f"concat shape {full.shape}"
        return pid, full, "ok"
    except subprocess.CalledProcessError as exc:
        return pid, None, f"subprocess_fail_{exc.returncode}"
    except Exception as exc:
        return pid, None, f"err_{type(exc).__name__}_{str(exc)[:40]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, required=True, choices=[5, 8, 10])
    ap.add_argument("--dataset", choices=("mdcath", "misato"), required=True)
    ap.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--foldseek-bin", default=shutil.which("foldseek"))
    ap.add_argument("--mmseqs-bin", default=shutil.which("mmseqs"))
    args = ap.parse_args()

    if not args.foldseek_bin or not Path(args.foldseek_bin).exists():
        raise SystemExit("foldseek binary not found; pass --foldseek-bin or "
                          "install foldseek 11.79cd10b on PATH.")
    if not args.mmseqs_bin or not Path(args.mmseqs_bin).exists():
        raise SystemExit("mmseqs binary not found; pass --mmseqs-bin or "
                          "install mmseqs 17.b804f-static on PATH.")

    real_bb_dir = args.data_dir / f"{args.dataset}_real_bb"
    pids = sorted(p.stem for p in real_bb_dir.glob("*.npz"))
    if args.limit:
        pids = pids[:args.limit]
    print(f"[{args.dataset}] K={args.K} on {len(pids)} pids; "
          f"foldseek={args.foldseek_bin}  mmseqs={args.mmseqs_bin}")

    work = [(p, str(real_bb_dir / f"{p}.npz"),
             args.K, args.foldseek_bin, args.mmseqs_bin) for p in pids]
    profiles: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_pid, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futures)):
            pid, prof, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == "ok":
                profiles[pid] = prof
            elif counts.get(status, 0) <= 3:
                print(f"  {pid}: {status}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(pids)}] {time.time()-t0:.0f}s "
                      f"ok={counts.get('ok', 0)}", flush=True)
    print(f"\nDONE in {time.time()-t0:.0f}s — {counts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **profiles)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
