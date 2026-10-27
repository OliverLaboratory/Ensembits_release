"""Freeze the current best checkpoint of a (possibly running) training run for probing.

What it does
------------
Copies `best.pt`, `config.json`, `stats.npz` and `history.json` of `output/<run>/`
into a new folder `output/snap_<run>_ep<best_epoch>/` and writes `snapshot.json`
there (source run, best epoch, best validation reconstruction loss from
history.json, sha256 of best.pt, code_sha256 of the run, time). The best epoch is
found with the trainer's rule (an evaluation improves when its val_recon is below
the best so far by more than 1e-5). The copy is made in a temporary folder and
renamed at the end (atomic); best.pt is hashed before and after copying, and
history.json is re-read after the copy (train.py writes best.pt before history.json):
the snapshot is accepted only if best.pt and the best epoch did not change, else it
retries a few times and then refuses (H5).

The probes accept the snapshot folder as `--variant` (its config.json still says
"running", so `--allow-unfinished` is still required); they record
`snapshot_of` in their result JSON (rmsf_common.snapshot_info), and
`rmsf_selection_table.py` marks snapshot rows. A running training can therefore be
probed on a frozen copy without the race between loading best.pt and recording its
hash (preretrain_audit_r2 F9).

Paper
-----
Not described in the paper (infrastructure only).

Origin
------
New (2026-10-16, preretrain_audit_r2 F9).

Used by the reported model: no (probing unfinished runs for model selection).

Inputs / outputs
----------------
  reads   output/<run>/{best.pt,config.json,stats.npz,history.json}
  writes  output/snap_<run>_ep<best_epoch>/{best.pt,config.json,stats.npz,history.json,
          snapshot.json}; refuses an existing destination

How to run
----------
    python downstream/snapshot_checkpoint.py --run <run> [--name <folder name>]
    python downstream/_run_model_battery.py --variant snap_<run>_ep<k> \\
        --tasks PRETOK,RMDIV,RMMIS,RMBIO --allow-unfinished
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

FILES = ('best.pt', 'config.json', 'stats.npz', 'history.json')
IMPROVE = 1e-5          # train/train.py: val_recon < best - 1e-5 saves best.pt


def best_epoch_of(history: dict):
    """(epoch, val_recon) of the evaluation that wrote best.pt (the trainer's rule)."""
    best, ep = float('inf'), None
    for e, v in zip(history.get('eval_epochs') or [], history.get('val_recon') or []):
        if v < best - IMPROVE:
            best, ep = float(v), int(e)
    return ep, (best if ep is not None else None)


def _sha(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _best_epoch_file(path):
    return best_epoch_of(json.loads(Path(path).read_text())) if Path(path).exists() else (None, None)


def snapshot(run_dir: Path, out_root: Path, name: str | None = None, retries: int = 5,
             wait_s: float = 5.0) -> Path:
    """Make the snapshot folder; returns its path. Raises RuntimeError when the training kept
    changing best.pt / history.json during `retries` attempts.

    FIXED(preretrain_audit_r2 H5, 2026-10-16): train.py writes best.pt BEFORE history.json,
    so a copy taken in between pairs a new best.pt with the old best epoch. Each attempt
    reads the best epoch from history.json, copies the files, and accepts only if best.pt is
    unchanged (source before == copy == source after) AND history.json (source re-read and
    the copied file) still gives the same best epoch; otherwise it waits and retries.
    """
    run_dir = Path(run_dir)
    for f in ('best.pt', 'config.json'):
        if not (run_dir / f).exists():
            raise FileNotFoundError(f'{run_dir / f} missing')
    last = None
    for attempt in range(retries):
        ep, best = _best_epoch_file(run_dir / 'history.json')
        nm = name or f'snap_{run_dir.name}_ep{ep if ep is not None else "unknown"}'
        dest = Path(out_root) / nm
        if dest.exists():
            raise FileExistsError(f'{dest} exists; snapshots are never overwritten')
        tmp = Path(out_root) / f'.{nm}.tmp{os.getpid()}'
        tmp.mkdir(parents=True)
        try:
            sha0 = _sha(run_dir / 'best.pt')
            for f in FILES:
                if (run_dir / f).exists():
                    shutil.copy2(run_dir / f, tmp / f)
            sha_copy, sha1 = _sha(tmp / 'best.pt'), _sha(run_dir / 'best.pt')
            ep_src, _ = _best_epoch_file(run_dir / 'history.json')
            ep_copy, _ = _best_epoch_file(tmp / 'history.json')
            if not (sha0 == sha_copy == sha1 and ep == ep_src == ep_copy):
                last = (f'attempt {attempt + 1}: best.pt or the best epoch changed while copying '
                        f'({ep} -> {ep_src} / {ep_copy})')
                shutil.rmtree(tmp, ignore_errors=True)
                time.sleep(wait_s)
                continue
            cfg = json.loads((tmp / 'config.json').read_text())
            (tmp / 'snapshot.json').write_text(json.dumps({
                'snapshot_of': run_dir.name, 'best_epoch': ep, 'best_val_recon_history': best,
                'best_pt_sha256': sha0, 'code_sha256': cfg.get('code_sha256'),
                'source_status': cfg.get('status'),
                'files': [f for f in FILES if (tmp / f).exists()], 'attempts': attempt + 1,
                'created': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
            os.replace(tmp, dest)
            return dest
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    raise RuntimeError(f'{run_dir.name}: the training changed best.pt / history.json during '
                       f'{retries} attempts ({last}); run again')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True, help='output/<run> to snapshot')
    ap.add_argument('--name', default=None, help='destination folder name (default snap_<run>_ep<k>)')
    a = ap.parse_args(argv)
    try:
        dest = snapshot(ROOT / 'output' / a.run, ROOT / 'output', a.name)
    except (FileNotFoundError, FileExistsError, RuntimeError) as e:
        print(f'[error] {e}'); return 1
    info = json.loads((dest / 'snapshot.json').read_text())
    print(f'wrote {dest} (best epoch {info["best_epoch"]}, best.pt {info["best_pt_sha256"][:12]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
