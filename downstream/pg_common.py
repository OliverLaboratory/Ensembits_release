"""ProteinGym helpers shared by the scorers and the aggregator (assays, variants, metrics, α CV).

What it does
------------
One place for everything the ProteinGym scripts must do identically for every
tokenizer and control (benchmark_audit D9):
  * the assay set: `read_assay_list` / `assay_list_sha256` for the fixed list
    `downstream/pg_assays_wt_le200.txt` (96 assays of ProteinGym v1.2 substitutions
    with WT length <= 200; built and checked by `pg_assay_list.py`);
  * the variants of one assay: `load_assay_rows` reads the ProteinGym CSV in float64
    (DMS_score, ESM2_650M), checks every mutated sequence against the WT and the
    `mutant` string, and returns the number of mutated positions `n_mut`;
  * structure reading with a content hash (`read_structure`), so every result records
    which ESMFold structures it was computed from (`structure_set_sha256`);
  * metrics: `spearman` (float64, a constant predictor scores 0 and is flagged),
    `zscore` (population std, constant -> 0), `blend_spearman` (alpha * z(method)
    + (1 - alpha) * z(ESM2)), `ranksum_spearman` (parameter-free rank(method) +
    rank(ESM2));
  * `louo_alpha_cv`: the blend weight alpha chosen by leave-one-UniProt-out cross-
    validation (for each protein, alpha is the grid value with the best mean
    Spearman over the OTHER proteins' assays; that protein's assays are scored with it);
  * atomic writers (`atomic_write_text`, `atomic_savez`).

Paper
-----
`2_results.tex`, \\paragraph{Mutation Effect Prediction} and Table
\\label{tab:proteingym_ensemble} (96 assays, z-score blend with ESM2-650M, alpha);
`appendix.tex` ProteinGym details. The CV rule for alpha, the rank-sum blend and the
-n_mut control are the camera-ready protocol (benchmark_audit D9); the paper text
FIXED(audit D.16, tex pass 2026-10-26): alpha now LOUO CV in paper.

Origin
------
New (2026-10-19, benchmark_audit D9). The CSV reading and the z-blend follow
`downstream/pg_score.py` (copied from the research repo) and
`Ensembits/downstreams/proteingym.py`; differences: float64 everywhere (the old
float32 cast created ties that changed the ESM2 Spearman of some assays), a constant
predictor scores 0 instead of NaN, variants without a finite ESM2 / DMS value are
dropped for every score at once (old: the blend keys silently disappeared for the
whole assay).

Used by the reported model: yes (through pg_score.py / pg_score_dihedral.py and
pg_aggregate.py).

Inputs / outputs
----------------
Reads data/external/proteingym_zero_shot_scores/<assay>.csv (PROTEINGYM_ZERO_SHOT_DIR),
data/proteingym/esmfold_cache_atom14/<assay>/<sha1[:16]>.npy and
downstream/pg_assays_wt_le200.txt. Writes nothing itself.

How to run
----------
Library:
    from pg_common import load_assay_rows, spearman, louo_alpha_cv
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

# The fixed assay list of the paper (pg_assay_list.py writes and checks it).
ASSAY_LIST = Path(__file__).resolve().parent / "pg_assays_wt_le200.txt"
ESM_COLUMN = "ESM2_650M"
# atom14 layout of the ESMFold cache (openfold residue_constants): N, CA, C, O, CB, ...
# The research script build_pg_token_caches.py assumed N, CA, C, CB, O (3Di read O as
# its Cbeta, ProToken read Cbeta as its O); preretrain_audit_r2 Q10 / benchmark_audit D9.
ATOM14_N, ATOM14_CA, ATOM14_C, ATOM14_O, ATOM14_CB = 0, 1, 2, 3, 4
# Default alpha grid of the CV rule (0, 0.05, ..., 1). alpha = 0 is ESM2 alone,
# alpha = 1 the method alone, so the CV may also say "do not blend".
ALPHA_GRID = tuple(round(0.05 * k, 2) for k in range(21))


# ---------------------------------------------------------------- assays
def uniprot_of(assay: str) -> str:
    """UniProt entry name of a ProteinGym assay id ("<ENTRY>_<SPECIES>_<Author>_<Year>...").

    ProteinGym assay ids start with the UniProt entry name, which has exactly one
    underscore (e.g. AMFR_HUMAN, A0A140D2T1_ZIKV); the reference file with the
    UniProt_ID column is not available here, so it is read from the id.
    """
    parts = assay.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected ProteinGym assay id {assay!r}")
    return "_".join(parts[:2])


def is_tsuboyama(assay: str) -> bool:
    """True for the Tsuboyama 2023 cDNA-display stability assays."""
    return "_Tsuboyama_2023_" in assay


def read_assay_list(path: Path | str = ASSAY_LIST) -> list[str]:
    """One assay id per line; blank lines and `#` comments ignored; order kept, deduplicated."""
    lines = Path(path).read_text().splitlines()
    out = [l.split("#", 1)[0].strip() for l in lines]
    return list(dict.fromkeys(a for a in out if a))


def assay_list_sha256(assays) -> str:
    """sha256 of the sorted, deduplicated assay ids joined by newlines (order-independent)."""
    return hashlib.sha256("\n".join(sorted(set(assays))).encode()).hexdigest()


def hash_seq(s: str) -> str:
    """File stem of a sequence's cached structure: first 16 hex chars of its SHA-1."""
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def parse_mutant(mutant: str) -> list[tuple[str, int, str]]:
    """'A12C:D15E' -> [('A', 11, 'C'), ('D', 14, 'E')] (0-based positions)."""
    out = []
    for s in mutant.split(":"):
        s = s.strip()
        if len(s) < 3 or not s[1:-1].isdigit():
            raise ValueError(f"bad substitution {s!r} in {mutant!r}")
        out.append((s[0], int(s[1:-1]) - 1, s[-1]))
    return out


def wt_from_row(mutant: str, mut_seq: str) -> str:
    """Wild-type sequence implied by one variant row (undo its substitutions)."""
    seq = list(mut_seq)
    for wt_aa, pos, mt_aa in parse_mutant(mutant):
        if pos >= len(seq) or seq[pos] != mt_aa:
            raise ValueError(f"{mutant!r} does not match the mutated sequence")
        seq[pos] = wt_aa
    return "".join(seq)


def _float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_assay_rows(csv_path: Path | str, wt_seq: str) -> tuple[list[dict], dict]:
    """Variant rows of one ProteinGym CSV, in file order.

    Each row: {mutant, mut_seq, DMS (float64), ESM2 (float64), n_mut}, where n_mut is
    the number of positions at which the mutated sequence differs from `wt_seq`.
    Rows whose mutated sequence has another length than the WT are dropped (none in
    the 96 assays). Returns (rows, stats) with stats = {n_rows, n_len_mismatch,
    n_inconsistent (mutant string disagrees with the sequence difference),
    n_dup_mutant, n_nan_dms, n_nan_esm2}; callers refuse an assay with
    n_inconsistent or n_dup_mutant > 0.
    """
    rows, seen = [], set()
    st = dict(n_rows=0, n_len_mismatch=0, n_inconsistent=0, n_dup_mutant=0,
              n_nan_dms=0, n_nan_esm2=0)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            st["n_rows"] += 1
            ms = r.get("mutated_sequence", "") or ""
            mut = r.get("mutant", "") or ""
            if len(ms) != len(wt_seq):
                st["n_len_mismatch"] += 1
                continue
            diff = {j for j, (a, b) in enumerate(zip(wt_seq, ms)) if a != b}
            try:
                subs = parse_mutant(mut)
                ok = ({p for _, p, _ in subs} == diff and
                      all(wt_seq[p] == w and ms[p] == m for w, p, m in subs))
            except ValueError:
                ok = False
            if not ok:
                st["n_inconsistent"] += 1
            if mut in seen:
                st["n_dup_mutant"] += 1
            seen.add(mut)
            dms, esm = _float(r.get("DMS_score")), _float(r.get(ESM_COLUMN))
            st["n_nan_dms"] += int(not np.isfinite(dms))
            st["n_nan_esm2"] += int(not np.isfinite(esm))
            rows.append({"mutant": mut, "mut_seq": ms, "DMS": dms, "ESM2": esm,
                         "n_mut": len(diff)})
    return rows, st


# ---------------------------------------------------------------- structures
def read_structure(path: Path | str) -> tuple[np.ndarray, str]:
    """Load one cached `.npy` structure and return (array, sha256 of the file bytes)."""
    data = Path(path).read_bytes()
    return np.load(io.BytesIO(data), allow_pickle=False), hashlib.sha256(data).hexdigest()


def structure_digest(file_shas: dict) -> str:
    """sha256 over sorted 'name:sha' lines (the structures one assay was scored on)."""
    lines = "\n".join(f"{k}:{file_shas[k]}" for k in sorted(file_shas))
    return hashlib.sha256(lines.encode()).hexdigest()


def structure_set_sha256(per_assay: dict) -> str:
    """sha256 over sorted 'assay:structure_sha256' lines (the whole structure set)."""
    return structure_digest(per_assay)


def check_atom14_slots(a14: np.ndarray, seq: str, tol: float = 0.25) -> None:
    """Refuse an array that is not atom14 N, CA, C, O, CB (ESMFold / openfold order).

    Checks the median |C - O| (1.23 A) and, over non-Gly residues, the median
    |CA - CB| (1.53 A); Gly has CB = 0 in atom14. Raises ValueError otherwise.
    """
    a14 = np.asarray(a14, dtype=np.float64)
    if a14.ndim != 3 or a14.shape[1] < 5 or a14.shape[0] != len(seq):
        raise ValueError(f"expected (L={len(seq)}, >=5, 3) atom14, got {a14.shape}")
    d_co = np.median(np.linalg.norm(a14[:, ATOM14_C] - a14[:, ATOM14_O], axis=-1))
    ng = np.array([c != "G" for c in seq])
    d_cacb = (np.median(np.linalg.norm(a14[ng, ATOM14_CA] - a14[ng, ATOM14_CB], axis=-1))
              if ng.any() else 1.53)
    if abs(d_co - 1.23) > tol or abs(d_cacb - 1.53) > tol:
        raise ValueError(f"atom14 slot check failed: median |C-O| {d_co:.2f} A "
                         f"(expect 1.23), |CA-CB| {d_cacb:.2f} A (expect 1.53)")


# ---------------------------------------------------------------- metrics
def spearman(pred, y) -> tuple[float, bool]:
    """(Spearman rho, constant flag) in float64.

    A constant predictor (or constant target) has no ranking information; it scores
    0.0 and the flag is True (scipy would return NaN, which the old code then dropped
    from the assay mean, so methods were averaged over different assays).
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    if p.size < 2 or np.ptp(p) == 0 or np.ptp(t) == 0:
        return 0.0, True
    return float(spearmanr(p, t).correlation), False


def zscore(x) -> np.ndarray:
    """Within-assay z-score with the population std; a constant vector -> zeros."""
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return np.zeros_like(x) if s == 0 else (x - x.mean()) / s


def blend_spearman(pred, esm2, y, alphas=ALPHA_GRID) -> np.ndarray:
    """Spearman of alpha * z(pred) + (1 - alpha) * z(esm2) for each alpha (higher = fitter)."""
    zp, ze = zscore(pred), zscore(esm2)
    return np.array([spearman(a * zp + (1.0 - a) * ze, y)[0] for a in alphas])


def ranksum_spearman(pred, esm2, y) -> float:
    """Parameter-free blend: Spearman of rank(pred) + rank(esm2) (average ranks for ties)."""
    return spearman(rankdata(np.asarray(pred, dtype=np.float64)) +
                    rankdata(np.asarray(esm2, dtype=np.float64)), y)[0]


def louo_alpha_cv(rho_by_alpha: dict, alphas=ALPHA_GRID, group_of=uniprot_of) -> dict:
    """Leave-one-UniProt-out choice of the blend weight alpha.

    rho_by_alpha: {assay: (n_alpha,) per-assay Spearman of the blend at each alpha}.
    For each protein g: alpha*_g = argmax over the grid of the mean Spearman over the
    assays of all OTHER proteins (unweighted mean over assays, as in the reported
    metric; ties -> the smallest alpha). Each assay of g is scored at alpha*_g.
    Returns {"rho_cv": {assay: rho}, "alpha_by_group": {g: alpha}, "alpha_by_assay":
    {assay: alpha}, "alpha_full": argmax over ALL assays (test-selected, for reference
    only), "rho_full_by_alpha": mean over all assays at each alpha}.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    assays = sorted(rho_by_alpha)
    R = np.stack([np.asarray(rho_by_alpha[a], dtype=np.float64) for a in assays])
    if R.shape[1] != len(alphas):
        raise ValueError("rho_by_alpha rows must have one value per alpha")
    groups = np.array([group_of(a) for a in assays])
    rho_cv, a_by_group, a_by_assay = {}, {}, {}
    for g in sorted(set(groups)):
        out = groups != g
        if not out.any():
            raise ValueError("leave-one-group-out CV needs at least two groups")
        k = int(np.argmax(R[out].mean(axis=0)))     # first max = smallest alpha
        a_by_group[g] = float(alphas[k])
        for i in np.where(~out)[0]:
            rho_cv[assays[i]] = float(R[i, k])
            a_by_assay[assays[i]] = float(alphas[k])
    full = R.mean(axis=0)
    return {"rho_cv": rho_cv, "alpha_by_group": a_by_group, "alpha_by_assay": a_by_assay,
            "alpha_full": float(alphas[int(np.argmax(full))]),
            "rho_full_by_alpha": [float(v) for v in full]}


# ---------------------------------------------------------------- writers
def atomic_write_text(path: Path | str, text: str) -> None:
    """Write text through a temporary file in the same folder, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path | str, obj, indent=2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent))


def atomic_savez(path: Path | str, **arrays) -> None:
    """np.savez_compressed through an open handle (no '.npz' suffix games), then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(paths) -> str:
    """sha256 over the given source files (name + bytes), for result provenance."""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()
