"""Diff every cell in the reproduced ``all_baselines_summary.md`` against
the canonical ``submission_exp/mds/all_baselines_summary.md``.

For each baseline row that appears in the reproduced file, look up the
same (section, subsection, baseline, column) in the canonical file and
flag any cell that diverges by more than the tolerance on the mean.
Cells the canonical file does not have (e.g. AA-concat ligand-aware
table, EC depth-2/3 in d1-only sweeps) are reported as ``CANON_MISS``
so the reviewer can tell them apart from numeric disagreements.

Also dumps every off-cell to ``submission_exp/audits/repro_inconsistencies.csv``
(absolute path under the main repo), with the Welch two-sided t-test
p-value so reviewers can see which gaps are statistically real.

Run:
    python tests/compare_compiled_vs_canonical.py
"""
from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# Defaults point at in-repo files. Override with env vars if you keep
# the canonical .md somewhere else (e.g. when running against a fresh
# private sweep).
REPRO = Path(os.environ.get("REPRO_SUMMARY", _REPO / "all_baselines_summary.md"))
CANON = Path(os.environ.get("CANON_SUMMARY", _HERE / "canonical" / "all_baselines_summary.md"))
CSV_OUT = Path(os.environ.get("REPRO_INCONSISTENCIES_CSV", _HERE / "audits" / "repro_inconsistencies.csv"))

# Tolerance scales with the canonical's own seed-variance:
#   |Δmean| ≤ K_SEM · σ_canon / √n_canon   (≈ K_SEM-σ band on the SEM)
#   |Δstd|  ≤ K_STD · σ_canon              (relative tolerance on the std)
# With K_SEM=2 we accept anything inside the canonical's 95 % CI on
# the mean — a realistic noise band for seed-stochastic probes.
# Floor (FLOOR_MEAN) keeps very-low-variance rows from demanding
# sub-numerical-noise agreement.
K_SEM = 2.5
K_STD = 0.50
FLOOR_MEAN = 0.002
FLOOR_STD = 0.003

CELL_RE = re.compile(
    r"(?P<mean>-?\d+\.\d+)\s*[±+]\s*(?P<std>\d+\.\d+)\s*\(n=(?P<n>\d+)\)")

# ── Map section + subsection from the new md → canonical ──────────────
# Key:  "## Section / ### Subsection"
# Value: canonical equivalent, or None if canonical has no such table.
SECTION_MAP = {
    # FlexRMSF
    ("FlexRMSF (misato, MLP head)", "misato (structure split)"):
        ("FlexRMSF (per-residue Spearman, MLP head)", "misato (structure split)"),
    ("FlexRMSF (misato, MLP head)", "misato (sequence split)"):
        ("FlexRMSF (per-residue Spearman, MLP head)", "misato (sequence split)"),
    ("FlexRMSF (misato, MLP head)", "misato (random split)"):
        None,   # canonical only has structure/sequence
    # EC depth-1
    ("EC depth-1 (misato, conv1d head)", "sequence split"):
        ("EC classification (misato, conv1d head)", "EC depth-1/sequence split"),
    ("EC depth-1 (misato, conv1d head)", "structure split"):
        ("EC classification (misato, conv1d head)", "EC depth-1/structure split"),
    ("EC depth-1 (misato, conv1d head)", "random split"):
        ("EC classification (misato, conv1d head)", "EC depth-1/random split"),
    # GO
    ("GO top-50 classification (misato, conv1d head)", "sequence split"):
        ("GO top-50 classification (misato, conv1d head)", "sequence split"),
    ("GO top-50 classification (misato, conv1d head)", "structure split"):
        ("GO top-50 classification (misato, conv1d head)", "structure split"),
    ("GO top-50 classification (misato, conv1d head)", "random split"):
        ("GO top-50 classification (misato, conv1d head)", "random split"),
    # Binding-site
    ("Binding-site (misato, per-residue BCE → AUROC / AP)", "sequence split"):
        ("Binding-site (misato, per-residue BCE → AUROC / AP)", "sequence split"),
    ("Binding-site (misato, per-residue BCE → AUROC / AP)", "structure split"):
        ("Binding-site (misato, per-residue BCE → AUROC / AP)", "structure split"),
    ("Binding-site (misato, per-residue BCE → AUROC / AP)", "random split"):
        ("Binding-site (misato, per-residue BCE → AUROC / AP)", "random split"),
    # Lig-aware (MACCS)
    ("Binding affinity, ligand-aware (MACCS) (misato, MLP head → R² / Spearman / MSE)", "sequence split"):
        ("Binding affinity, ligand-aware (misato, MACCS-167 + per-residue MLP → R² / Spearman / MSE)", "sequence split"),
    ("Binding affinity, ligand-aware (MACCS) (misato, MLP head → R² / Spearman / MSE)", "structure split"):
        ("Binding affinity, ligand-aware (misato, MACCS-167 + per-residue MLP → R² / Spearman / MSE)", "structure split"),
    ("Binding affinity, ligand-aware (MACCS) (misato, MLP head → R² / Spearman / MSE)", "random split"):
        ("Binding affinity, ligand-aware (misato, MACCS-167 + per-residue MLP → R² / Spearman / MSE)", "random split"),
    # Lig-aware + AA concat — canonical summary does not have this table.
    ("Binding affinity, ligand-aware + AA concat (misato, MLP head → R² / Spearman / MSE)", "sequence split"): None,
    ("Binding affinity, ligand-aware + AA concat (misato, MLP head → R² / Spearman / MSE)", "structure split"): None,
    ("Binding affinity, ligand-aware + AA concat (misato, MLP head → R² / Spearman / MSE)", "random split"): None,
}


def parse_md(text: str) -> dict:
    """Return ``{section: {subsection: {baseline: {col: cell}}}}``."""
    out: dict = {}
    section = None
    sub = None
    ec_depth = None
    header_cols: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            section = line[3:].strip()
            sub = None
            ec_depth = None
            header_cols = None
            out.setdefault(section, {})
        elif line.startswith("### "):
            heading = line[4:].strip()
            if heading.startswith("EC depth-"):
                ec_depth = heading  # full "EC depth-1" etc.
                sub = None
            else:
                # not nested under EC: plain subsection
                sub = heading if ec_depth is None else f"{ec_depth}/{heading}"
                header_cols = None
                if section is not None:
                    out[section].setdefault(sub, {})
        elif line.startswith("#### "):
            sub = f"{ec_depth}/{line[5:].strip()}" if ec_depth else line[5:].strip()
            header_cols = None
            if section is not None:
                out[section].setdefault(sub, {})
        elif line.strip().startswith("|") and section is not None:
            # Section-direct tables (no ###/#### header) live under an
            # empty-string sub-key. Used by ProteinGym which has one table
            # right under ``## ProteinGym ...``.
            if sub is None:
                sub = ""
                out[section].setdefault(sub, {})
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header_cols is None:
                header_cols = cells
                continue
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            if len(cells) != len(header_cols):
                continue
            baseline = cells[0]
            row = {col: val for col, val in zip(header_cols[1:], cells[1:])}
            out[section][sub][baseline] = row
    return out


def parse_cell(s: str):
    m = CELL_RE.search(s)
    if not m:
        return None
    return float(m["mean"]), float(m["std"]), int(m["n"])


# ── ProteinGym verification ──────────────────────────────────────────
#
# PG is deterministic (1 number per tokenizer, no seeds), so the
# comparison is just an exact-match check on two columns:
#
#   - ``alone``               — tokenizer-only Spearman
#   - ``+ ESM2 (α=0.3)``      — grid-best z-blend (canonical reports
#                               α=0.4 in its table column, but the
#                               canonical body annotates each row with
#                               ``Grid-best α=0.3 → <value>`` — that
#                               number is the one we reproduce).
#
# Canonical baseline names are bold-wrapped and annotated, so we strip
# ``**``, ``(...)``, and ``post-fix headliner`` decorations before
# matching against the repro's plain names.

_PG_GRIDBEST_RE = re.compile(r"Grid-best\s*α\s*=\s*0\.3\s*→\s*(?P<v>0\.\d+)")
_PG_ALPHA04_RE = re.compile(r"\b0\.\d{3,4}\b")
# Map canonical canonical-name → repro display-name (when they differ).
_PG_CANON_TO_REPRO_NAME = {
    "ours/combined/ESM3": "ours/combined/ESM3",
    "aminoaseed": "aminoaseed",
    "esm3struct": "esm3struct",
    "3di_tokens": "3di_tokens",
    "protoken": "protoken",
    "ESM2-650M alone": "ESM2-650M alone",
}


def _strip_canon_pg_name(s: str) -> str:
    s = s.replace("**", "").strip()
    # drop trailing ``(...)`` annotations: "aminoaseed (atom14)" → "aminoaseed"
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    return s.strip()


def verify_pg(repro: dict, canon: dict, csv_rows: list[dict]) -> tuple[int, int]:
    """Return (matches, mismatches). Mutates ``csv_rows`` to add PG diffs."""
    # Find PG section in both.
    repro_sec = next((k for k in repro if k.startswith("ProteinGym")), None)
    canon_sec = next((k for k in canon if k.startswith("ProteinGym")), None)
    if not (repro_sec and canon_sec):
        return 0, 0
    # PG has no subsection — the table is directly under the ``## ``
    # heading; ``parse_md`` keeps it under the first (empty) sub key it
    # encounters, which by construction is the only one.
    if not repro[repro_sec] or not canon[canon_sec]:
        return 0, 0
    repro_rows = next(iter(repro[repro_sec].values()))
    canon_rows = next(iter(canon[canon_sec].values()))

    # Repro col headers: ``alone``, ``+ ESM2 (α=0.3)``
    # Canon col headers: ``alone``, ``+ ESM2 (α=0.4)``, ``Δ vs ESM2-alone``, ``n_assays``, ``status``
    #   (we reach the inline ``Grid-best α=0.3 → X`` via the ``status`` column)
    n_match = n_off = 0
    for canon_name, canon_row in canon_rows.items():
        stripped = _strip_canon_pg_name(canon_name)
        repro_name = _PG_CANON_TO_REPRO_NAME.get(stripped)
        if repro_name is None or repro_name not in repro_rows:
            continue
        r_row = repro_rows[repro_name]
        # ── alone column (exact match expected — deterministic) ──
        # Canonical wraps numbers in ``**`` (bold) — strip before parsing.
        try:
            r_alone = float(r_row["alone"].replace("*", "").strip())
            c_alone = float(canon_row["alone"].replace("*", "").strip())
        except (KeyError, ValueError, AttributeError):
            r_alone = c_alone = None
        # ── α=0.3 column ─ inline value preferred, else table α=0.4 (skip) ──
        c_a03 = None
        status = canon_row.get("status", "")
        m = _PG_GRIDBEST_RE.search(status)
        if m:
            c_a03 = float(m["v"])
        elif stripped == "ESM2-650M alone":
            c_a03 = None  # no α-blend for the anchor
        try:
            r_a03_raw = r_row.get("+ ESM2 (α=0.3)", "—")
            r_a03 = float(r_a03_raw) if r_a03_raw != "—" else None
        except ValueError:
            r_a03 = None

        for col, r_val, c_val in (
            ("alone", r_alone, c_alone),
            ("+ ESM2 (α=0.3)", r_a03, c_a03),
        ):
            if r_val is None or c_val is None:
                continue
            d = r_val - c_val
            # PG is deterministic — tolerance is 4-decimal rounding band.
            ok = abs(d) <= 5e-4
            if ok:
                n_match += 1
            else:
                n_off += 1
            csv_rows.append({
                "section": "ProteinGym mutation-effects",
                "subsection": "—",
                "baseline": repro_name,
                "column": col,
                "new_mean": f"{r_val:.6f}",
                "new_std": "",
                "new_n": 1,
                "canon_mean": f"{c_val:.6f}",
                "canon_std": "",
                "canon_n": 1,
                "delta_mean": f"{d:+.6f}",
                "delta_std": "",
                "tol_mean": "0.000500",
                "tol_std": "",
                "welch_t": "",
                "welch_df": "",
                "welch_p_two_sided": "",
                "off_mean": int(not ok),
                "off_std": 0,
            })
    return n_match, n_off


def _welch(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int):
    """Welch's two-sample t (unequal variance). Returns (t, df, p_two_sided)."""
    se2 = s1 * s1 / n1 + s2 * s2 / n2
    if se2 <= 0:
        return float("nan"), float("nan"), float("nan")
    t = (m1 - m2) / math.sqrt(se2)
    num = se2 * se2
    den = (s1 * s1 / n1) ** 2 / max(n1 - 1, 1) + (s2 * s2 / n2) ** 2 / max(n2 - 1, 1)
    df = num / den if den > 0 else float("nan")
    try:
        from scipy import stats
        p = 2.0 * (1.0 - stats.t.cdf(abs(t), df))
    except ImportError:
        # Crude normal approximation if scipy is unavailable
        p = math.erfc(abs(t) / math.sqrt(2))
    return t, df, p


def main() -> int:
    repro = parse_md(REPRO.read_text())
    canon = parse_md(CANON.read_text())

    n_match = n_mean_off = n_std_off = n_dash = n_partial = n_canon_miss = 0
    miss_canon: list[str] = []
    diffs: list[str] = []
    csv_rows: list[dict] = []

    for (sec, sub), can_pair in SECTION_MAP.items():
        if sec not in repro or sub not in repro[sec]:
            continue
        rows = repro[sec][sub]
        if can_pair is None:
            n = len(rows)
            n_canon_miss += n
            miss_canon.append(f"  (no canon)  {sec} / {sub}: {n} rows untested")
            continue
        c_sec, c_sub = can_pair
        if c_sec not in canon or c_sub not in canon[c_sec]:
            print(f"  !! Canonical section missing: {c_sec} / {c_sub}")
            n_canon_miss += len(rows)
            continue
        c_rows = canon[c_sec][c_sub]
        # Column name normalization (1:1 in our case)
        for baseline, cols in rows.items():
            for col, cell in cols.items():
                pr = parse_cell(cell)
                if pr is None:
                    # cell is "—" or partial; track for status
                    if cell.strip() == "—":
                        n_dash += 1
                    else:
                        n_partial += 1
                    continue
                pm, ps, pn = pr
                if baseline not in c_rows or col not in c_rows[baseline]:
                    n_canon_miss += 1
                    miss_canon.append(
                        f"  CANON_MISS  {sec} / {sub} / {baseline} / {col}: "
                        f"new={pm:.4f}±{ps:.4f} (n={pn})")
                    continue
                cr = parse_cell(c_rows[baseline][col])
                if cr is None:
                    n_canon_miss += 1
                    miss_canon.append(
                        f"  CANON_DASH  {sec} / {sub} / {baseline} / {col}: "
                        f"new={pm:.4f}±{ps:.4f}  canon={c_rows[baseline][col]!r}")
                    continue
                cm, cs, cn = cr
                dm = pm - cm
                ds = ps - cs
                # Per-cell tolerance from canonical variance (SEM band)
                tol_mean = max(FLOOR_MEAN, K_SEM * cs / max(cn, 1) ** 0.5)
                tol_std = max(FLOOR_STD, K_STD * cs)
                ok_mean = abs(dm) <= tol_mean
                ok_std = abs(ds) <= tol_std
                if ok_mean and ok_std:
                    n_match += 1
                else:
                    if not ok_mean:
                        n_mean_off += 1
                    if not ok_std:
                        n_std_off += 1
                    flag = []
                    if not ok_mean:
                        flag.append(f"Δmean={dm:+.4f} (tol±{tol_mean:.4f})")
                    if not ok_std:
                        flag.append(f"Δstd={ds:+.4f} (tol±{tol_std:.4f})")
                    diffs.append(
                        f"  ✗ {sec[:40]:<40} / {sub:<30} / {baseline:<32} / {col:<10}  "
                        f"new={pm:.4f}±{ps:.4f}(n={pn})  "
                        f"canon={cm:.4f}±{cs:.4f}(n={cn})  {' '.join(flag)}")
                    t, df, pval = _welch(pm, ps, pn, cm, cs, cn)
                    csv_rows.append({
                        "section": sec,
                        "subsection": sub,
                        "baseline": baseline,
                        "column": col,
                        "new_mean": f"{pm:.6f}",
                        "new_std": f"{ps:.6f}",
                        "new_n": pn,
                        "canon_mean": f"{cm:.6f}",
                        "canon_std": f"{cs:.6f}",
                        "canon_n": cn,
                        "delta_mean": f"{dm:+.6f}",
                        "delta_std": f"{ds:+.6f}",
                        "tol_mean": f"{tol_mean:.6f}",
                        "tol_std": f"{tol_std:.6f}",
                        "welch_t": f"{t:.4f}" if not math.isnan(t) else "",
                        "welch_df": f"{df:.2f}" if not math.isnan(df) else "",
                        "welch_p_two_sided": f"{pval:.6g}" if not math.isnan(pval) else "",
                        "off_mean": int(not ok_mean),
                        "off_std": int(not ok_std),
                    })

    print("=" * 80)
    print(f"DIVERGENCES (mean Δ > {K_SEM}·σ/√n  or std Δ > {K_STD}·σ_canon"
          f"; floors {FLOOR_MEAN}/{FLOOR_STD})")
    print("=" * 80)
    if diffs:
        for d in diffs:
            print(d)
    else:
        print("  (none)")

    if miss_canon:
        print()
        print("=" * 80)
        print("UNCHECKED ROWS (no canonical counterpart)")
        print("=" * 80)
        for m in miss_canon[:30]:
            print(m)
        if len(miss_canon) > 30:
            print(f"  ... and {len(miss_canon) - 30} more")

    # ── ProteinGym (deterministic; off-table comparison) ─────────────
    pg_match, pg_off = verify_pg(repro, canon, csv_rows)
    if pg_match or pg_off:
        print()
        print("=" * 80)
        print("ProteinGym (deterministic, exact-match):")
        print(f"  matched: {pg_match}   off (>5e-4): {pg_off}")
        n_match += pg_match
        n_mean_off += pg_off

    # ── CSV dump ──────────────────────────────────────────────────────
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section", "subsection", "baseline", "column",
        "new_mean", "new_std", "new_n",
        "canon_mean", "canon_std", "canon_n",
        "delta_mean", "delta_std", "tol_mean", "tol_std",
        "welch_t", "welch_df", "welch_p_two_sided",
        "off_mean", "off_std",
    ]
    # Sort: most-significant first (smallest p, then largest |Δmean|)
    def _sort_key(r):
        try:
            p = float(r["welch_p_two_sided"])
        except ValueError:
            p = 1.0
        try:
            d = abs(float(r["delta_mean"]))
        except ValueError:
            d = 0.0
        return (p, -d)
    csv_rows.sort(key=_sort_key)
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  matched           : {n_match}")
    print(f"  mean off >tol     : {n_mean_off}")
    print(f"  std  off >tol     : {n_std_off}")
    print(f"  dashes (—)        : {n_dash}")
    print(f"  partial (n<10)    : {n_partial}")
    print(f"  unchecked (canon) : {n_canon_miss}")
    print(f"  csv written       : {CSV_OUT}  ({len(csv_rows)} rows)")
    return n_mean_off


if __name__ == "__main__":
    import sys
    sys.exit(main())
