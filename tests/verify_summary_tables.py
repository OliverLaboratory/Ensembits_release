"""Verify the cells in submission_exp/mds/all_baselines_summary.md.

For every section + cell, this loads the underlying per-seed JSONs,
recomputes mean ± std, and checks against the .md text (tolerance
±0.0002 on mean, ±0.0010 on std — both well below the displayed
precision).

Currently covers:
  - mdCATH div RMSF             (28 cells)
  - misato RMSF structure       (28 cells)
  - misato RMSF sequence        (28 cells)
  - EC depth-1 sequence (top-1) (28 cells)
  - Binding-site sequence       (28 cells)
  - Binding affinity sequence (no ligand)  (28 cells)
  - Binding affinity sequence (ligand-aware) (28 cells)

Other tables follow the same pattern; this script is the structural
template — extend per-section by adding entries to ``SECTIONS``.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# Dev-only: requires the original canonical sweep's per-seed JSON tree
# which is not distributed with this repo. Set ENSEMBITS_CANONICAL_ROOT
# to the absolute path that contains both the canonical
# ``submission_exp/mds/all_baselines_summary.md`` and the
# ``output/{probe_rmsf_div,probe_rmsf_misato,ec_conv1d_misato,…}/`` JSON
# subdirs the rest of this script reads.
_root_env = os.environ.get("ENSEMBITS_CANONICAL_ROOT")
if not _root_env:
    print("[skip] ENSEMBITS_CANONICAL_ROOT not set — this script requires "
          "the canonical lab tree (not in the data bundle). End users "
          "should run compare_compiled_vs_canonical.py instead.",
          file=sys.stderr)
    sys.exit(0)
ROOT = Path(_root_env)
SUMMARY = ROOT / "submission_exp" / "mds" / "all_baselines_summary.md"
# Two display precisions in the .md: most cells use 4 decimals
# (RMSF/EC/binding), the lig-aware affinity tables use 3 decimals.
# 0.0005 handles 3-decimal rounding losslessly; bumped from 0.0002 to
# absorb that without weakening the check for 4-decimal cells (any
# real regression would shift by ≫0.0005).
TOL_MEAN = 0.0005
TOL_STD = 0.0010

CELL_RE = re.compile(
    r"(?P<mean>-?\d+\.\d+)\s*[±+]\s*(?P<std>\d+\.\d+)\s*\(n=(?P<n>\d+)\)")


# ── Name → file-tag mapping (used by every section that reads
#    `<run_dir>/baseline_<tag>[_seed<n>].json` or similar) ───────
NAME_TO_TAG: dict[str, str] = {
    "3di_tokens":     "3di_tokens",
    "vote_3di":       "vote_3di",
    "aa":             "aa",
    "esm2":           "esm2",
    "esm3struct":     "esm3struct",
    "aminoaseed":     "aminoaseed",
    "protoken":       "protoken",
    "protprofile_K10": "protprofile_K10",
    "protprofile_K8":  "protprofile_K8",
    "protprofile_K5":  "protprofile_K5",
    "random":         "random",
    "mini3di":        "mini3di",
}
_VARIANT_TAGS = {
    "50-D": "_P10_dynamic_rvq_2048_128_128_k3_varP_consMSE01_distillMax_P10_realbb",
    "ESM3": "_P10_esm3desc_K16_rvq_2048_128_128_varP_consMSE01_distillMax_P10_realbb",
}
for ds in ("mdcath", "misato", "combined"):
    for desc, suf in _VARIANT_TAGS.items():
        prefix = ("combined_mdcath_misato" if ds == "combined" else ds)
        for proj in ("", " +proj"):
            proj_suf = "_aaproj" if proj else ""
            base = f"{prefix}{suf}{proj_suf}"
            for p_inf, p_tag in (("", "_P10inf"), (" (P=1)", "_P1inf")):
                NAME_TO_TAG[f"ours/{ds}/{desc}{proj}{p_inf}"] = base + p_tag


# ── Section verifiers ──────────────────────────────────────────────

def parse_section(text: str, header_re: str) -> str:
    m = re.search(header_re + r".*?(?=\n## |\Z)", text, flags=re.DOTALL)
    return m.group(0) if m else ""


def parse_subsection(text: str, header_re: str) -> str:
    """Like parse_section but stops at the next heading at any level
    (``####``, ``###``, ``##``, or end of doc) — so it doesn't concatenate
    sibling sub-tables and lose rows to dict-overwrite during parsing."""
    m = re.search(header_re + r".*?(?=\n#### |\n### |\n## |\Z)",
                  text, flags=re.DOTALL)
    return m.group(0) if m else ""


def parse_subsection_in_parent(text: str, parent_re: str, sub_re: str) -> str:
    """Find the section under ``## <parent>`` heading containing the
    subsection match. Stops at the next ``## `` or end of document.
    Useful when the same sub-heading text (``### sequence split``) appears
    under multiple parents (binding-site vs binding-affinity vs lig-aware).
    """
    parent_match = re.search(parent_re + r".*?(?=\n## |\Z)", text, flags=re.DOTALL)
    if not parent_match:
        return ""
    return parse_subsection(parent_match.group(0), sub_re)


def parse_table(table_text: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    header_cols: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    for line in table_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not header_cols:
            header_cols = cells; continue
        if all(set(c) <= {"-", ":"} for c in cells):
            continue
        if len(cells) != len(header_cols):
            continue
        rows[cells[0]] = {col: val for col, val in zip(header_cols[1:], cells[1:])}
    return header_cols, rows


def _glob_seeds(run_dir: Path, fname_at_seed: callable) -> list[Path]:
    """Helper: build seed 0..9 paths via ``fname_at_seed(seed_idx)``."""
    paths = []
    for s in range(10):
        p = run_dir / fname_at_seed(s)
        if p.exists():
            paths.append(p)
    return paths


def _extract(path: Path, json_path: list[str]) -> float | None:
    d = json.loads(path.read_text())
    for k in json_path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return float(d)


def verify_section(name: str, md_section,
                    run_dir: Path,
                    fname_at_seed: callable,
                    json_path: list[str],
                    cell_col: str = "Spearman") -> tuple[int, int, int]:
    print(f"\n=== {name} ===")
    text = SUMMARY.read_text()
    if isinstance(md_section, tuple):
        # (parent_re, subsection_re)
        section = parse_subsection_in_parent(text, *md_section)
    elif md_section.startswith("### ") or md_section.startswith("#### "):
        section = parse_subsection(text, md_section)
    else:
        section = parse_section(text, md_section)
    _, rows = parse_table(section)
    pass_n = fail_n = miss_n = 0
    for baseline, row in rows.items():
        cell = row.get(cell_col, "")
        m = CELL_RE.search(cell)
        if not m:
            continue
        md_mean = float(m["mean"]); md_std = float(m["std"]); md_n = int(m["n"])
        tag = NAME_TO_TAG.get(baseline)
        if tag is None:
            miss_n += 1
            continue
        seeds = _glob_seeds(run_dir, lambda s: fname_at_seed(tag, s))
        if len(seeds) < md_n:
            print(f"  ? {baseline:<32} md_n={md_n} but found {len(seeds)} JSONs")
            miss_n += 1
            continue
        vals = []
        for p in seeds[:md_n]:
            v = _extract(p, json_path)
            if v is None:
                vals = None; break
            vals.append(v)
        if vals is None:
            miss_n += 1
            continue
        m_mean = float(np.mean(vals)); m_std = float(np.std(vals, ddof=1))
        ok = abs(m_mean - md_mean) <= TOL_MEAN and abs(m_std - md_std) <= TOL_STD
        marker = "✓" if ok else "✗"
        print(f"  {marker} {baseline:<32} md={md_mean:.4f}±{md_std:.4f}  "
              f"recomp={m_mean:.4f}±{m_std:.4f}")
        pass_n += int(ok); fail_n += int(not ok)
    print(f"  → pass={pass_n} fail={fail_n} miss={miss_n}")
    return pass_n, fail_n, miss_n


# ── Section runner ─────────────────────────────────────────────────

def main():
    out_root = ROOT / "output"
    runs = []

    # mdcath RMSF (no split suffix)
    runs.append(("mdcath RMSF (val, CATH-H)",
                 r"### mdcath div \(val, CATH-H split\)",
                 out_root / "probe_rmsf_div",
                 lambda tag, s: f"baseline_{tag}.json" if s == 0
                                else f"baseline_{tag}_seed{s}.json",
                 ["best_val_spearman"], "Spearman"))

    # misato RMSF structure
    runs.append(("misato RMSF (structure)",
                 r"### misato \(structure split\)",
                 out_root / "probe_rmsf_misato",
                 lambda tag, s: f"baseline_{tag}_structure.json" if s == 0
                                else f"baseline_{tag}_structure_seed{s}.json",
                 ["best_val_spearman"], "Spearman"))

    # misato RMSF sequence (no suffix in filename — same dir, suffixless filename = sequence)
    runs.append(("misato RMSF (sequence)",
                 r"### misato \(sequence split\)",
                 out_root / "probe_rmsf_misato",
                 lambda tag, s: f"baseline_{tag}.json" if s == 0
                                else f"baseline_{tag}_seed{s}.json",
                 ["best_val_spearman"], "Spearman"))

    # EC depth-1 sequence — anchored under `### EC depth-1` then `#### sequence split`
    runs.append(("EC depth-1 sequence (top-1)",
                 (r"### EC depth-1", r"#### sequence split"),
                 out_root / "ec_conv1d_misato",
                 lambda tag, s: f"{tag}__d1.json" if s == 0
                                else f"{tag}__seed{s}__d1.json",
                 ["top1_hit"], "top-1"))

    # Binding-site sequence — anchored under `## Binding-site`
    runs.append(("Binding-site sequence (AUROC)",
                 (r"## Binding-site", r"### sequence split"),
                 out_root / "binding_misato",
                 lambda tag, s: f"baseline_{tag}.json" if s == 0
                                else f"baseline_{tag}_seed{s}.json",
                 ["results", "sequence_binding", "auroc"], "AUROC"))

    # Binding-affinity sequence (no ligand) — anchored under
    # `## Binding affinity (misato, mean+max-pool MLP …)` (NOT lig-aware)
    runs.append(("Binding affinity sequence, no-lig (Spearman)",
                 (r"## Binding affinity \(misato, mean", r"### sequence split"),
                 out_root / "binding_misato",
                 lambda tag, s: f"baseline_{tag}.json" if s == 0
                                else f"baseline_{tag}_seed{s}.json",
                 ["results", "sequence_affinity", "spearman"], "Spearman"))

    # Binding-affinity structure, ligand-aware — `_lig` suffix (lig only,
    # no aa concat). The lig-aware table starts with `### structure split`.
    runs.append(("Binding affinity structure, lig-aware (Spearman)",
                 (r"## Binding affinity, ligand-aware", r"### structure split"),
                 out_root / "binding_misato",
                 lambda tag, s: f"baseline_{tag}_lig.json" if s == 0
                                else f"baseline_{tag}_lig_seed{s}.json",
                 ["results", "structure_affinity", "spearman"], "Spearman"))

    # And the sequence subsection
    runs.append(("Binding affinity sequence, lig-aware (Spearman)",
                 (r"## Binding affinity, ligand-aware", r"### sequence split"),
                 out_root / "binding_misato",
                 lambda tag, s: f"baseline_{tag}_lig.json" if s == 0
                                else f"baseline_{tag}_lig_seed{s}.json",
                 ["results", "sequence_affinity", "spearman"], "Spearman"))

    totals = [0, 0, 0]
    for args in runs:
        p, f, m = verify_section(*args)
        totals[0] += p; totals[1] += f; totals[2] += m

    print("\n" + "=" * 60)
    print(f"TOTAL  pass={totals[0]}  fail={totals[1]}  miss={totals[2]}")
    print("=" * 60)
    return totals[1]   # exit code = number of failures


if __name__ == "__main__":
    import sys
    sys.exit(main())
