"""CryptoBench: build per-chain backbones and per-residue cryptic-site labels.

What it does
------------
Step 1 of the CryptoBench pipeline (see `README.md` in this folder).

  1. Reads the CryptoBench annotation `dataset.json` (apo PDB id -> list of
     holo partners, each with the apo/holo chain, the pocket residues on both
     sides as "{auth_asym_id}_{auth_seq_id}", the ligand, pRMSD and the
     `is_main_holo_structure` flag) and `folds.json` (test, train-0 .. train-3;
     apo ids clustered at <= 10% sequence identity between subsets).
  2. For every (structure, chain) that carries a pocket, collects the positive
     residue ids. `--side apo` (the cryptic task): input = the ligand-free apo
     chain, positives = UNION of the apo-side pocket selections over all its holo
     partners. `--side holo` (open-pocket control): input = the holo chain,
     positives = its own pocket. `--main-holo-only` keeps only the
     representative pocket per apo. The fold is always inherited from the apo id.
  3. Parses the mmCIF of each chain with gemmi (model 0), keeps the POLYMER residues
     (label_seq_id present) that are one of the 20 standard amino acids, MSE / SEC / PYL,
     or (default `--modified-residues parent`) any other modified amino acid mapped to its
     parent (MLY -> K, SEP -> S, ...), drops residues missing any of N/CA/C, and maps labels
     on (auth_seq_id, insertion code) after the drop. Chains shorter than 16 residues are
     skipped (the descriptor needs K = 16 neighbours; in this release 5 chains, 2 of them
     in test: 4z0y F and H).
  4. Writes one npz per chain and a manifest with fold, length, positives,
     number of pockets and max pRMSD per entry.

Paper
-----
Not in the paper yet. The CryptoBench experiment is to be added to the main
text: `camera_ready/require_fix.md` #24, `camera_ready/code_fix.md` E3.
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Benchmark: V. Škrhák, M. Novotný, C. P. Feidakis, R. Krivák, D. Hoksza,
"CryptoBench: cryptic protein-ligand binding sites dataset and benchmark",
Bioinformatics 41(1):btae745, 2025, doi:10.1093/bioinformatics/btae745;
data on OSF (doi:10.17605/OSF.IO/PZ4A9).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/cryptobench_build_apo.py`.
Changes: comments; ROOT comes from `ensembits.paths.REPO_ROOT`; the hard-coded
locations (originally ROOT = the research repo root, raw data
ROOT/data/cryptobench, output ROOT/data/cached_descriptors/cryptobench_<side><suffix>/,
manifest ROOT/data/cryptobench/<side><suffix>_manifest.json) are now resolved by
`_cb_paths.py` (run-tag folder by default, `--legacy-layout` for the old
locations). Parsing, labelling and filtering changed by the FIXED items below only.

Fixes 2026-10-21 (FIXED markers; camera_ready/cryptobench_audit/PORT.md, audits B / C):
  * altloc rule made explicit (audit C finding 7): first conformer = for every atom the
    first-listed alternative location (gemmi `res[name][0]`, altloc A where altlocs exist),
    as the old code did implicitly; with residue-level microheterogeneity (two residues at
    one seqid) the first one with an amino-acid name is kept and later ones are skipped.
    (gemmi `remove_alternative_conformations()` was tried and rejected: in 4z0y B/D it keeps
    the modified HS8 at 252, which the amino-acid filter then drops, cutting a test chain.)
    Every tokenizer reads the same cache, so one rule holds for all rows.
  * polymer residues only: a residue without label_seq_id (non-polymer HETATM, e.g. free
    TRP / GLY / PHE / PRO ligands that share the chain name) is skipped; the old build
    appended 22 such ligand "residues" to 19 chains (5 in test, all labelled 0). New
    finding of the port (2026-10-21), see cryptobench_audit/PORT.md.
  * stores `label_seq` (L,) int32 = mmCIF label_seq_id (SEQRES position): the gap-aware
    residue index of the baselines (audit C findings 2 / 5); always present (polymer only).
  * stores `o` (L, 3) float32, the real carbonyl O (NaN if not resolved), for ProToken
    (audit C finding 9; O was left at the origin).
  * labels matched on (auth_seq_id, insertion code): a selection id "60A" now labels
    residue 60A and "60" no longer labels 60A (audit C finding 8; train folds only).
  * stores `nc` (L,) int8: residue lines a NON-cryptic pocket of the apo entry
    (`noncryptic-pockets.json`, apo side; matched like the labels), for the pre-declared
    sensitivity analysis that masks such negatives (DECISIONS.md CB6). Holo side: zeros.
  * manifest meta records `build_version`, the altloc / label rules.
  MSE / SEC / PYL are kept as before (audit B finding 4).
Fixes 2026-10-22 (port re-audit, camera_ready/cryptobench_audit/FIXLIST_20261022.md):
  * FIXED(REAUDIT CB3 F1; authors' TENTATIVE answer 2026-10-22): modified polymer residues
    (MLY, LLP, SEP, CME, CSO, MIS, TPO, PTR, OCS, ...) are KEPT and mapped to their parent
    amino acid (`--modified-residues parent`, default) instead of dropped (`drop`, the
    2026-10-21 behaviour). Parent from the entry's `_pdbx_struct_mod_residue`
    (label_comp_id -> parent_comp_id), else gemmi's residue table (lower-case one-letter
    code of a modified amino acid); a residue with neither stays dropped and is counted.
    The npz stores `comp` (residue name) per residue; the manifest records the mode, the
    mapping (comp -> parent, source, count) and the counts. The mode is part of the
    manifest, so changing it changes the manifest sha256 and every downstream fingerprint.
  * FIXED(REAUDIT CB3 F8): microheterogeneity: a seqid counts as taken only once a residue
    with a full backbone was kept (an alternative is tried if the first lacks N/CA/C).
  * FIXED(REAUDIT CB5 F4): a rebuild into an existing folder moves the old structure folder
    and manifest to `_pre_<date>/` first (they were overwritten in place).
  * parse path: `gemmi.cif.read` + `make_structure_from_block` (to reach
    `_pdbx_struct_mod_residue`) replaced `gemmi.read_structure`; since fix round 2 it is
    followed by `merge_chain_parts()`, which is what read_structure did by default (the
    outputs were identical on all 1,276 chains without it, FIXAUDIT FX2 F4).
Fix round 2, 2026-10-22 (FIXLIST2_20261022.md items 4 / 8):
  * FIXED(FIXAUDIT FX4 F2 / F3): the parent of a modified residue comes FIRST from the PDB
    Chemical Component Dictionary (`data/external/components.cif.gz`,
    `_chem_comp.mon_nstd_parent_comp_id`, when it names exactly one standard amino acid),
    then from the entry's `_pdbx_struct_mod_residue`, then gemmi's table. So 4CF -> F (the
    entry annotates the wild-type ILE of an engineered mutation) and AME -> M (no
    `_pdbx_struct_mod_residue` in 3q6u) are kept with their chemical parent. Multi-parent
    chromophores (PIA: ALA,TYR,GLY) fall back to the entry's annotation.
  * FIXED(FIXAUDIT FX4 F5): every residue dropped at the N / CA / C check is counted per name
    (`dropped_incomplete_backbone`, e.g. the PIA chromophore, whose atoms are N1 / CA1 / C3), and
    microheterogeneity alternatives skipped are counted (`skipped_alternative`).
  * FIXED(FIXAUDIT FX2 F3 / FX3 F3): the manifest records the sha256 of this script, of the
    release files dataset.json / folds.json / noncryptic-pockets.json and of the CCD; the runner
    rebuilds when any differs (it compared only the build version and the mode).

Used by the reported model: yes, the input structures and labels of every
CryptoBench row (Ensembits and baselines).

Inputs / outputs
----------------
Read  : `data/cryptobench/dataset.json`, `folds.json`, `cif-files/<pdb>.cif`
        (CryptoBench OSF download; `--cryptobench-dir` to override).
Write : `<structs-root>/cryptobench_<side><suffix>/<pdb>_<chain>.npz` with
          bb (1, L, 3, 3) float32 N/CA/C (P = 1), y (L,) int8, seq str,
          auth (L,) str, icode (L,) <U1, cb (L, 3) float32 (NaN if absent),
          label_seq (L,) int32, o (L, 3) float32 (NaN if absent), nc (L,) int8 (2026-10-21),
          comp (L,) <U5 residue name (2026-10-22);
        `<manifest-dir>/<side><suffix>_manifest.json`.
        Default <structs-root> = <manifest-dir> =
        data/cached_descriptors/cryptobench_runs/<run_tag>/.

The rebuttal build (legacy layout) has: apo 1,276 entries / 374,269 residues /
5.39% positive; apo_main 1,274; holo 1,289 (`rebuttal/cryptobench.md` §1).

Dependencies left in the research repo: none (needs `gemmi`, not installed in
the analysis environment; the original structure stage used a separate environment with gemmi).

How to run
----------
  $PY downstream/cryptobench/cryptobench_build_apo.py --workers 32                      # apo
  $PY downstream/cryptobench/cryptobench_build_apo.py --workers 32 --side apo --main-holo-only --out-suffix _main
  $PY downstream/cryptobench/cryptobench_build_apo.py --workers 32 --side holo --main-holo-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402,F401  data/ and output/ stay in the research repo
import _cb_paths as CBP  # noqa: E402

# Set in main() from the CLI (originally module constants under
# the research repo root). Worker processes inherit them through fork.
CB = None    # raw CryptoBench folder
CIF = None   # CB / "cif-files"
OUT = None   # structure cache folder of this set

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


BUILD_VERSION = "cb_build_v4_2026-10-22"     # = cryptobench_common.STRUCT_BUILD_VERSION
ALTLOC_RULE = ("first-listed altloc per atom (gemmi res[name][0]); microheterogeneity: first "
               "amino-acid residue with a full N/CA/C backbone per seqid; polymer residues "
               "only (label_seq_id present)")
MODRES_MODES = ("parent", "drop")
MODRES = "parent"   # set in main() from --modified-residues (workers inherit it through fork)
CCD_PARENTS = {}    # {comp_id: mon_nstd_parent_comp_id} of the CCD, set in main() (fork)
RELEASE_FILES = ("dataset.json", "folds.json", "noncryptic-pockets.json")


def default_ccd() -> Path:
    from ensembits.paths import EXTERNAL
    return Path(EXTERNAL) / "components.cif.gz"


def load_ccd_parents(path) -> dict:
    """{comp_id: parent string} from the CCD mmCIF (gz): one `_chem_comp` record per block;
    a value on the following line (long-form CIF) is read too; '?' / '.' kept as is."""
    import gzip
    out, cid, want_val = {}, None, False
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if want_val:
                out[cid] = line.strip().strip('"').strip("'")
                want_val = False
                continue
            if line.startswith("_chem_comp.id ") or line.startswith("_chem_comp.id\t"):
                cid = line.split(None, 1)[1].strip().strip('"').strip("'")
            elif line.startswith("_chem_comp.mon_nstd_parent_comp_id"):
                parts = line.split(None, 1)
                if len(parts) > 1 and parts[1].strip():
                    out[cid] = parts[1].strip().strip('"').strip("'")
                else:
                    want_val = True
    return out
STANDARD3 = {k for k in AA3TO1 if k not in ("MSE", "SEC", "PYL")}
LABEL_RULE = "auth_seq_id + insertion code (selection id '<num><icode>')"


def parse_selection(sel):
    """['B_12', 'B_14', 'B_60A', ...] -> {chain: {'12', '14', '60A', ...}} (number + insertion
    code as one string, matched against `res_key`). Values may be str."""
    out = defaultdict(set)
    if isinstance(sel, str):                      # some rows are stringified lists
        sel = json.loads(sel.replace("'", '"'))
    for s in sel:
        chain, _, resi = str(s).partition("_")
        if not resi:
            continue
        out[chain].add(resi.strip())
    return out


def modres_parents(block) -> dict:
    """{label_comp_id: parent_comp_id} of an mmCIF block (`_pdbx_struct_mod_residue`)."""
    out = {}
    tab = block.find("_pdbx_struct_mod_residue.", ["label_comp_id", "parent_comp_id"])
    for row in tab:
        c, p = row.str(0).upper(), row.str(1).upper()
        if c and p:
            out.setdefault(c, p)
    return out


def residue_letter(name: str, parents: dict, mode: str = None, ccd: dict = None):
    """(one-letter code, source) of a polymer residue, or (None, reason) to drop it.

    20 standard + MSE / SEC / PYL as before; other names under mode 'parent': the CCD parent
    (`mon_nstd_parent_comp_id`) if it is exactly one standard amino acid, else the parent from
    the entry's `_pdbx_struct_mod_residue` if standard, else gemmi's table (a modified amino
    acid has a lower-case one-letter code); under mode 'drop': dropped."""
    mode = mode or MODRES
    ccd = CCD_PARENTS if ccd is None else ccd
    if name in AA3TO1:
        return AA3TO1[name], "standard"
    if mode == "drop":
        return None, "dropped_by_mode"
    c = (ccd.get(name) or "").upper()
    if c in STANDARD3:
        return AA3TO1[c], "ccd"
    p = parents.get(name)
    if p in STANDARD3:
        return AA3TO1[p], "pdbx_struct_mod_residue"
    try:
        import gemmi
        info = gemmi.find_tabulated_residue(name)
    except Exception:                              # noqa: BLE001
        info = None
    if info is not None and info.is_amino_acid():
        one = info.one_letter_code
        if one and one.isalpha() and one.islower() and one.upper() in "ACDEFGHIKLMNPQRSTVWY":
            return one.upper(), "gemmi_table"
    return None, "no_parent"


def chain_backbone(cif_path, chain_id, stats=None):
    """(L,3,3) N/CA/C, sequence, auth_seq_ids, icodes, ... for one auth chain.

    Residues missing any of N/CA/C are dropped (they cannot enter the
    descriptor); the returned auth ids let labels be mapped after the drop.
    stats (dict, optional): filled with {comp: [letter, source, n kept]} of the modified
    residues kept and {"dropped": {comp: n}} of polymer residues dropped for their name.
    """
    import gemmi
    # FIXED(REAUDIT CB3 F1, 2026-10-22): one parse of the block, so the entry's
    #   `_pdbx_struct_mod_residue` (parents of modified residues) is available too.
    block = gemmi.cif.read(str(cif_path)).sole_block()
    parents = modres_parents(block)
    st = gemmi.make_structure_from_block(block)
    st.merge_chain_parts()        # what gemmi.read_structure does by default (FX2 F4)
    st.setup_entities()
    stats = {} if stats is None else stats

    if len(st) == 0:
        return None
    model = st[0]                                 # first model only
    ch = None
    for c in model:
        if c.name == chain_id:
            ch = c
            break
    if ch is None:
        return None
    xyz, seq, auth, icode, cb, lseq, oxy, comp = [], [], [], [], [], [], [], []
    seen = set()
    for res in ch:
        name = res.name.upper()
        # FIXED(port 2026-10-21, new finding): polymer residues only; free amino-acid
        #   ligands (no label_seq_id) were appended to the chain before.
        if res.label_seq is None:
            continue                              # ligands, waters, ions
        letter, src = residue_letter(name, parents)
        if letter is None:
            d = stats.setdefault("dropped", {})
            d[name] = d.get(name, 0) + 1
            continue                              # polymer residue with no amino-acid parent
        # FIXED(CB audit C finding 7, 2026-10-21; REAUDIT CB3 F8, 2026-10-22):
        #   microheterogeneity: the first amino-acid residue of a seqid WITH a full backbone
        #   is kept, later alternatives are skipped.
        sid = (res.seqid.num, res.seqid.icode)
        if sid in seen:
            d = stats.setdefault("skipped_alternative", {})
            d[name] = d.get(name, 0) + 1
            continue
        try:
            # first conformer of each atom (altlocs beyond [0] are ignored)
            a_n, a_ca, a_c = res["N"][0], res["CA"][0], res["C"][0]
        except (RuntimeError, IndexError, KeyError):
            # FIXED(FIXAUDIT FX4 F5): counted (e.g. the PIA chromophore: N1 / CA1 / C3)
            d = stats.setdefault("incomplete_backbone", {})
            d[name] = d.get(name, 0) + 1
            continue                              # incomplete backbone
        seen.add(sid)
        if src not in ("standard",):
            m = stats.setdefault("modified", {})
            m.setdefault(name, [letter, src, 0])[2] += 1
        comp.append(name)
        xyz.append([[a_n.pos.x, a_n.pos.y, a_n.pos.z],
                    [a_ca.pos.x, a_ca.pos.y, a_ca.pos.z],
                    [a_c.pos.x, a_c.pos.y, a_c.pos.z]])
        # real Cbeta where the structure has it (mini3di needs Cb; GLY and any
        # unresolved Cb stay NaN so the consumer can fill them with mini3di's
        # own tetrahedral approximation via `cb_from_bb_or_real`)
        try:
            a_cb = res["CB"][0]
            cb.append([a_cb.pos.x, a_cb.pos.y, a_cb.pos.z])
        except (RuntimeError, IndexError, KeyError):
            cb.append([np.nan, np.nan, np.nan])
        # FIXED(CB audit C findings 2 / 5 / 9, 2026-10-21): SEQRES position (gap-aware
        #   residue index of the baselines) and the real carbonyl O (ProToken input).
        lseq.append(int(res.label_seq) if res.label_seq is not None else 0)
        try:
            a_o = res["O"][0]
            oxy.append([a_o.pos.x, a_o.pos.y, a_o.pos.z])
        except (RuntimeError, IndexError, KeyError):
            oxy.append([np.nan, np.nan, np.nan])
        seq.append(letter)
        auth.append(str(res.seqid.num))
        icode.append((res.seqid.icode or "").strip())
    if len(xyz) < 16:                             # need > K=16 neighbours
        return None
    return (np.asarray(xyz, dtype=np.float32), "".join(seq),
            np.asarray(auth, dtype=object), np.asarray(icode, dtype="<U1"),
            np.asarray(cb, dtype=np.float32), np.asarray(lseq, dtype=np.int32),
            np.asarray(oxy, dtype=np.float32), np.asarray(comp, dtype="<U5"))


def res_key(num: str, icode: str) -> str:
    """Label key of a residue: auth_seq_id followed by its insertion code ('60', '60A')."""
    return f"{num}{(icode or '').strip()}"


def build_one(job):
    """Parse one chain, label it, write its npz; return a manifest record."""
    apo, chain, pos_ids, fold, n_pockets, prmsd, nc_ids = job
    st = {}
    try:
        got = chain_backbone(CIF / f"{apo.lower()}.cif", chain, st)
    except Exception as e:                        # noqa: BLE001
        return {"apo": apo, "chain": chain, "status": f"parse_error:{type(e).__name__}"}
    if got is None:
        return {"apo": apo, "chain": chain, "status": "no_chain_or_too_short",
                "fold": fold, "n_label_ids": len(pos_ids)}
    bb, seq, auth, icode, cb, lseq, oxy, comp = got
    L = len(seq)
    # FIXED(CB audit C finding 8, 2026-10-21): old: matched on the number only (a label
    #   "60A" never matched, a label "60" also hit residue 60A). New: number + icode.
    keys = [res_key(a, i) for a, i in zip(auth.tolist(), icode.tolist())]
    hit = pos_ids & set(keys)
    y = np.fromiter((1 if k in pos_ids else 0 for k in keys), dtype=np.int8, count=L)
    # residues of a non-cryptic pocket of this entry (sensitivity analysis, CB6)
    nc = np.fromiter((1 if k in nc_ids else 0 for k in keys), dtype=np.int8, count=L)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{apo}_{chain}.npz",
                        bb=bb[None].astype(np.float32), y=y, seq=seq,
                        auth=auth.astype("U8"), icode=icode, cb=cb,
                        label_seq=lseq, o=oxy, nc=nc, comp=comp)
    return {"apo": apo, "chain": chain, "status": "ok", "L": L,
            "n_pos": int(y.sum()), "n_label_ids": len(pos_ids),
            "n_matched": len(hit), "fold": fold, "n_nc": int(nc.sum()),
            "n_label_seq_missing": int((lseq < 1).sum()),
            "n_o_missing": int((~np.isfinite(oxy).all(axis=1)).sum()),
            "n_pockets": n_pockets, "pRMSD_max": prmsd,
            "modified": st.get("modified", {}), "dropped_by_name": st.get("dropped", {}),
            "dropped_incomplete_backbone": st.get("incomplete_backbone", {}),
            "skipped_alternative": st.get("skipped_alternative", {}),
            "path": str(OUT / f"{apo}_{chain}.npz")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip-multichain", action="store_true",
                    help="drop pockets whose chain field spans multiple chains")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--side", choices=["apo", "holo"], default="apo",
                    help="apo: ligand-free input, labels from the holo partner (the "
                         "cryptic task). holo: ligand-bound input with that structure's "
                         "own pocket residues -- the open-pocket control, which bounds "
                         "how much of the apo difficulty is crypticity vs the head.")
    ap.add_argument("--out-manifest", default=None)
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the cache dir + manifest, e.g. _main")
    ap.add_argument("--main-holo-only", action="store_true",
                    help="keep only the representative pocket per apo "
                         "(`is_main_holo_structure`), so the control has one structure "
                         "per protein and is size-matched to the apo arm")
    ap.add_argument("--ccd", default=None,
                    help="PDB Chemical Component Dictionary (default data/external/components.cif.gz)")
    ap.add_argument("--modified-residues", choices=MODRES_MODES, default="parent",
                    help="parent (default; authors' tentative answer 2026-10-22): keep modified "
                         "polymer residues mapped to their parent amino acid; drop: the "
                         "2026-10-21 behaviour (20 standard + MSE / SEC / PYL only)")
    CBP.add_path_args(ap)
    args = ap.parse_args()
    paths = CBP.resolve_paths(args, create=("structs_root", "manifest_dir"))
    global OUT, CB, CIF, MODRES, CCD_PARENTS
    MODRES = args.modified_residues
    ccd_path = Path(args.ccd) if args.ccd else default_ccd()
    if MODRES == "parent":
        if not ccd_path.exists():
            raise SystemExit(f"CCD {ccd_path} missing (needed for --modified-residues parent)")
        CCD_PARENTS = load_ccd_parents(ccd_path)
    CB = paths.cryptobench_dir
    CIF = CB / "cif-files"
    # original: ROOT / "data" / "cached_descriptors" / f"cryptobench_{side}{suffix}"
    OUT = CBP.structs_dir(paths, f"{args.side}{args.out_suffix}")
    # original default: CB / f"{side}{suffix}_manifest.json"
    manifest_path = args.out_manifest or str(CBP.manifest_path(paths, f"{args.side}{args.out_suffix}"))

    dataset = json.load(open(CB / "dataset.json"))
    folds = json.load(open(CB / "folds.json"))
    fold_of = {pid: name for name, ids in folds.items() for pid in ids}

    # union the selections -> {(pdb_id, chain): set(auth_seq_id)}
    # Folds are always inherited from the APO structure (CryptoBench clusters apo
    # ids at <=10% identity), so the holo control uses the same protein partition
    # and stays comparable to the apo arm.
    pos = defaultdict(set)
    npock = defaultdict(int)
    pmax = defaultdict(float)
    fold_src = {}
    sel_key = "apo_pocket_selection" if args.side == "apo" else "holo_pocket_selection"
    chain_key = "apo_chain" if args.side == "apo" else "holo_chain"
    for apo, holos in dataset.items():
        for h in holos:
            if args.skip_multichain and "-" in str(h.get(chain_key, "")):
                continue
            if (args.main_holo_only
                    and str(h.get("is_main_holo_structure", "")).lower() != "true"):
                continue
            pid = apo if args.side == "apo" else h["holo_pdb_id"]
            for chain, ids in parse_selection(h[sel_key]).items():
                pos[(pid, chain)] |= ids
                npock[(pid, chain)] += 1
                fold_src.setdefault((pid, chain), fold_of.get(apo, "unassigned"))
                try:
                    pmax[(pid, chain)] = max(pmax[(pid, chain)], float(h.get("pRMSD", 0)))
                except (TypeError, ValueError):
                    pass
    fold_of = fold_src

    # FIXED(DECISIONS.md CB6, 2026-10-21): residues lining the entry's NON-cryptic pockets
    #   (apo side only), stored per residue as `nc` for the masked sensitivity analysis.
    ncp = defaultdict(set)
    nc_file = CB / "noncryptic-pockets.json"
    if args.side == "apo":
        if not nc_file.exists():
            raise SystemExit(f"{nc_file} missing (needed for the nc flag)")
        for apo, pockets in json.load(open(nc_file)).items():
            for h in pockets:
                for chain, ids in parse_selection(h.get("apo_pocket_selection", [])).items():
                    ncp[(apo, chain)] |= ids

    # FIXED(REAUDIT CB5 F4, 2026-10-22): never overwrite a previous build in place.
    import cryptobench_common as CC
    old_npz = OUT.exists() and any(OUT.glob("*.npz"))
    if old_npz or Path(manifest_path).exists():
        moved = CC.move_aside_tree(OUT) if old_npz else []
        if Path(manifest_path).exists():
            moved += CC.move_aside(manifest_path)
        print(f"[cryptobench/{args.side}] previous build moved aside ({len(moved)} paths)")

    jobs = [(pid, chain, ids, fold_of.get((pid, chain), "unassigned"),
             npock[(pid, chain)], pmax[(pid, chain)], frozenset(ncp.get((pid, chain), ())))
            for (pid, chain), ids in sorted(pos.items())]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[cryptobench/{args.side}] {len(dataset)} apo structures -> {len(jobs)} "
          f"({args.side},chain) entries to build; "
          f"folds={ {k: len(v) for k, v in folds.items()} }", flush=True)

    # Parse all chains in parallel (fork start method: workers see OUT / CIF).
    recs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(build_one, jobs, chunksize=4), 1):
            recs.append(r)
            if i % 200 == 0:
                ok = sum(x["status"] == "ok" for x in recs)
                print(f"  {i}/{len(jobs)}  ok={ok}", flush=True)

    ok = [r for r in recs if r["status"] == "ok"]
    bad = [r for r in recs if r["status"] != "ok"]
    modmap, dropped, dropped_bb, skipped_alt = {}, {}, {}, {}
    for r in ok:
        for c, (letter, src, n) in r["modified"].items():
            modmap.setdefault(c, {"parent": letter, "source": src, "n": 0})["n"] += n
        for src_d, dst in ((r["dropped_by_name"], dropped),
                           (r["dropped_incomplete_backbone"], dropped_bb),
                           (r["skipped_alternative"], skipped_alt)):
            for c, n in src_d.items():
                dst[c] = dst.get(c, 0) + n
    import cryptobench_common as CC
    fingerprint = {"build_code_sha256": CC.file_sha256(__file__),
                   "release_sha256": {f: CC.file_sha256(CB / f) for f in RELEASE_FILES
                                      if (CB / f).exists()},
                   "ccd_sha256": CC.file_sha256(ccd_path) if MODRES == "parent" else None}
    json.dump({"entries": ok, "failed": bad,
               "meta": {"dataset": str(CB / "dataset.json"), "cif": str(CIF),
                        "skip_multichain": args.skip_multichain, "side": args.side,
                        "build_version": BUILD_VERSION, "altloc_rule": ALTLOC_RULE,
                        "label_rule": LABEL_RULE, "modified_residues": MODRES,
                        "modified_mapping": modmap, "dropped_by_name": dropped,
                        "dropped_incomplete_backbone": dropped_bb,
                        "skipped_alternative": skipped_alt, **fingerprint,
                        "nc_source": "noncryptic-pockets.json" if args.side == "apo" else None}},
              open(manifest_path, "w"), indent=1)

    # Summary: sizes, class balance, label ids lost to the backbone filter.
    L = np.array([r["L"] for r in ok])
    P = np.array([r["n_pos"] for r in ok])
    unmatched = sum(r["n_label_ids"] - r["n_matched"] for r in ok)
    total_ids = sum(r["n_label_ids"] for r in ok)
    print(f"\nbuilt {len(ok)} entries ({len(bad)} failed)")
    if bad:
        from collections import Counter
        print("  failure modes:", dict(Counter(r["status"] for r in bad)))
    print(f"  residues: total={L.sum()} median L={int(np.median(L))} "
          f"min={L.min()} max={L.max()}")
    print(f"  positives: total={P.sum()} ({100*P.sum()/L.sum():.2f}% of residues) "
          f"median/entry={int(np.median(P))}, entries with 0 positives={int((P==0).sum())}")
    print(f"  label ids not found in structure: {unmatched}/{total_ids} "
          f"({100*unmatched/max(total_ids,1):.2f}%)")
    print(f"  modified residues kept ({MODRES}): {sum(v['n'] for v in modmap.values())} "
          f"({len(modmap)} names); polymer residues dropped by name: {sum(dropped.values())}; "
          f"dropped at the backbone check: {sum(dropped_bb.values())} {dropped_bb}")
    print(f"  residues on a non-cryptic pocket (nc): {sum(r['n_nc'] for r in ok)}; "
          f"label_seq_id missing: {sum(r['n_label_seq_missing'] for r in ok)}; "
          f"O missing: {sum(r['n_o_missing'] for r in ok)}")
    by_fold = defaultdict(int)
    for r in ok:
        by_fold[r["fold"]] += 1
    print("  entries per fold:", dict(by_fold))
    print(f"  wrote {manifest_path} and {OUT}/*.npz")


if __name__ == "__main__":
    main()
