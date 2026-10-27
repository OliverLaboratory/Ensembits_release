"""Tests of the baseline chain pieces and the constant-prediction rule (2026-10-18).

What it does
------------
Authors' decisions of 2026-10-18 (baseline fidelity audits, option A, table v2):
  * `data_prep/build_misato_chain_breaks.py` on synthetic complexes (mmCIF with SEQRES
    positions): a hidden chain boundary inside one MISATO molecule is a `boundary`; a gap
    is a `missing_gap` with n_missing from the SEQRES step; a stretched bond with SEQRES
    step 1 is `consecutive`; (a) a chain_index change inside one deposited chain is placed
    (missing_gap) and (b) its step is the SEQRES step although the author numbers are
    consecutive; a same-chain pair 25 Å apart is a `boundary` (source "physical"); a
    mutated flank residue is placed by the alignment fallback; homo-dimer boundaries name
    the copies; an extra MISATO copy sets `extra_copies`; unplaceable breaks are
    `unresolved`; PDB input falls back to author numbers (flagged); unit tests of the
    flank, physical and renumbered-deletion rules;
  * `data_prep/baselines/_chain_pieces.py`: cut at boundary / unresolved, whole across
    missing_gap / consecutive, residue index = SEQRES step (Cα estimate when unknown);
  * the mini3di builders end to end on those complexes (tokens = per-true-chain encoding,
    sidecar "true_chain_v2" + table sha256, refused after the table changes); mdCATH-div /
    BioEmu-CATH tokens bit-identical to the old rule on a synthetic single chain;
  * the residue index reaches the encoders (fake AminoAseed / ESM3 models, fake ProToken,
    and the PDB residue numbers written by `ensembits.baselines.protoken`);
  * the pooled builders refuse caches / shards of another chain split;
  * constant predictions: the probe records constant_prediction / n_unique_inputs; the
    compilers show a cell whose seeds are all one-token constant predictions as "—†"
    (not ranked, not missing), refuse any other NaN, and detect the older results
    (best_epoch 0 + NaN + one-token token cache);
  * read-only regression cases of the audits on the built table (skipped without a v2 table).

Paper
-----
`appendix.tex` \\label{app:baselines} (the baselines are applied per chain);
`2_results.tex` \\label{result:rmsf} (RMSF tables).

Origin
------
New (2026-10-18).

Used by the reported model: no (tests).

Inputs / outputs
----------------
Synthetic data in pytest tmp folders only (ENSEMBITS_REPO_ROOT points there).

How to run
----------
    python -m pytest tests/test_baseline_chain_pieces.py
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
BASELINES = CODE_ROOT / "data_prep" / "baselines"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream"))
sys.path.insert(0, str(CODE_ROOT / "data_prep"))
sys.path.insert(0, str(BASELINES))

import _chain_pieces as CP  # noqa: E402
import rmsf_common as RC  # noqa: E402
from ensembits.baselines._real_or_ideal_bb import (cb_from_bb_or_real,  # noqa: E402
                                                   split_chains_by_peptide_bond)

AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
         "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
         "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
# 1AAA (v2 rules): molecule 0 = deposited chain A (12 residues, SEQRES 1-6 and 10-15: 3
# missing) + chain B (10, a hidden boundary inside the molecule); molecule 1 = chain C part 1
# (SEQRES 1-8, a stretched bond after its 5th residue: consecutive); molecule 2 = chain C
# part 2 (SEQRES 13-18) after a missing loop, with author numbers 9-14 (renumbered: auth
# step 1, SEQRES step 5) -> the chain_index change is a missing_gap of 4, kept whole.
LA, LB, LC1, LC2 = 12, 10, 8, 6
LTOT = LA + LB + LC1 + LC2
WANT_PIECES = [(0, LA), (LA, LA + LB), (LA + LB, LTOT)]
WANT_RI = np.array(list(range(6)) + list(range(9, 15)) + list(range(LB)) + list(range(LC1))
                   + list(range(12, 18)))
LAB_A = list(range(1, 7)) + list(range(10, 16))
LAB_C = list(range(1, 9)) + list(range(13, 19))
AUTH_C = [(i, "") for i in range(1, 15)]          # renumbered over the deletion


def _helix(labels, rng, origin, stretch_at=None, offset_at=None, offset=None):
    """(n, 4, 3) N/CA/C/O and (n, 3) Cβ of the residues `labels` (SEQRES positions) of a
    helix-like chain (C-N about 0.9 Å inside it), so missing SEQRES positions leave a gap.
    stretch_at: index of a residue after a stretched bond (C-N 2.6 Å, Cα 3.9 Å);
    offset_at / offset: residues from that index on moved by `offset` (far parts)."""
    full = np.arange(min(labels) - 1, max(labels) + 2)
    t = full * np.deg2rad(100.0)
    ca_f = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * full], 1) + origin
    ca_f += rng.normal(scale=0.03, size=ca_f.shape)
    idx = np.searchsorted(full, labels)
    ca = ca_f[idx]; prev = ca_f[idx - 1]; nxt = ca_f[idx + 1]
    n = ca + 0.24 * (prev - ca) / np.linalg.norm(prev - ca, axis=1, keepdims=True) * 3.8
    c = ca + 0.24 * (nxt - ca) / np.linalg.norm(nxt - ca, axis=1, keepdims=True) * 3.8
    bb = np.stack([n, ca, c, c + np.array([0.0, 1.2, 0.0])], 1)
    cb = ca + np.cross(n - ca, c - ca) * 0.5 + 0.3
    if stretch_at is not None:
        d = ca[stretch_at] - ca[stretch_at - 1]; dh = d / np.linalg.norm(d)
        e = np.cross(dh, [0.0, 0.0, 1.0]); e /= np.linalg.norm(e)
        s = -0.24 * np.linalg.norm(d) * dh + 2.6 * e
        bb[stretch_at:] += s; cb[stretch_at:] += s
    if offset_at is not None:
        bb[offset_at:] += offset; cb[offset_at:] += offset
    return bb.astype(np.float32), cb.astype(np.float32)


def _chain(L, rng, origin):
    """A gap-free helix of L residues (mdCATH / BioEmu / encoder tests)."""
    return _helix(list(range(1, L + 1)), rng, origin)


def _seq(rng, L):
    return "".join(rng.choice(list(AA), size=L))


def _complex_1aaa(rng):
    a = _helix(LAB_A, rng, np.zeros(3))
    b = _helix(list(range(1, LB + 1)), rng, np.array([30.0, 0.0, 0.0]))
    c = _helix(LAB_C, rng, np.array([0.0, 40.0, 0.0]), stretch_at=5)
    bb = np.concatenate([a[0], b[0], c[0]]); cb = np.concatenate([a[1], b[1], c[1]])
    ci = np.array([0] * (LA + LB) + [1] * LC1 + [2] * LC2, np.int32)
    return bb, cb, ci


def _pdb_lines(chains):
    """chains: [(chain id, [(one letter, auth num, ins, label, CA xyz)])] -> CA ATOM records."""
    out, k = [], 1
    for cid, res in chains:
        for aa, num, ins, _lab, (x, y, z) in res:
            out.append(f"ATOM  {k:5d}  CA  {THREE[aa]} {cid}{num:4d}{ins or ' '}   "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"); k += 1
    return "".join(out) + "END\n"


def _cif_text(chains):
    head = ("data_X\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
            "_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_seq_id\n"
            "_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
            "_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n_atom_site.pdbx_PDB_model_num\n")
    rows, k = [], 1
    for cid, res in chains:
        for aa, num, ins, lab, (x, y, z) in res:
            rows.append(f"ATOM {k} CA . {THREE[aa]} {lab} {THREE[aa]} {cid} {num} {ins or '?'} "
                        f"{x:.3f} {y:.3f} {z:.3f} 1"); k += 1
    return head + "\n".join(rows) + "\n#\n"


def _write_entry(dep, pid, chains, fmt):
    if fmt == "pdb":
        (dep / f"{pid}.pdb").write_text(_pdb_lines(chains))
    else:
        with gzip.open(dep / f"{pid}.cif.gz", "wt") as fh:
            fh.write(_cif_text(chains))


def _fake_misato(root, rng, fmt="cif"):
    """misato_real_bb, ca5 / ca8 pickles and deposited entries of 1AAA ... 5EEE:
    1AAA see above; 2BBB one molecule of 10 + 10 whose deposited entry has other chains
    (unresolved); 3CCC one deposited chain whose two parts are 25 Å apart (physical
    boundary); 4DDD homo-dimer X, Y in one molecule, each with a 2-residue gap whose left
    flank residue differs from the deposited one (alignment fallback; X|Y boundary names
    the copies); 5EEE two MISATO copies of the single deposited chain Z (extra_copies);
    6FFF one deposited chain with a 2-residue gap whose two parts drifted 25 Å apart in the
    MISATO frames but are close in the deposited coordinates (missing_gap: the physical
    check uses the deposited Cα); 7GGG one deposited chain L split by MISATO into four
    molecules of 8, 1, 2 and 9 residues at one-residue gaps (the 1- and 2-residue
    molecules are placed between their long neighbours: one piece); 8HHH a chain M whose
    SEQRES gap MISATO closed with a bond (bonded label gap) + a chain N molecule.
    Deposited Cα = MISATO frame-0 Cα unless given."""
    from ensembits.data_versions import MISATO_PARSER_VERSION
    cache = root / "data" / "cached_descriptors"; rb = cache / "misato_real_bb"
    dep = root / "deposited"; rb.mkdir(parents=True); dep.mkdir(exist_ok=True)
    s1 = _seq(rng, LTOT); s2 = _seq(rng, 20); s3 = _seq(rng, 20); sx = _seq(rng, 20)
    sz = _seq(rng, 15); s6 = _seq(rng, 20)
    lab_6 = list(range(1, 11)) + list(range(13, 23))
    dep6 = _helix(lab_6, np.random.default_rng(7), np.zeros(3))[0][:, 1]   # undrifted
    s7 = _seq(rng, 20); lab_7 = list(range(1, 9)) + [10] + [12, 13] + list(range(15, 24))
    s8 = _seq(rng, 29); lab_8 = list(range(1, 11)) + list(range(12, 21))
    lab_x = list(range(1, 11)) + list(range(13, 23))
    sx_mis = sx[:9] + ("A" if sx[9] != "A" else "C") + sx[10:]      # flank residue mutated

    def frames_of(make):
        return [make() for _ in range(8)]
    makers = {
        "1AAA": (lambda: _complex_1aaa(rng), s1),
        "2BBB": (lambda: (lambda x, y: (np.concatenate([x[0], y[0]]), np.concatenate([x[1], y[1]]),
                                        np.zeros(20, np.int32)))(
            _chain(10, rng, np.zeros(3)), _chain(10, rng, np.array([25.0, 0, 0]))), s2),
        "3CCC": (lambda: (*_helix(list(range(1, 11)) + list(range(13, 23)), rng, np.zeros(3),
                                  offset_at=10, offset=np.array([25.0, 0, 0])),
                          np.zeros(20, np.int32)), s3),
        "4DDD": (lambda: (lambda x, y: (np.concatenate([x[0], y[0]]), np.concatenate([x[1], y[1]]),
                                        np.zeros(40, np.int32)))(
            _helix(lab_x, rng, np.zeros(3)), _helix(lab_x, rng, np.array([40.0, 0, 0]))),
            sx_mis + sx_mis),
        "5EEE": (lambda: (lambda x, y: (np.concatenate([x[0], y[0]]), np.concatenate([x[1], y[1]]),
                                        np.array([0] * 15 + [1] * 15, np.int32)))(
            _chain(15, rng, np.zeros(3)), _chain(15, rng, np.array([40.0, 0, 0]))), sz + sz),
        "6FFF": (lambda: (*_helix(lab_6, rng, np.zeros(3), offset_at=10,
                                  offset=np.array([25.0, 0, 0])), np.array([0] * 10 + [1] * 10, np.int32)), s6),
        "7GGG": (lambda: (*_helix(lab_7, rng, np.zeros(3)),
                          np.array([0] * 8 + [1] + [2] * 2 + [3] * 9, np.int32)), s7),
        "8HHH": (lambda: (lambda x, y: (np.concatenate([x[0], y[0]]), np.concatenate([x[1], y[1]]),
                                        np.array([0] * 19 + [1] * 10, np.int32)))(
            _chain(19, rng, np.zeros(3)), _chain(10, rng, np.array([40.0, 0, 0]))), s8),
    }
    ca5, ca8, ref = {}, {}, {}
    for pid, (make, seq) in makers.items():
        frames = frames_of(make)
        bb8 = np.stack([f[0] for f in frames]); cb8 = np.stack([f[1] for f in frames])
        np.savez(rb / f"{pid}.npz", bb_8=bb8, cb_8=cb8, ca_8=bb8[:, :, 1], seq=np.array(seq),
                 chain_index=frames[0][2], parser_version=np.array(MISATO_PARSER_VERSION),
                 K=np.array(8), fps=np.array("kabsch"))
        ca8[pid] = bb8[:, :, 1].copy(); ca5[pid] = bb8[:5, :, 1].copy(); ref[pid] = (bb8, cb8)
    pickle.dump(ca5, open(cache / "coords_ca5_misato.pkl", "wb"))
    pickle.dump(ca8, open(cache / "coords_ca8_misato.pkl", "wb"))
    f0 = {pid: ref[pid][0][0][:, 1] for pid in ref}                   # frame-0 Cα
    xyz = lambda pid, i: tuple(float(v) for v in f0[pid][i])
    ent = {
        "1AAA": [("A", [(a, n, "", n, xyz("1AAA", i)) for i, (a, n) in enumerate(zip(s1[:LA], LAB_A))]),
                 ("B", [(a, n, "", n, xyz("1AAA", LA + i)) for i, (a, n) in
                        enumerate(zip(s1[LA:LA + LB], range(1, LB + 1)))]),
                 ("C", [(a, au[0], au[1], lab, xyz("1AAA", LA + LB + i)) for i, (a, au, lab) in
                        enumerate(zip(s1[LA + LB:], AUTH_C, LAB_C))])],
        "2BBB": [("A", [(a, n, "", n, (0.0, 0.0, 3.8 * n)) for a, n in
                        zip(_seq(np.random.default_rng(99), 20), range(1, 21))])],
        "3CCC": [("P", [(a, n, "", n, xyz("3CCC", i)) for i, (a, n) in
                        enumerate(zip(s3, list(range(1, 11)) + list(range(13, 23))))])],
        "4DDD": [(c, [(a, n, "", n, xyz("4DDD", o + i)) for i, (a, n) in enumerate(zip(sx, lab_x))])
                 for c, o in (("X", 0), ("Y", 20))],
        "5EEE": [("Z", [(a, n, "", n, xyz("5EEE", i)) for i, (a, n) in enumerate(zip(sz, range(1, 16)))])],
        "6FFF": [("Q", [(a, n, "", n, tuple(float(v) for v in dep6[i])) for i, (a, n) in
                        enumerate(zip(s6, lab_6))])],
        "7GGG": [("L", [(a, n, "", n, xyz("7GGG", i)) for i, (a, n) in enumerate(zip(s7, lab_7))])],
        "8HHH": [("M", [(a, n, "", n, xyz("8HHH", i)) for i, (a, n) in enumerate(zip(s8[:19], lab_8))]),
                 ("N", [(a, n, "", n, xyz("8HHH", 19 + i)) for i, (a, n) in
                        enumerate(zip(s8[19:], range(1, 11)))])],
    }
    for pid, ch in ent.items():
        _write_entry(dep, pid, ch, fmt)
    return cache, dep, ref


def _env(root, **extra):
    e = dict(os.environ)
    e["ENSEMBITS_REPO_ROOT"] = str(root)
    e["ENSEMBITS_MDCATH_DIV_SRC"] = str(Path(root) / "data" / "external" / "mdcath_div")
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.update(extra)
    return e


def _run(root, module, argv, folder=BASELINES):
    code = (f"import sys; sys.path.insert(0, {str(folder)!r}); import {module} as M; "
            f"sys.exit(M.main({argv!r}))")
    return subprocess.run([sys.executable, "-c", code], env=_env(root), capture_output=True,
                          text=True, timeout=600)


def _build_breaks(root, dep, extra=()):
    return _run(root, "build_misato_chain_breaks", ["--deposited-dir", str(dep), "--workers", "1",
                                                     *extra], CODE_ROOT / "data_prep")


def _mini3di_tokens(bb, cb_real, pieces):
    import mini3di
    enc = mini3di.Encoder()
    cb = cb_from_bb_or_real(bb, cb_real)
    out = []
    for s, e in pieces:
        st = enc.encode_atoms(bb[s:e, 1].astype(np.float32), cb[s:e].astype(np.float32),
                              bb[s:e, 0].astype(np.float32), bb[s:e, 2].astype(np.float32))
        out.append(np.asarray(st.filled(), dtype=np.uint8))
    return np.concatenate(out)


# ---------------------------------------------------------------- classification + pieces
def test_chain_breaks_classification_v2(tmp_path):
    cache, dep, _ = _fake_misato(tmp_path, np.random.default_rng(0), "cif")
    r = _build_breaks(tmp_path, dep)
    assert r.returncode == 0, r.stdout + r.stderr
    out = cache / "misato_chain_breaks.json"
    t = json.loads(out.read_text())
    assert t["schema"] == "misato-chain-breaks-v2"
    b1 = {b["pos"]: b for b in t["complexes"]["1AAA"]["breaks"]}
    assert (b1[6]["kind"], b1[6]["n_missing"], b1[6]["left_auth"], b1[6]["right_auth"],
            b1[6]["left_label"], b1[6]["right_label"]) == ("missing_gap", 3, "6", "10", 6, 10)
    assert b1[LA]["kind"] == "boundary" and b1[LA]["source"] == "deposited"     # hidden boundary
    assert (b1[LA]["left_chain"], b1[LA]["right_chain"]) == ("A", "B")
    assert b1[LA + LB]["kind"] == "boundary" and b1[LA + LB]["source"] == "chain_index"
    assert b1[LA + LB + 5]["kind"] == "consecutive" and b1[LA + LB + 5]["label_step"] == 1
    # (a) a chain_index change inside one deposited chain is placed: missing_gap, kept whole;
    # (b) its step is the SEQRES step 5 although the author numbers are consecutive (8 -> 9)
    c = b1[LA + LB + LC1]
    assert (c["origin"], c["source"], c["kind"], c["n_missing"], c["label_step"]) == \
        ("chain_index", "chain_index", "missing_gap", 4, 5)
    assert (c["left_auth"], c["right_auth"]) == ("8", "9")
    assert [(b["pos"], b["kind"]) for b in t["complexes"]["2BBB"]["breaks"]] == [(10, "unresolved")]
    b3 = t["complexes"]["3CCC"]["breaks"]
    assert len(b3) == 1 and b3[0]["kind"] == "boundary" and b3[0]["source"] == "physical"
    assert b3[0]["n_missing"] == 2 and b3[0]["excess"] > 10
    assert b3[0]["physical_basis"] == "deposited" and b3[0]["deposited_ca"] > 20
    # v2.1: drifted apart in the MISATO frames only (separate molecules), close in the
    # deposited entry -> one polypeptide, kept whole
    b6 = t["complexes"]["6FFF"]["breaks"]
    assert len(b6) == 1 and (b6[0]["kind"], b6[0]["n_missing"], b6[0]["origin"]) == \
        ("missing_gap", 2, "chain_index")
    assert b6[0]["frame0_ca"] > b6[0]["ca_limit"] >= b6[0]["deposited_ca"] and b6[0]["excess"] < 0
    b4 = {b["pos"]: b for b in t["complexes"]["4DDD"]["breaks"]}
    for pos, ch in ((10, "X"), (30, "Y")):          # mutated flank: alignment fallback
        assert (b4[pos]["kind"], b4[pos]["n_missing"], b4[pos]["placed_by"],
                b4[pos]["left_chain"], b4[pos]["right_chain"]) == ("missing_gap", 2, "alignment", ch, ch)
    assert (b4[20]["kind"], b4[20]["left_chain"], b4[20]["right_chain"]) == ("boundary", "X", "Y")
    t5 = t["complexes"]["5EEE"]
    assert t5["extra_copies"] is True and [b["kind"] for b in t5["breaks"]] == ["boundary"]
    assert t["complexes"]["1AAA"]["extra_copies"] is False
    side = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert side["n_physical_boundary"] == 1 and side["n_complexes_extra_copies"] == 1
    assert set(side["deposited_files"]) == {f"{p}.cif.gz" for p in
                                            ("1AAA", "2BBB", "3CCC", "4DDD", "5EEE", "6FFF",
                                             "7GGG", "8HHH")}
    # v2.3 (i): 1- and 2-residue molecules placed between their long neighbours
    b7 = t["complexes"]["7GGG"]["breaks"]
    assert [(b["pos"], b["kind"], b["n_missing"], b["source"]) for b in b7] == \
        [(8, "missing_gap", 1, "chain_index"), (9, "missing_gap", 1, "chain_index"),
         (11, "missing_gap", 1, "chain_index")]
    assert CP.chain_pieces("misato", "7GGG", 20, table=t["complexes"])[0] == [(0, 20)]
    # v2.3 (iii): a SEQRES gap closed by a bond in MISATO is counted, not cut
    assert t["complexes"]["8HHH"]["n_bonded_label_gaps"] == 1
    assert [b["kind"] for b in t["complexes"]["8HHH"]["breaks"]] == ["boundary"]
    assert side["n_physical_basis"] == {"deposited": side["n_physical_basis"]["deposited"]}
    assert _build_breaks(tmp_path, dep).returncode == 1                      # refuses overwrite
    # pieces and SEQRES numbering
    pieces, ri = CP.chain_pieces("misato", "1AAA", LTOT, table=t["complexes"])
    assert pieces == WANT_PIECES and np.array_equal(ri, WANT_RI)
    pieces, ri = CP.chain_pieces("misato", "4DDD", 40, table=t["complexes"])
    assert pieces == [(0, 20), (20, 40)] and ri[:20].tolist() == list(range(10)) + list(range(12, 22))
    pieces, _ = CP.chain_pieces("misato", "3CCC", 20, table=t["complexes"])
    assert pieces == [(0, 10), (10, 20)]
    with pytest.raises(ValueError, match="L="):
        CP.chain_pieces("misato", "1AAA", LTOT + 1, table=t["complexes"])
    with pytest.raises(ValueError, match="not in"):
        CP.chain_pieces("misato", "9ZZZ", 5, table=t["complexes"])


def test_chain_breaks_pdb_fallback(tmp_path):
    # PDB-format entries have no SEQRES position: author numbers are used and flagged
    cache, dep, _ = _fake_misato(tmp_path, np.random.default_rng(0), "pdb")
    assert _build_breaks(tmp_path, dep).returncode == 0
    t = json.loads((cache / "misato_chain_breaks.json").read_text())["complexes"]
    b1 = {b["pos"]: b for b in t["1AAA"]["breaks"]}
    assert b1[6]["n_missing_source"] == "auth_seq_id (PDB fallback)" and b1[6]["n_missing"] == 3
    c = b1[LA + LB + LC1]           # auth step 1 over the deletion: caught by the physical check
    assert (c["kind"], c["source"], c["label_step"]) == ("boundary", "physical", 1)


def test_placement_rules_units():
    import build_misato_chain_breaks as B
    # a flank cut by the sequence end never places a break alone
    ch = {"A": [(a, i + 1, "", i + 1) for i, a in enumerate("ACDEFGHIKLMNPEC")]}
    assert B.flank_place("ACDEFGHIKLMNPEC", 13, ch) is None
    assert B.flank_place("ACDEFGHIKLMNPEC", 6, ch)[0] == "same"
    # physical check: too far for one polypeptide -> boundary; renumbered deletion
    # v2.1: the check uses the deposited Cα (frame 0 only as a flagged fallback)
    k = B._kind_from_step(1, 13.0, 13.0, True, dep_ca=12.0, dep_step=1)
    assert k["kind"] == "boundary" and k["physical"] and k["physical_basis"] == "deposited"
    k = B._kind_from_step(3, 25.0, 24.0, True, dep_ca=7.0, dep_step=3)      # MD drift only
    assert (k["kind"], k["n_missing"], k["frame0_ca"], k["deposited_ca"]) == ("missing_gap", 2, 25.0, 7.0)
    assert k["ca_limit"] == pytest.approx(3.80 * 3 + 0.5) and k["excess"] < 0
    k = B._kind_from_step(1, 13.0, 13.0, True)                              # no deposited Cα
    assert k["kind"] == "boundary" and k["physical_basis"] == "frame0"
    k = B._kind_from_step(1, 4.2, 4.5, True)
    assert k["kind"] == "consecutive" and k.get("renumbered_deletion")
    k = B._kind_from_step(1, 4.2, 4.5, True, dep_ca=3.8, dep_step=1)       # deposited: consecutive
    assert k["kind"] == "consecutive" and not k.get("renumbered_deletion")
    k = B._kind_from_step(12, 7.6, 8.8, True, dep_ca=7.0, dep_step=12)
    assert (k["kind"], k["n_missing"], k["n_missing_source"]) == ("missing_gap", 11, "label_seq_id")
    # v2.2: step 1 at deposited 5.0 Å = strained bond (consecutive); at 8.2 Å = boundary
    k = B._kind_from_step(1, 5.5, 4.8, True, dep_ca=5.0, dep_step=1)
    assert (k["kind"], k["n_missing"], k.get("strained_bond")) == ("consecutive", 0, True)
    k = B._kind_from_step(1, 8.0, 8.0, True, dep_ca=8.2, dep_step=1)
    assert k["kind"] == "boundary" and k["physical"] and not k.get("strained_bond")
    k = B._kind_from_step(2, 8.6, 9.4, True, dep_ca=8.6, dep_step=2)       # just over 8.1 Å
    assert k["kind"] == "boundary" and 0 < k["excess"] < 3
    assert B._kind_from_step(0, 3.8, 3.0, True)["kind"] == "unresolved"


def test_geometric_choice_units():
    # v2.3 (ii): two deposited chains with the same sequence, different coordinates; the
    # MISATO windows lie on chain Y -> geometry places the break there (runner-up X)
    import build_misato_chain_breaks as B
    rng = np.random.default_rng(11)
    sq = _seq(rng, 24)
    lab = list(range(1, 13)) + list(range(14, 26))
    line = np.stack([np.zeros(24), np.zeros(24), 3.8 * np.arange(24.0)], 1)
    line[12:] += np.array([0.0, 2.0, 2.0])
    hx = line + rng.normal(scale=1.2, size=line.shape)          # same chain, other geometry
    chains = {"X": [(a, n, "", n, tuple(hx[i])) for i, (a, n) in enumerate(zip(sq, lab))],
              "Y": [(a, n, "", n, tuple(line[i])) for i, (a, n) in enumerate(zip(sq, lab))]}
    dep = B._Dep(chains)
    g = B.geom_place(sq, line.copy(), 12, 0, 24, dep, {})
    assert g["left"] == ("Y", 11) and g["right"] == ("Y", 12) and g["rmsd"] < 0.1
    assert g["runner_up"]["rmsd"] > 0.5 and g["n_pairs"] >= 2
    seq_p = {"kind": "missing_gap", "placed_by": "flank", "left_chain": "X", "right_chain": "X",
             "left_label": 12, "right_label": 14}
    p = B.geom_choose(seq_p, g, chains, 7.0, 6.0, True)
    assert (p["placed_by"], p["left_chain"], p["kind"], p["n_missing"]) == ("geometry", "Y", "missing_gap", 1)
    assert p["sequence_placement"]["left_chain"] == "X" and p["geom_runner_up"]["rmsd"] > p["geom_rmsd"]
    # a tie (symmetric copies) keeps the sequence placement
    g2 = dict(g, near=[(("Y", 11), ("Y", 12)), (("X", 11), ("X", 12))])
    assert B.geom_choose(seq_p, g2, chains, 7.0, 6.0, True)["placed_by"] == "flank"


def test_pieces_from_breaks_rules():
    br = [{"pos": 3, "kind": "missing_gap", "n_missing": 4, "ca": 9.0},
          {"pos": 5, "kind": "missing_gap", "n_missing": None, "ca": 11.6},   # round(11.6/3.8)=3
          {"pos": 7, "kind": "consecutive", "n_missing": 0, "ca": 4.5},
          {"pos": 8, "kind": "unresolved", "n_missing": None, "ca": 20.0},
          {"pos": 10, "kind": "boundary", "n_missing": None, "ca": 30.0}]
    pieces, ri = CP.pieces_from_breaks(12, br)
    assert pieces == [(0, 8), (8, 10), (10, 12)]
    assert ri.tolist() == [0, 1, 2, 7, 8, 11, 12, 13, 0, 1, 0, 1]
    with pytest.raises(ValueError):
        CP.pieces_from_breaks(5, [{"pos": 2, "kind": "weird", "ca": 1.0}])
    for ds in ("mdcath", "mdcath_div", "bioemu"):
        p, r = CP.chain_pieces(ds, "any", 17)
        assert p == [(0, 17)] and r.tolist() == list(range(17))
    assert CP.check_pieces([(0, 3), (3, 5)], 5) == [(0, 3), (3, 5)]
    for bad in ([(0, 3)], [(0, 3), (4, 5)], [(0, 0), (0, 5)]):
        with pytest.raises(ValueError):
            CP.check_pieces(bad, 5)


# ---------------------------------------------------------------- mini3di builders
def test_mini3di_builders_misato_true_chains(tmp_path):
    cache, dep, ref = _fake_misato(tmp_path, np.random.default_rng(1))
    assert _build_breaks(tmp_path, dep).returncode == 0
    r = _run(tmp_path, "build_mini3di_misato", ["--dataset", "misato"])
    assert r.returncode == 0, r.stdout + r.stderr
    r = _run(tmp_path, "build_mini3di_tokens_K_misato", ["--dataset", "misato"])
    assert r.returncode == 0, r.stdout + r.stderr
    tok1 = np.load(cache / "mini3di_tokens_misato.npz")
    tokK = np.load(cache / "mini3di_tokens_K8_misato.npz")
    pieces = {"1AAA": WANT_PIECES, "2BBB": [(0, 10), (10, 20)], "3CCC": [(0, 10), (10, 20)],
              "4DDD": [(0, 20), (20, 40)], "5EEE": [(0, 15), (15, 30)], "6FFF": [(0, 20)],
              "7GGG": [(0, 20)], "8HHH": [(0, 19), (19, 29)]}
    for pid, (bb8, cb8) in ref.items():
        assert np.array_equal(tok1[pid], _mini3di_tokens(bb8[0], cb8[0], pieces[pid]))
        for f in range(8):
            assert np.array_equal(tokK[pid][:, f], _mini3di_tokens(bb8[f], cb8[f], pieces[pid]))
    old = _mini3di_tokens(ref["1AAA"][0][0], ref["1AAA"][1][0],
                          split_chains_by_peptide_bond(ref["1AAA"][0][0]))
    assert not np.array_equal(tok1["1AAA"], old)        # the 2.5 Å split cut the gaps too
    sha = RC.file_record(cache / "misato_chain_breaks.json")["sha256"]
    for f in ("mini3di_tokens_misato.npz", "mini3di_tokens_K8_misato.npz"):
        side = json.loads((cache / f"{f}.provenance.json").read_text())
        assert side["chain_split"] == "true_chain_v2" and side["chain_breaks"]["sha256"] == sha
    code = ("import sys, json; sys.path.insert(0, %r); import rmsf_common as RC\n"
            "c = RC.CACHE / 'mini3di_tokens_misato.npz'\n"
            "rec = RC.check_cache_provenance(c, 'misato', frames='frame 0')\n"
            "assert rec['chain_split'] == 'true_chain_v2' and rec['chain_breaks_sha256']\n"
            "t = RC.MISATO_CHAIN_BREAKS; t.write_text(json.dumps(json.loads(t.read_text()), indent=1))\n"
            "try:\n    RC.check_cache_provenance(c, 'misato', frames='frame 0'); print('BAD')\n"
            "except RC.CacheError as e: assert 'another misato_chain_breaks' in str(e)\n"
            "print('OK')") % str(CODE_ROOT / "downstream")
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout and "BAD" not in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("dataset", ["mdcath", "bioemu"])
def test_mini3di_single_chain_datasets_unchanged(tmp_path, dataset):
    rng = np.random.default_rng(2)
    cache = tmp_path / "data" / "cached_descriptors"
    ref = {}
    for i in range(2):
        frames = [_chain(20, rng, np.zeros(3)) for _ in range(10)]
        bb10 = np.stack([f[0] for f in frames]); cb10 = np.stack([f[1] for f in frames])
        ref[f"s{i}"] = (bb10, cb10)
    if dataset == "mdcath":
        src = tmp_path / "data" / "external" / "mdcath_div"; src.mkdir(parents=True)
        rb = cache / "mdcath_real_bb"; rb.mkdir(parents=True)
        (src / "_manifest.json").write_text(json.dumps({"representatives": {p: {} for p in ref}}))
        for p, (bb10, cb10) in ref.items():
            np.savez(src / f"{p}.npz", ca_10=bb10[:, :, 1], ca_5=bb10[:5, :, 1])
            np.savez(rb / f"{p}.npz", bb_10=bb10, cb_10=cb10, seq=np.array("A" * 20))
    else:
        rb = cache / "bioemu_cath_real_bb"; ex = cache / "bioemu_cath_extras"
        rb.mkdir(parents=True); ex.mkdir(parents=True)
        (rb / "layout_rows.json").write_text(json.dumps([{"system": p} for p in ref]))
        for p, (bb10, cb10) in ref.items():
            np.savez(rb / f"{p}.npz", bb_10=bb10)
            np.savez(ex / f"{p}.npz", cb_10=cb10, seq=np.array("A" * 20))
    for mod in ("build_mini3di_misato", "build_mini3di_tokens_K_misato"):
        r = _run(tmp_path, mod, ["--dataset", dataset])
        assert r.returncode == 0, r.stdout + r.stderr
    tok1 = np.load(cache / f"mini3di_tokens_{dataset}.npz")
    tokK = np.load(cache / f"mini3di_tokens_K10_{dataset}.npz")
    for p, (bb10, cb10) in ref.items():
        assert split_chains_by_peptide_bond(bb10[0]) == [(0, 20)]      # no gap
        assert np.array_equal(tok1[p], _mini3di_tokens(bb10[0], cb10[0], [(0, 20)]))
        for f in range(10):
            assert np.array_equal(tokK[p][:, f], _mini3di_tokens(
                bb10[f], cb10[f], split_chains_by_peptide_bond(bb10[f])))
    side = json.loads((cache / f"mini3di_tokens_{dataset}.npz.provenance.json").read_text())
    assert side["chain_split"] == "whole protein (single chain)" and side["chain_breaks"] is None


# ---------------------------------------------------------------- residue index -> encoders
class _FakeQuant:
    def __call__(self, z):
        return None, z.sum(-1).long() % 7, None, None


def test_aminoaseed_residue_index_reaches_encoder():
    import torch
    import build_aminoaseed_tokens as B
    seen = []

    class Enc:
        def encode(self, coords, mask, _none, residue_index):
            seen.append(residue_index.squeeze(0).tolist())
            return torch.zeros(1, coords.shape[1], 2)

    model = type("M", (), {"encoder": Enc(), "quantizer": _FakeQuant()})()
    bb, _, _ = _complex_1aaa(np.random.default_rng(3))
    out = B.tokenize_protein(model, bb[:, 1], "cpu", bb=bb, pieces=WANT_PIECES, residue_index=WANT_RI)
    assert len(out) == LTOT
    assert seen == [WANT_RI[s:e].tolist() for s, e in WANT_PIECES]
    seen.clear(); B.tokenize_protein(model, bb[:, 1], "cpu", bb=bb, pieces=[(0, LTOT)])
    assert seen == [list(range(LTOT))]                       # default numbering unchanged
    with pytest.raises(ValueError):
        B.tokenize_protein(model, bb[:, 1], "cpu", bb=bb, pieces=WANT_PIECES, residue_index=WANT_RI[:5])


def test_esm3_residue_index_reaches_encoder():
    import torch
    import build_esm3struct_tokens as B
    seen = []

    class Enc:
        def encode(self, coords, residue_index=None):
            seen.append(None if residue_index is None else residue_index.squeeze(0).tolist())
            return None, torch.zeros(1, coords.shape[1], dtype=torch.long)

    bb, _, _ = _complex_1aaa(np.random.default_rng(4))
    B.tokenize_protein(Enc(), bb[:, 1], "cpu", bb=bb, pieces=WANT_PIECES, residue_index=WANT_RI)
    assert seen == [WANT_RI[s:e].tolist() for s, e in WANT_PIECES]
    seen.clear(); B.tokenize_protein(Enc(), bb[:, 1], "cpu", bb=bb, pieces=[(0, LTOT)])
    assert seen == [None]                                  # ESM3 default (positions 0..L-1)


def test_protoken_residue_numbers(tmp_path):
    import build_protoken_tokens as P
    from ensembits.baselines.protoken import _write_minimal_pdb
    seen = []

    class Tok:
        def tokenize(self, ca, seq, bb=None, residue_numbers=None):
            seen.append(None if residue_numbers is None else list(residue_numbers))
            return np.arange(len(ca))
    bb, _, _ = _complex_1aaa(np.random.default_rng(5))
    seq = "A" * (LTOT)
    idx = P.tokenize_by_chain(Tok(), bb[:, 1], seq, bb, "real", pieces=WANT_PIECES,
                              residue_index=WANT_RI)
    assert len(idx) == LTOT
    assert seen == [(WANT_RI[s:e] + 1).tolist() for s, e in WANT_PIECES]
    seen.clear(); P.tokenize_by_chain(Tok(), bb[:, 1], seq, bb, "real", pieces=[(0, LTOT)])
    assert seen == [None]
    f = tmp_path / "x.pdb"
    _write_minimal_pdb(bb[:LA], "A" * LA, str(f), WANT_RI[:LA] + 1)
    nums = sorted({int(l[22:26]) for l in f.read_text().splitlines() if l.startswith("ATOM")})
    assert nums == (WANT_RI[:LA] + 1).tolist()
    _write_minimal_pdb(bb[:LA], "A" * LA, str(f))
    nums = sorted({int(l[22:26]) for l in f.read_text().splitlines() if l.startswith("ATOM")})
    assert nums == list(range(1, LA + 1))
    with pytest.raises(ValueError):
        _write_minimal_pdb(bb[:3], "AAA", str(f), [3, 2, 1])


# ---------------------------------------------------------------- pooled builders
def _fake_table(tmp_path, monkeypatch):
    t = tmp_path / "misato_chain_breaks.json"
    t.write_text(json.dumps({"schema": "misato-chain-breaks-v2", "complexes": {}}))
    monkeypatch.setattr(RC, "MISATO_CHAIN_BREAKS", t)
    return t


def test_pooled_3di_requires_true_chains(tmp_path, monkeypatch):
    import build_pooled_baseline_features as PB
    from ensembits.data_versions import read_provenance, write_provenance
    _fake_table(tmp_path, monkeypatch)
    monkeypatch.setattr(PB, "CACHE", tmp_path)
    np.save(tmp_path / "mini3di_centroids.npy", np.arange(40, dtype=np.float32).reshape(20, 2))
    k8 = tmp_path / "mini3di_tokens_K8_misato.npz"
    np.savez(k8, P1=np.zeros((5, 8), np.uint8))
    out = tmp_path / "3di_pooled_misato.npz"
    for bad in ({"chain_break_A": 2.5}, {"chain_split": "chain_index"},
                {"chain_split": "true_chain_v2", "chain_breaks": {"sha256": "x"}}):
        write_provenance(k8, bad)
        with pytest.raises(SystemExit):
            PB.pool_3di(out)
        assert not out.exists()
    write_provenance(k8, {**RC.split_record("misato"), "cache_sha256": "x"})
    PB.pool_3di(out)
    assert read_provenance(out)["chain_split"] == "true_chain_v2"


def test_merge_pooled_protoken_requires_chain_split(tmp_path):
    from ensembits.data_versions import read_provenance, write_provenance
    cache = tmp_path / "data" / "cached_descriptors"; cache.mkdir(parents=True)
    t = cache / "misato_chain_breaks.json"
    t.write_text(json.dumps({"schema": "misato-chain-breaks-v2", "complexes": {}}))
    for k in range(2):
        np.savez(cache / f"protoken_pooled_misato_shard{k}of2.npz", **{f"P{k}": np.zeros((3, 32))})
        write_provenance(cache / f"protoken_pooled_misato_shard{k}of2.npz", {"chain_split": "chain_index"})
    argv = ["--pooled", "--num-shards", "2"]
    r = _run(tmp_path, "merge_protoken_shards", argv)
    assert r.returncode == 1 and "chain_split" in r.stdout
    sha = RC.file_record(t)["sha256"]
    for k in range(2):
        write_provenance(cache / f"protoken_pooled_misato_shard{k}of2.npz",
                         {"chain_split": "true_chain_v2", "chain_breaks": {"sha256": sha}})
    r = _run(tmp_path, "merge_protoken_shards", argv)
    assert r.returncode == 0, r.stdout + r.stderr
    side = read_provenance(cache / "protoken_pooled_misato.npz")
    assert side["chain_split"] == "true_chain_v2" and side["n_proteins"] == 2


# ---------------------------------------------------------------- constant predictions
def test_probe_records_constant_prediction():
    import torch
    from probe_rmsf_eval import fit_probe_early_stopping

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(4, 3); self.lin = torch.nn.Linear(3, 1)
        def forward(self, t): return self.lin(self.emb(t)).squeeze(-1)
    rng = np.random.default_rng(0)
    y = lambda n: torch.tensor(rng.gamma(2.0, 1.0, n), dtype=torch.float32)
    one = lambda n: torch.full((n,), 2, dtype=torch.long)
    torch.manual_seed(0)
    res = fit_probe_early_stopping(Probe(), one(60), y(60), one(30), y(30), one(30), y(30),
                                   device="cpu", seed=0, batch_size=16, max_epochs=5, patience=2)
    assert res["constant_prediction"] is True and res["n_unique_inputs"] == 1
    assert res["best_epoch"] == 0 and res["test_spearman"] != res["test_spearman"]
    many = lambda n: torch.tensor(rng.integers(0, 4, n), dtype=torch.long)
    res = fit_probe_early_stopping(Probe(), many(60), y(60), many(30), y(30), many(30), y(30),
                                   device="cpu", seed=0, batch_size=16, max_epochs=5, patience=2)
    assert res["constant_prediction"] is False and res["n_unique_inputs"] is None


def test_constant_seed_detection(tmp_path, monkeypatch):
    nan = float("nan")
    new = {"test_spearman": nan, "constant_prediction": True, "n_unique_inputs": 1}
    assert RC.constant_prediction_seed(new, "test_spearman")
    assert not RC.constant_prediction_seed({**new, "n_unique_inputs": 3}, "test_spearman")
    assert not RC.constant_prediction_seed({**new, "test_spearman": 0.2}, "test_spearman")
    assert not RC.constant_prediction_seed({"test_spearman": nan, "constant_prediction": False},
                                           "test_spearman")
    # results written before 2026-10-18: best_epoch 0 + NaN + one-token token cache
    monkeypatch.setattr(RC, "ROOT", tmp_path); monkeypatch.setattr(RC, "CACHE", tmp_path / "c")
    (tmp_path / "c").mkdir()
    np.savez(tmp_path / "c" / "ensembits_runX_P1inf_tokens_misato.npz",
             A=np.full(5, 7), B=np.full(3, 7))
    np.savez(tmp_path / "c" / "ensembits_runY_P1inf_tokens_misato.npz", A=np.arange(5))
    leg = {"test_spearman": nan, "best_epoch": 0, "p_inference": 1, "checkpoint": {"variant": "runX"}}
    assert RC.constant_prediction_seed(leg, "test_spearman")            # mdCATH / BioEmu style
    assert not RC.constant_prediction_seed({**leg, "checkpoint": {"variant": "runY"}}, "test_spearman")
    assert not RC.constant_prediction_seed({**leg, "best_epoch": 3}, "test_spearman")
    mis = {"test_spearman": nan, "best_epoch": 0,
           "feature_caches": [{"path": "c/ensembits_runX_P1inf_tokens_misato.npz"}]}
    assert RC.constant_prediction_seed(mis, "test_spearman")            # MISATO style


def test_compiler_constant_cells(tmp_path, monkeypatch):
    import _compile_rebuttal_results as CR
    import rmsf_selection_table as SEL
    nan = float("nan")
    monkeypatch.setattr(CR.RC, "check_rmsf_group", lambda *a, **k: [])
    monkeypatch.setattr(CR.RC, "records_changed", lambda *a, **k: [])
    monkeypatch.setattr(CR, "_table_same", lambda *a, **k: [])
    monkeypatch.setattr(CR, "_table_hash", lambda *a, **k: None)
    monkeypatch.setattr(CR, "_table_label", lambda *a, **k: None)
    CR.set_rmsf_protocol("test"); CR.reset_rmsf_checks()

    def files(name, rows):
        out = []
        for s, js in enumerate(rows):
            f = tmp_path / f"{name}_seed{s}.json"; f.write_text(json.dumps(js)); out.append(f)
        return out
    const = files("c", [{"test_spearman": nan, "constant_prediction": True, "n_unique_inputs": 1}] * 10)
    v = CR._rm_agg("misato", "ensembits_x", "seq P=1", const, "test_spearman")
    assert getattr(v, "constant", False) and v[2] == 10
    assert CR.cell(v) == "—†" and CR.check_seeds("t", "r", ["c"], [v]) == []
    assert CR.enforce_seeds([v])[0] is v
    other = CR.agg([0.5, 0.6])
    assert CR._rank_cols([[v], [other]]) == [(1, None)]                 # never ranked
    md_note = [r for r in CR._RM_CONSTANT]
    assert md_note == [("misato", "ensembits_x", "seq P=1")]
    # any other NaN refuses the cell
    mixed = files("m", [{"test_spearman": 0.4}] * 9 + [{"test_spearman": nan, "best_epoch": 0}])
    v = CR._rm_agg("misato", "baseline_y", "seq", mixed, "test_spearman")
    assert getattr(v, "refused", False) and v[2] == 9
    assert any("NaN" in w for d, st_, c, why in CR._RM_REFUSED for w in why)
    ok = files("o", [{"test_spearman": 0.4}] * 10)
    assert CR._rm_agg("misato", "baseline_z", "seq", ok, "test_spearman")[2] == 10
    # selection table: "—†" with the footnote
    cells = {("mdcath_div", "P=1", "pooled"): SEL.Constant((nan, None, 10))}
    assert SEL.fmt(cells[("mdcath_div", "P=1", "pooled")]) == "—† (n=10)"
    rec = {"best_val_recon": None, "hist_epoch": None, "mismatch": False, "snapshot_of": None,
           "status": "finished"}
    md = SEL.markdown([("v", rec, cells)], ["mdcath_div"], "test", "allframes")
    assert RC.CONSTANT_NOTE + "." in md


# ---------------------------------------------------------------- the real v2 table (read-only)
def test_real_table_regressions():
    """Regression cases of the fidelity audits (2026-10-18) on the built table; skipped when
    data/cached_descriptors/misato_chain_breaks.json is not a v2 table."""
    p = RC.MISATO_CHAIN_BREAKS
    if not p.exists():
        pytest.skip("no misato_chain_breaks.json")
    t = json.loads(p.read_text())
    if t.get("schema") != "misato-chain-breaks-v2":
        pytest.skip("not a v2 table")
    T = t["complexes"]
    at = lambda pid, pos: next(b for b in T[pid]["breaks"] if b["pos"] == pos)
    # (a) chain_index changes inside one deposited chain: missing_gap (SEQRES step)
    for pid, pos, n in (("1A2C", 183, 7), ("1GHW", 183, 7), ("3SHC", 175, 7), ("3QTV", 175, 7),
                        ("1D3Q", 175, 7), ("1TA2", 172, 8), ("1SL3", 172, 8), ("1ZGV", 172, 8),
                        ("4XAR", 88, 22), ("2R1Y", 352, 6)):
        b = at(pid, pos)
        assert (b["origin"], b["kind"], b["n_missing"]) == ("chain_index", "missing_gap", n), (pid, b)
    # (b) SEQRES step, not the author numbers (insertion codes, renumbering)
    for pid, pos, n in (("6PYD", 237, 7), ("5TQF", 164, 4), ("4X8U", 163, 3), ("5L30", 163, 3),
                        ("4NG9", 163, 4), ("3MHW", 90, 1), ("4D85", 161, 10), ("1TQF", 161, 13),
                        ("2B8L", 163, 12)):
        b = at(pid, pos)
        assert (b["kind"], b["n_missing"]) == ("missing_gap", n), (pid, b)
    # physical check: one author chain holding two polypeptides -> boundary
    for pid, pos in (("1TA2", 26), ("1D6W", 28), ("1D9I", 28), ("1G37", 27), ("1NM6", 26),
                     ("1NT1", 26), ("1SL3", 26), ("1JWT", 36), ("1OXG", 13)):
        b = at(pid, pos)
        assert (b["kind"], b["source"]) == ("boundary", "physical"), (pid, b)
    # alignment fallback: flank residue differs from the deposited one
    for pid, pos in (("3S9E", 251), ("3U93", 249), ("4U4S", 254), ("4YMB", 250), ("5FHO", 252),
                     ("5M2V", 248), ("6N9T", 216), ("1RTF", 185), ("3TPR", 159)):
        b = at(pid, pos)
        assert b["kind"] == "missing_gap" and b["left_chain"] == b["right_chain"], (pid, b)
    # homo-oligomer: the copies actually assigned
    assert [(at("3V2P", q)["left_chain"], at("3V2P", q)["right_chain"]) for q in (45, 90, 135, 180)] \
        == [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
    b = at("6N9T", 435)          # second copy: E or F (identical copies; v2.3 chooses by geometry)
    assert b["kind"] == "missing_gap" and b["left_chain"] == b["right_chain"] in ("E", "F")
    # v2.3: geometric choice and short molecules
    assert (at("2WTV", 162)["kind"], at("2WTV", 163)["kind"]) == ("missing_gap", "missing_gap")
    b = at("3MG8", 1114)
    assert (b["kind"], b["left_chain"], b["right_chain"], b.get("strained_bond")) == ("consecutive", "J", "J", True)
    for pid in ("2ZP0", "2AEI"):
        assert [(at(pid, q)["kind"], at(pid, q)["n_missing"]) for q in (11, 12)] == \
            [("missing_gap", 1), ("missing_gap", 1)]
