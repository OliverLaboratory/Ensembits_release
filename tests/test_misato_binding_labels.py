"""Tests of the MISATO binding-site labels v2 (benchmark_audit D4) and of the label-file checks.

What it does
------------
Synthetic, CPU-only tests of `data_prep/build_misato_binding_site_labels.py`:
  * PDB parsing: alt-locs, waters, calcium ions named CA (not amino acids), insertion codes
    kept apart; mol2 heavy atoms;
  * the pocket transfer through the sequence alignment on a two-chain complex with an
    author-number restart, an insertion code, a residue MISATO dropped, a non-standard
    residue, a chain MISATO lacks and a missing C-terminal residue (each unaligned pocket
    residue gets its reason), and the old author-number rule labels other residues;
  * the homo-oligomer copy choice (a single MISATO chain aligned to the wrong copy moves to
    the copy whose crystal-ligand neighbourhood matches; an A <-> B swap; an extra MISATO
    copy; the MISATO ligand molecule is never a copy; a symmetric site stays `ambiguous`);
  * the trimmed Kabsch superposition (recovers a rotation despite outliers);
  * `ensembits.misato_binding_labels`: the loader refuses a file without the v2 sidecar or
    with another sha256; result records, `binding_label_problem`, `stale_binding_label_outputs`;
  * the rebuttal compiler refuses binding-site cells built on the old labels (affinity cells
    are not checked), and both probes read the v2 labels and record them.
When `data/annotations/misato_binding_site_v2.npz` exists in the research repo it is checked
too (lengths = misato_real_bb, no all-negative complex, known pocket residues of 1A4W and the
copy fixes of 1LAN, 1XFF and the other swapped complexes). No probe is run and no test metric is read.

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{Binding Site}.

How to run
----------
    python -m pytest tests/test_misato_binding_labels.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "data_prep"))
sys.path.insert(0, str(CODE_ROOT / "downstream"))

import build_misato_binding_site_labels as B  # noqa: E402


def _atom(rec, serial, name, resname, chain, resseq, icode, xyz, el, alt=" "):
    return (f"{rec:<6}{serial:>5} {name:<4}{alt}{resname:>3} {chain}{resseq:>4}{icode}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2}\n")


def _residue_lines(chain, resseq, icode, resname, x0, rec="ATOM", serial=1):
    """N, CA, C, O, H of one residue near x = x0."""
    out = []
    for k, (nm, el) in enumerate((("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("H", "H"))):
        out.append(_atom(rec, serial + k, nm, resname, chain, resseq, icode,
                         (x0 + 0.3 * k, 0.0, 0.0), el))
    return out


def test_parse_pdb_residues():
    lines = (_residue_lines("A", 10, " ", "ALA", 0.0)
             + _residue_lines("A", 10, "A", "GLY", 4.0)                       # insertion code
             + [_atom("ATOM", 20, "CB", "ALA", "A", 10, " ", (9, 9, 9), "C", alt="B")]  # alt B
             + [_atom("HETATM", 21, "CA", "CA", "A", 301, " ", (1, 1, 1), "CA")]        # Ca2+
             + [_atom("HETATM", 22, "O", "HOH", "A", 401, " ", (2, 2, 2), "O")])        # water
    res = B.parse_pdb_residues(lines)
    keys = [r["key"] for r in res]
    assert keys == [("A", 10, ""), ("A", 10, "A"), ("A", 301, "")]
    assert res[0]["heavy"].shape == (4, 3)               # H and the alt-loc B atom skipped
    aa = B.amino_acid_residues(res)
    assert [r["key"] for r in aa] == [("A", 10, ""), ("A", 10, "A")]   # calcium is no CA


def test_parse_mol2_heavy():
    mol2 = ["@<TRIPOS>MOLECULE\n", "x\n", "@<TRIPOS>ATOM\n",
            "1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0\n", "2 H1 1.0 0.0 0.0 H 1 LIG 0.0\n",
            "3 O1 2.0 0.0 0.0 O.2 1 LIG 0.0\n", "@<TRIPOS>BOND\n", "1 1 2 1\n"]
    assert B.parse_mol2_heavy(mol2).tolist() == [[0, 0, 0], [2, 0, 0]]


# ---------------------------------------------------------------- transfer
def _prot(spec):
    """[(chain, resseq, icode, resname)] -> residue dicts as amino_acid_residues returns."""
    return [{"key": (c, n, i), "resname": rn, "ca": (float(k), 0.0, 0.0),
             "heavy": np.zeros((1, 3))} for k, (c, n, i, rn) in enumerate(spec)]


THREE = {v: k for k, v in B.ONE.items() if k in B.STANDARD and k not in ("HID", "HIE", "HIP", "CYX")}


def test_transfer_pocket_two_chains():
    a_seq = "MKWAYIAKQRDEFGHWLPNS"                    # chain A, author 10..28, 15A inserted
    a_nums = [(n, "") for n in range(10, 16)] + [(15, "A")] + [(n, "") for n in range(16, 29)]
    spec = [("A", n, i, THREE[c]) for (n, i), c in zip(a_nums, a_seq)]
    spec[16] = ("A", 25, "", "TPO")                   # phospho-Thr: non-standard, MISATO lacks it
    b_seq = "GSHMLEDPVC"                               # chain B, author numbering restarts at 1
    spec += [("B", n, "", THREE[c]) for n, c in zip(range(1, 11), b_seq)]
    spec += [("C", 1, "", "TRP"), ("C", 2, "", "TRP")]   # chain C, absent from MISATO
    prot = _prot(spec)
    pseq = "".join(B.ONE.get(r["resname"], "X") for r in prot)
    # MISATO: chain A without Lys16 and TPO25, chain B without its last residue (Cys10)
    mseq = "MKWAYIA" + "QRDEFGHW" + "PNS" + "GSHMLEDPV"
    pairs = B.align_pairs(mseq, pseq)
    pocket = [(("A", 15, "A"), "ALA"), (("A", 18, ""), "ARG"), (("B", 2, ""), "SER"),
              (("A", 25, ""), "TPO"), (("A", 16, ""), "LYS"), (("C", 1, ""), "TRP"),
              (("B", 10, ""), "CYS"), (("D", 5, ""), "GLY")]
    labels, pos, un = B.transfer_pocket(pocket, prot, pairs, mseq, len(mseq))
    # A15A (Ala) is MISATO index 6, A18 (Arg) index 8, B2 (Ser) index 19
    assert np.flatnonzero(labels).tolist() == [6, 8, 19]
    assert {(i, k) for i, k, _ in pos} == {(6, ("A", 15, "A")), (8, ("A", 18, "")),
                                          (19, ("B", 2, ""))}
    why = {k: w for k, _, w in un}
    assert why == {("A", 25, ""): "nonstandard_residue", ("A", 16, ""): "internal_gap",
                   ("C", 1, ""): "chain_absent", ("B", 10, ""): "chain_terminus",
                   ("D", 5, ""): "not_in_protein_pdb"}
    # the old rule (author number == MISATO position + 1, chains ignored) labels other residues
    old = np.zeros(len(mseq), np.int8)
    for (c, n, i), _ in pocket:
        if 1 <= n <= len(mseq):
            old[n - 1] = 1
    assert not np.array_equal(old, labels)


def test_transfer_pocket_type_mismatch():
    prot = _prot([("A", 1, "", "ALA"), ("A", 2, "", "GLY"), ("A", 3, "", "SER"),
                  ("A", 4, "", "LEU")])
    pairs = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    labels, pos, un = B.transfer_pocket([(("A", 2, ""), "GLY")], prot, pairs, "AWSL", 4)
    assert labels.sum() == 0 and un[0][2] == "type_mismatch"


# ---------------------------------------------------------------- copies
SEQ10 = "ACDEFGHIKL"


def _copies(chains="AB"):
    spec = [(c, n, "", THREE[x]) for c in chains for n, x in zip(range(1, 11), SEQ10)]
    return _prot(spec)


def test_choose_copies_moves_single_chain_to_ligand_copy():
    prot = _copies("AB")                                         # PDBbind: copies A, B
    mseq = SEQ10                                                 # MISATO: one copy
    pairs = np.array([[i, 10 + i] for i in range(10)])          # aligned to copy B
    dmis = np.full(10, 20.0); dmis[[2, 3, 4]] = 4.0              # MISATO ligand near 2-4
    dcry = np.full(20, 20.0); dcry[[2, 3, 4]] = 4.0              # crystal ligand near A 3-5
    new, info = B.choose_copies(pairs, prot, np.zeros(10, int), dmis, dcry, mseq)
    assert new[:, 1].tolist() == list(range(10))                 # moved to copy A
    assert {r["chain"] for r in info["reassigned"]} == {"A", "B"}
    assert info["supported"] == [(0, 10)]


def test_choose_copies_swap():
    """A <-> B swap (1XFF): MISATO keeps the copy order, the ligand sits on the other copy."""
    prot = _copies("AB")
    mseq = SEQ10 + SEQ10
    pairs = np.array([[i, i] for i in range(20)])                # A -> 0-9, B -> 10-19
    dmis = np.full(20, 20.0); dmis[12:17] = 4.0                  # MISATO ligand at copy 2
    dcry = np.full(20, 20.0); dcry[2:7] = 4.0                    # crystal ligand at chain A
    new, info = B.choose_copies(pairs, prot, np.array([0] * 10 + [1] * 10), dmis, dcry, mseq)
    m = dict((int(i), int(j)) for i, j in new)
    assert all(m[10 + k] == k for k in range(10))                # A on the second copy
    assert all(m[k] == 10 + k for k in range(10))                # B on the first copy
    labels, _, _ = B.transfer_pocket([(("A", 3, ""), "ASP"), (("A", 4, ""), "GLU")], prot,
                                     new, mseq, 20)
    assert np.flatnonzero(labels).tolist() == [12, 13]           # next to the MISATO ligand


def test_choose_copies_extra_misato_copy_and_exclude():
    """MISATO holds two copies of PDBbind's single chain (6IM4); a copy that is the MISATO
    ligand molecule itself (4J44) is never used."""
    prot = _copies("A")
    mseq = SEQ10 + SEQ10
    pairs = np.array([[i, i] for i in range(10)])                # global: A -> first copy
    dmis = np.full(20, 20.0); dmis[12:17] = 4.0                  # (gain 5 > bonus 3)
    dcry = np.full(10, 20.0); dcry[2:7] = 4.0
    new, _ = B.choose_copies(pairs, prot, np.array([0] * 10 + [1] * 10), dmis, dcry, mseq)
    assert new[:, 0].tolist() == list(range(10, 20))
    new, _ = B.choose_copies(pairs, prot, np.array([0] * 10 + [1] * 10), dmis, dcry, mseq,
                             exclude=range(10, 20))
    assert np.array_equal(new, pairs)


def test_choose_copies_symmetric_site_kept():
    prot = _copies("AB")
    mseq = SEQ10 + SEQ10
    pairs = np.array([[i, i] for i in range(20)])
    dmis = np.full(20, 20.0); dmis[[2, 3, 12, 13]] = 4.0
    dcry = np.full(20, 20.0); dcry[[2, 3, 12, 13]] = 4.0         # ligand on the two-fold axis
    new, info = B.choose_copies(pairs, prot, np.array([0] * 10 + [1] * 10), dmis, dcry, mseq)
    assert np.array_equal(new, pairs) and not info["reassigned"]
    assert sorted(info["ambiguous"]) == [(0, 10), (10, 20)]


def test_trimmed_superpose():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 3)) * 10
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Q = P @ R.T + np.array([1.0, -2.0, 3.0])
    Q[:5] += 15.0                                                 # outliers
    R2, t2, dev = B.trimmed_superpose(P, Q, np.ones(50, bool))
    assert np.allclose(R2, R, atol=1e-6) and np.all(dev[5:] < 1e-6) and np.all(dev[:5] > 10)


# ---------------------------------------------------------------- label module
def _label_file(tmp_path, schema="misato-binding-site-v3"):
    from ensembits.data_versions import write_provenance
    from ensembits.featurize import file_sha256
    p = tmp_path / "misato_binding_site_v3.npz"
    with open(p, "wb") as fh:
        np.savez(fh, AAAA=np.array([0, 1, 1, 0], np.int8))
    write_provenance(p, {"schema": schema, "sha256": file_sha256(p)})
    return p


def test_label_module(tmp_path):
    import ensembits.misato_binding_labels as BL
    p = _label_file(tmp_path)
    lab = BL.load_binding_site_labels(p)
    assert lab["AAAA"].dtype == np.float32 and lab["AAAA"].tolist() == [0, 1, 1, 0]
    rec = BL.binding_labels_record(p)
    assert rec["binding_labels_schema"] == "misato-binding-site-v3"
    assert BL.binding_label_problem(rec, p) is None
    assert "old misato_binding_site.npz" in BL.binding_label_problem({}, p)
    assert "are not the current" in BL.binding_label_problem({"binding_labels_sha256": "0" * 64}, p)
    good = tmp_path / "good.json"; good.write_text(json.dumps({"results": {"x_binding": {}}, **rec}))
    old = tmp_path / "old.json"; old.write_text(json.dumps({"results": {"x_binding": {}}}))
    aff = tmp_path / "aff.json"; aff.write_text(json.dumps({"results": {"x_affinity": {}}}))
    stale = BL.stale_binding_label_outputs([good, old, aff, tmp_path / "missing.json"], p)
    assert [o.name for o, _ in stale] == ["old.json"]
    # an old-style file (no v2 sidecar) or a changed file is refused
    q = tmp_path / "v1.npz"
    with open(q, "wb") as fh:
        np.savez(fh, AAAA=np.zeros(4, np.int8))
    with pytest.raises(SystemExit, match="schema"):
        BL.load_binding_site_labels(q)
    with open(p, "wb") as fh:
        np.savez(fh, AAAA=np.ones(4, np.int8))
    with pytest.raises(SystemExit, match="sha256"):
        BL.load_binding_site_labels(p)


def test_compiler_refuses_old_binding_labels(tmp_path, monkeypatch):
    import ensembits.paths as P
    import ensembits.misato_binding_labels as BL
    import _compile_rebuttal_results as CR
    from ensembits.data_versions import misato_splits_record
    sp = tmp_path / "misato_splits_v2.json"
    sp.write_text(json.dumps({"stats": {"schema": "misato-splits-v2"}}))
    monkeypatch.setattr(P, "MISATO_SPLITS", sp)
    lab = _label_file(tmp_path)
    monkeypatch.setattr(BL, "MISATO_BINDING_SITE", lab)
    bd = tmp_path / "bd"; bd.mkdir()
    monkeypatch.setattr(CR, "BD", bd)
    V = "runX"
    res = {"sequence_binding": {"auroc": 0.7, "items_sha256": "i" * 64},
           "sequence_affinity": {"spearman": 0.5, "items_sha256": "j" * 64}}
    # function_reaudit (2026-10-19): protocol tag, verified inputs, affinity input records
    import function_common as FC
    monkeypatch.setattr(FC, "affinity_inputs_record", lambda *a, **k: {
        "affinity_labels": {"sha256": "a" * 64}, "ligand_maccs": {"sha256": "m" * 64}})
    prot = {**FC.protocol_record([{"file": "tokens.npz", "verified": True}],
                               input_std=FC.InputStandardisation.lookup("t", 2, [FC.standardize_lookup_table(np.eye(2))[1]]).run_record()),
            "affinity_labels": {"sha256": "a" * 64}, "ligand_maccs": {"sha256": "m" * 64}}
    for s in range(10):
        (bd / f"{V}_lig_{CR.TAG}{CR._sd_ec(s)}.json").write_text(json.dumps(
            {"results": res, **misato_splits_record(sp), **BL.binding_labels_record(lab), **prot}))
    CR._BIND_REFUSED.clear()
    v = CR.m_bind(V, False, "sequence", "binding", "auroc")
    assert not getattr(v, "refused", False) and not CR._BIND_REFUSED
    f = bd / f"{V}_lig_{CR.TAG}_seed4.json"                       # built on the old labels
    f.write_text(json.dumps({"results": res, **misato_splits_record(sp), **prot}))
    v = CR.m_bind(V, False, "sequence", "binding", "auroc")
    assert getattr(v, "refused", False)
    assert CR._BIND_REFUSED and CR._BIND_REFUSED[0][1] == f.name
    CR._BIND_REFUSED.clear()
    v = CR.m_bind(V, False, "sequence", "affinity", "spearman")    # affinity: labels unused
    assert not getattr(v, "refused", False) and not CR._BIND_REFUSED


def test_probes_read_v2_labels():
    for rel in ("downstream/tokenize_misato_binding_eval.py",
                "downstream/baselines/conv1d_binding_baselines.py"):
        src = (CODE_ROOT / rel).read_text()
        assert "MISATO_BINDING_SITE as BS_LABELS" in src, rel
        assert "load_binding_site_labels(BS_LABELS)" in src, rel
        assert "binding_labels_record(BS_LABELS)" in src, rel
        assert "binding_label_problem(existing, BS_LABELS)" in src, rel
        assert "np.load(BS_LABELS)" not in src, rel


# ---------------------------------------------------------------- real file
from ensembits.misato_binding_labels import MISATO_BINDING_SITE as REAL  # noqa: E402


@pytest.mark.skipif(not REAL.exists(), reason="misato_binding_site_v2.npz not built")
def test_real_label_file():
    from ensembits.misato_binding_labels import load_binding_site_labels
    from ensembits.data_versions import read_provenance
    lab = load_binding_site_labels()
    side = read_provenance(REAL)
    s = side["summary"]
    assert len(lab) == side["n_labelled"] == s["status"]["ok"]
    assert all(y.sum() > 0 for y in lab.values())                  # no all-negative complex
    assert s["new_labels"]["frac_positives_within_10A"] > 0.9
    rb = B.REAL_BB_DIR
    for pid in sorted(lab)[::100]:
        with np.load(rb / f"{pid}.npz") as z:
            assert z["ca_8"].shape[1] == len(lab[pid])
    rep = json.loads(Path(str(REAL).replace(".npz", ".report.json")).read_text())["complexes"]
    # thrombin 1A4W: S1 pocket Asp189 and Ser195 of the heavy chain (author numbering)
    pos = {(c, n, i): (k, rn) for k, c, n, i, rn in rep["1A4W"]["positives"]}
    with np.load(rb / "1A4W.npz") as z:
        seq = str(z["seq"])
    for key, aa in ((("H", 189, ""), "D"), (("H", 195, ""), "S")):
        assert key in pos and seq[pos[key][0]] == aa
    # 1LAN: MISATO holds one of four copies; labels sit on the copy next to the ligand (A)
    mv = {r["chain"]: r for r in rep["1LAN"]["copies"]["reassigned"]}
    assert mv["A"]["to"] is not None and mv["A"]["from"] is None
    assert {k for _, k, _, _, _ in rep["1LAN"]["positives"]} == {"A"}
    assert rep["1LAN"]["n_pos"] == rep["1LAN"]["n_pocket"]
    # re-audit regression (2026-10-19): swapped / extra copies now carry the pocket next to the
    # MISATO ligand (old joint-free copy rule: AUROC 0.06-0.65 against the ligand distance)
    for pid in ("1XFF 1FCH 2CE9 2XXR 2Y7I 3FAS 3LMK 3MI3 3S9E 4EZX 4IO3 4IO4 4IO5 4IO6 4IO7 "
                "4JE8 4TS1 4UC5 4YB5 5ZW6 6IM4 6NU5 6IOR 1JQ8").split():
        assert rep[pid]["validation"]["auroc_dist_new"] > 0.9, pid
    assert {r["chain"] for r in rep["1XFF"]["copies"]["reassigned"]} == {"A", "B"}
