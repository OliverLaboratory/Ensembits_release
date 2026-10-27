"""Tests of the ProteinGym camera-ready fixes (benchmark_audit D9).

What it does
------------
Synthetic checks (no GPU, no real model, no ProteinGym score of any checkpoint):
  * pg_common: float64 Spearman (a constant predictor scores 0, flagged), z-score,
    rank-sum, CSV rows (n_mut, inconsistency, NaN counts), UniProt ids, atom14 slot
    check, leave-one-UniProt-out alpha;
  * pg_score_baselines: atom14 slots (O = slot 3, CB = slot 4, Gly CB = NaN), 3Di gets
    the real CB, ProToken gets N/CA/C/O and the real sequence, AminoAseed the projected
    codebook;
  * pg_score.evaluate_assay / main on a synthetic cache: float64 per-variant file with
    mutant / n_mut, a NaN ESM2 variant dropped for every key, the `inputs` provenance
    block, the sidecar shas, refusal of an old-schema per-assay JSON on resume;
  * pg_aggregate.run: alignment, -n_mut control, CV alpha distribution, paired tests,
    refusal of a structure mismatch between tags;
  * the fixed assay list (96 assays, 64 Tsuboyama) and pg_load's provenance record.

Paper
-----
`2_results.tex` \\paragraph{Mutation Effect Prediction}, Table
\\label{tab:proteingym_ensemble} (protocol fixed by benchmark_audit D9).

How to run
----------
  python -m pytest tests/test_proteingym_d9.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream"))

import pg_common as C            # noqa: E402
import pg_score                  # noqa: E402
import pg_score_baselines as B   # noqa: E402
import pg_aggregate as AGG       # noqa: E402

AA = "ACDEFHIKLMNPQRSTVWY"       # no G


# ---------------------------------------------------------------- helpers
def make_a14(seq, jitter=None, seed=0):
    """Synthetic atom14 (L, 14, 3): N, CA, C, O, CB with C-O 1.23 A and CA-CB 1.53 A."""
    L = len(seq)
    rng = np.random.default_rng(seed)
    a = np.zeros((L, 14, 3), dtype=np.float32)
    ca = np.stack([np.arange(L) * 3.8, np.zeros(L), np.zeros(L)], 1)
    if jitter is not None:
        ca = ca + jitter
    a[:, 1] = ca
    a[:, 0] = ca + [-1.0, 0.9, 0.0]
    a[:, 2] = ca + [1.0, 0.9, 0.0]
    a[:, 3] = a[:, 2] + [0.0, 1.23, 0.0]
    a[:, 4] = ca + [0.0, 0.0, 1.53]
    for i, c in enumerate(seq):
        if c == "G":
            a[i, 4] = 0.0
    a[:, 5:] = rng.normal(size=(L, 9, 3)) * 0.0
    return a


def mutate(wt, subs):
    s = list(wt)
    for _, p, m in subs:
        s[p] = m
    return "".join(s)


def write_assay(pg_dir, cache_dir, assay, wt, variants, rng):
    """CSV + atom14 cache of one synthetic assay. variants: list of [(wt, pos, mt), ...]."""
    cdir = cache_dir / assay
    cdir.mkdir(parents=True)
    np.save(cdir / f"{C.hash_seq(wt)}.npy", make_a14(wt))
    (cdir / "manifest.json").write_text(json.dumps({"wt": wt, "wt_hash": C.hash_seq(wt)}))
    with open(pg_dir / f"{assay}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mutant", "mutated_sequence", "DMS_score", "ESM2_650M"])
        for subs in variants:
            ms = mutate(wt, subs)
            jit = np.zeros((len(wt), 3))
            for _, p, _m in subs:
                jit[p, 1] = rng.normal() * 2.0
            np.save(cdir / f"{C.hash_seq(ms)}.npy", make_a14(ms, jitter=jit))
            mut = ":".join(f"{a}{p + 1}{m}" for a, p, m in subs)
            w.writerow([mut, ms, rng.normal() - len(subs), rng.normal()])


def fake_tokenize(model, a14, seq=None):
    """Tokens from the CA y coordinate (deterministic), one level."""
    t = (np.floor(np.asarray(a14)[:, 1, 1] * 3).astype(np.int64)) % 8
    return t, t[:, None]


def random_variants(wt, n, rng, max_mut=3):
    out, seen = [], set()
    while len(out) < n:
        k = int(rng.integers(1, max_mut + 1))
        pos = sorted(rng.choice(len(wt), size=k, replace=False).tolist())
        subs = []
        for p in pos:
            m = AA[int(rng.integers(len(AA)))]
            if m == wt[p]:
                m = "W" if wt[p] != "W" else "Y"
            subs.append((wt[p], p, m))
        key = tuple(subs)
        if key not in seen:
            seen.add(key); out.append(subs)
    return out


@pytest.fixture
def synthetic_pg(tmp_path, monkeypatch):
    """6 synthetic assays of 4 'UniProt' entries with a fake cache; pg_score repointed."""
    pg_dir, cache, out = tmp_path / "pg", tmp_path / "cache", tmp_path / "out"
    pg_dir.mkdir(); cache.mkdir(); out.mkdir()
    rng = np.random.default_rng(0)
    assays = ["AAA1_HUMAN_Smith_2020", "AAA1_HUMAN_Tsuboyama_2023_1ABC",
              "BBB2_YEAST_Jones_2019", "CCC3_ECOLI_Tsuboyama_2023_2XYZ",
              "CCC3_ECOLI_Lee_2021_binding", "DDD4_MOUSE_Tsuboyama_2023_3DEF"]
    for i, a in enumerate(assays):
        wt = "".join(AA[int(x)] for x in rng.integers(len(AA), size=30))
        wt = wt[:5] + "G" + wt[6:]
        write_assay(pg_dir, cache, a, wt, random_variants(wt, 40, rng), rng)
    monkeypatch.setattr(pg_score, "PG_DIR", pg_dir)
    monkeypatch.setattr(pg_score, "CACHE_DIR", cache)
    monkeypatch.setattr(pg_score, "OUT_DIR", out)
    lst = tmp_path / "assays.txt"
    lst.write_text("\n".join(assays) + "\n")
    return {"assays": assays, "pg": pg_dir, "cache": cache, "out": out, "list": lst}


# ---------------------------------------------------------------- pg_common
def test_spearman_constant_and_float64():
    assert C.spearman([1, 1, 1], [1, 2, 3]) == (0.0, True)
    assert C.spearman([1, 2, 3], [5, 5, 5]) == (0.0, True)
    # two values equal in float32 but not in float64 are not tied
    x = np.array([1.0, 1.0 + 1e-9, 2.0, 3.0])
    assert np.float32(x[0]) == np.float32(x[1])
    rho, const = C.spearman(x, [1.0, 2.0, 3.0, 4.0])
    assert not const and rho == pytest.approx(1.0)


def test_zscore_ranksum():
    assert np.all(C.zscore([2.0, 2.0, 2.0]) == 0)
    z = C.zscore([1.0, 2.0, 3.0])
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    y = np.arange(10.0)
    assert C.ranksum_spearman(y, y, y) == pytest.approx(1.0)
    # alpha = 0 is ESM2 alone, alpha = 1 the method alone
    rng = np.random.default_rng(1)
    p, e = rng.normal(size=50), rng.normal(size=50)
    b = C.blend_spearman(p, e, y=np.r_[y, y, y, y, y], alphas=(0.0, 1.0))
    assert b[0] == pytest.approx(C.spearman(e, np.r_[y, y, y, y, y])[0])
    assert b[1] == pytest.approx(C.spearman(p, np.r_[y, y, y, y, y])[0])


def test_uniprot_and_tsuboyama():
    assert C.uniprot_of("AMFR_HUMAN_Tsuboyama_2023_4G3O") == "AMFR_HUMAN"
    assert C.uniprot_of("A0A140D2T1_ZIKV_Sourisseau_2019") == "A0A140D2T1_ZIKV"
    assert C.is_tsuboyama("AMFR_HUMAN_Tsuboyama_2023_4G3O")
    assert not C.is_tsuboyama("GLPA_HUMAN_Elazar_2016")


def test_load_assay_rows(tmp_path):
    wt = "ACDEFG"
    p = tmp_path / "x.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mutant", "mutated_sequence", "DMS_score", "ESM2_650M"])
        w.writerow(["A1W", "WCDEFG", "1.5", "0.1"])
        w.writerow(["A1W:D3Y", "WCYEFG", "0.5", "nan"])
        w.writerow(["C2W", "ACDEFW", "0.2", "0.3"])          # string disagrees with sequence
        w.writerow(["A1W", "WCDEFG", "", "0.4"])            # repeated id, NaN DMS
    rows, st = C.load_assay_rows(p, wt)
    assert [r["n_mut"] for r in rows] == [1, 2, 1, 1]
    assert st["n_inconsistent"] == 1 and st["n_dup_mutant"] == 1
    assert st["n_nan_esm2"] == 1 and st["n_nan_dms"] == 1
    assert isinstance(rows[0]["DMS"], float)
    assert C.wt_from_row("A1W:D3Y", "WCYEFG") == "ACDEFG"


def test_atom14_slot_check():
    seq = "ACDGKL"
    a = make_a14(seq)
    C.check_atom14_slots(a, seq)
    swapped = a.copy(); swapped[:, [3, 4]] = a[:, [4, 3]]
    with pytest.raises(ValueError):
        C.check_atom14_slots(swapped, seq)


def test_louo_alpha_cv():
    grid = (0.0, 0.5, 1.0)
    rho = {"AAA1_HUMAN_x_2020": [0.1, 0.5, 0.2],   # group A peaks at 0.5
           "AAA1_HUMAN_y_2021": [0.1, 0.4, 0.2],
           "BBB2_YEAST_z_2019": [0.1, 0.2, 0.6],   # group B peaks at 1.0
           "CCC3_ECOLI_w_2018": [0.3, 0.3, 0.3]}   # flat
    cv = C.louo_alpha_cv(rho, grid)
    # held out A: best on B+C = argmax([0.4, 0.5, 0.9]/2) -> 1.0
    assert cv["alpha_by_group"]["AAA1_HUMAN"] == 1.0
    assert cv["rho_cv"]["AAA1_HUMAN_x_2020"] == 0.2
    # held out B: A+C means [0.167, 0.4, 0.233] -> 0.5
    assert cv["alpha_by_group"]["BBB2_YEAST"] == 0.5
    assert cv["rho_cv"]["BBB2_YEAST_z_2019"] == 0.2
    assert set(cv["alpha_by_assay"]) == set(rho)
    # ties -> smallest alpha
    tie = C.louo_alpha_cv({"A_B_c_1": [0.2, 0.2, 0.2], "C_D_e_2": [0.2, 0.2, 0.2]}, grid)
    assert set(tie["alpha_by_group"].values()) == {0.0}


# ---------------------------------------------------------------- baselines
def test_atom14_parts_slots():
    seq = "AGC"
    a = make_a14(seq)
    n, ca, c, o, cb = B.atom14_parts(a, seq)
    assert np.allclose(o, a[:, 3]) and np.allclose(n, a[:, 0]) and np.allclose(c, a[:, 2])
    assert np.isnan(cb[1]).all() and np.allclose(cb[[0, 2]], a[[0, 2], 4])


def test_3di_reads_real_cb(monkeypatch):
    pytest.importorskip("mini3di")
    import ensembits.baselines as eb
    seen = {}
    def fake(ca, n=None, c=None, cb=None):
        seen.update(ca=ca, n=n, c=c, cb=cb)
        return np.zeros(len(ca), dtype=np.int64)
    monkeypatch.setattr(eb, "three_di_tokens", fake)
    seq = "AGCDE"
    a = make_a14(seq)
    B.tokenize_3di(None, a, seq)
    ng = [0, 2, 3, 4]
    assert np.allclose(seen["cb"][ng], a[ng, 4])             # real CB, not O
    assert not np.allclose(seen["cb"][ng], a[ng, 3])
    assert np.isfinite(seen["cb"][1]).all() and np.linalg.norm(seen["cb"][1] - a[1, 1]) > 1.0
    assert np.allclose(seen["n"], a[:, 0]) and np.allclose(seen["c"], a[:, 2])


def test_protoken_reads_o_and_sequence():
    class Fake:
        def tokenize(self, ca, seq, bb=None):
            self.seq, self.bb = seq, bb
            return np.arange(len(ca))
    tok, seq = Fake(), "ACGDE"
    a = make_a14(seq)
    t, t_all = B.tokenize_protoken(tok, a, seq)
    assert tok.seq == seq and t_all.shape == (5, 1)
    assert np.allclose(tok.bb[:, 3], a[:, 3]) and np.allclose(tok.bb[:, 1], a[:, 1])


def test_aminoaseed_uses_projected_codebook(monkeypatch):
    import ensembits.baselines as eb
    class FakeAA:
        def __init__(self, weights_path, repo_path, device):
            self.codebook = np.full((4, 3), 7.0, dtype=np.float32)   # the projected one
            self._raw = np.zeros((4, 3), dtype=np.float32)
    monkeypatch.setattr(eb, "AminoAseed", FakeAA)
    load, tokfn, weights = B.make_loader("aminoaseed")
    tok, cbs = load("cpu")
    assert cbs[0] is tok.codebook and tokfn is B.tokenize_nca_c


# ---------------------------------------------------------------- pg_score
def test_evaluate_assay_fixes(synthetic_pg, monkeypatch):
    monkeypatch.setattr(pg_score, "tokenize_ca", fake_tokenize)
    a = synthetic_pg["assays"][0]
    # one variant with a NaN ESM2 value
    p = synthetic_pg["pg"] / f"{a}.csv"
    rows = list(csv.reader(open(p)))
    rows[3][3] = "nan"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    cb = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    res = pg_score.evaluate_assay(a, None, [cb], out_tag="t")
    assert res["n_variants"] == 39 and res["n_dropped"]["nonfinite"] == 1
    for k in ("spearman_zmix_a_a03", "spearman_ranksum_a", "spearman_ranksum_d"):
        assert np.isfinite(res[k])
    assert res["pg_version"] == pg_score.PG_VERSION
    inp = res["inputs"]
    assert inp["n_structures"] == 41 and len(inp["structure_sha256"]) == 64
    npz = synthetic_pg["out"] / "per_variant_pg_t" / f"{a}.npz"
    assert C.file_sha256(npz) == res["per_variant_sha256"]
    z = np.load(npz)
    assert z["fitness"].dtype == np.float64 and z["score_a"].dtype == np.float64
    assert len(z["mutant"]) == 39 and z["n_mut"].min() >= 1
    # the constant hamming score of a one-token codebook is 0, flagged
    res2 = pg_score.evaluate_assay(a, None, [np.zeros((8, 4), np.float32)], out_tag="t0")
    assert res2["spearman_score_a_allL0"] == 0.0 and "score_a" in res2["constant_scores"]


def test_main_baseline_hook_and_sidecar(synthetic_pg, monkeypatch, capsys):
    monkeypatch.setattr(pg_score, "tokenize_ca", fake_tokenize)
    cb = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    prov = {"variant": "baseline:fake", "best_pt_sha256": "x", "code_sha256": None}
    base = {"load": lambda device: (None, [cb]), "provenance": prov}
    argv = ["--assay-list", str(synthetic_pg["list"]), "--gpus", "0", "--out-tag", "fk"]
    assert pg_score.main(argv, baseline=base) == 0
    out = capsys.readouterr().out
    assert "a=+" not in out and "mean(" not in out          # blind console
    side = json.loads((synthetic_pg["out"] / "per_assay_pg_fk.assays_scored.json").read_text())
    assert side["n_assays_scored"] == 6 and side["pg_version"] == pg_score.PG_VERSION
    assert side["assay_list_sha256"] == C.assay_list_sha256(synthetic_pg["assays"])
    assert len(side["structure_set_sha256"]) == 64
    # an old-schema per-assay JSON is not resumed
    f = synthetic_pg["out"] / "per_assay_pg_fk" / f"{synthetic_pg['assays'][0]}.json"
    rec = json.loads(f.read_text()); rec.pop("pg_version"); f.write_text(json.dumps(rec))
    assert pg_score.main(argv, baseline=base) == 1


# ---------------------------------------------------------------- pg_aggregate
def _run_two_tags(synthetic_pg, monkeypatch):
    monkeypatch.setattr(pg_score, "tokenize_ca", fake_tokenize)
    rng = np.random.default_rng(3)
    for tag in ("m1", "m2"):
        cb = rng.normal(size=(8, 4)).astype(np.float32)
        base = {"load": lambda device, cb=cb: (None, [cb]),
                "provenance": {"variant": f"baseline:{tag}", "best_pt_sha256": tag,
                               "code_sha256": None}}
        assert pg_score.main(["--assay-list", str(synthetic_pg["list"]), "--gpus", "0",
                              "--out-tag", tag], baseline=base) == 0


def test_aggregate_cv_controls_tests(synthetic_pg, monkeypatch):
    _run_two_tags(synthetic_pg, monkeypatch)
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    res = AGG.run(["m1", "m2"], {"m1": "M1"}, synthetic_pg["assays"], grid, 0.5,
                  primary=["m1"], out_dir=synthetic_pg["out"])
    S = res["summary"]
    assert set(S) == {"M1", "m2", AGG.NMUT, AGG.ESM2}
    assert S["M1"]["alone"]["n"] == 6 and S["M1"]["alone"]["n_uniprot"] == 4
    assert S["M1"]["alone"]["n_tsuboyama"] == 3
    # the -n_mut control is blended under the same CV rule
    assert set(res["alpha"]) == {"M1", "m2", AGG.NMUT}
    assert set(res["alpha"]["M1"]["by_group"]) == {"AAA1_HUMAN", "BBB2_YEAST", "CCC3_ECOLI",
                                                   "DDD4_MOUSE"}
    assert all(a in grid for a in res["alpha"]["M1"]["by_group"].values())
    # -n_mut alone equals the Spearman of -n_mut recomputed from the CSV
    a0 = synthetic_pg["assays"][0]
    z = np.load(synthetic_pg["out"] / "per_variant_pg_m1" / f"{a0}.npz")
    assert res["per_assay"][a0][AGG.NMUT]["alone"] == pytest.approx(
        C.spearman(-z["n_mut"], z["fitness"])[0])
    names = [t["comparison"] for t in res["tests"]]
    assert "M1 vs m2 (+ESM2 CV)" in names and "M1 vs -n_mut (alone)" in names
    assert all(0 <= t["p_wilcoxon_holm"] <= 1 for t in res["tests"])
    md = AGG.to_markdown(res)
    assert "-n_mut" in md and "CV α" in md and "Pre-specified test family" in md
    json.dumps(res)                                    # serialisable


def test_aggregate_refuses_structure_mismatch(synthetic_pg, monkeypatch):
    _run_two_tags(synthetic_pg, monkeypatch)
    a0 = synthetic_pg["assays"][0]
    f = synthetic_pg["out"] / "per_assay_pg_m2" / f"{a0}.json"
    rec = json.loads(f.read_text()); rec["inputs"]["structure_sha256"] = "0" * 64
    f.write_text(json.dumps(rec))
    with pytest.raises(AGG.ProvenanceError):
        AGG.run(["m1", "m2"], {}, synthetic_pg["assays"], (0.0, 1.0), None,
                out_dir=synthetic_pg["out"], family=["tokenizers_vs_esm2"])
    # a per-variant file that does not match its JSON is refused too
    np.savez(synthetic_pg["out"] / "per_variant_pg_m1" / f"{a0}.npz", x=np.zeros(1))
    with pytest.raises(AGG.ProvenanceError):
        AGG.run(["m1"], {}, synthetic_pg["assays"], (0.0, 1.0), None,
                out_dir=synthetic_pg["out"], family=["tokenizers_vs_esm2"])


# ---------------------------------------------------------------- list + pg_load
def test_fixed_assay_list():
    assays = C.read_assay_list()
    assert len(assays) == 96 and assays == sorted(assays)
    assert sum(C.is_tsuboyama(a) for a in assays) == 64
    assert len({C.uniprot_of(a) for a in assays}) == 88


def test_pg_load_records_provenance(synthetic_pg, monkeypatch):
    _run_two_tags(synthetic_pg, monkeypatch)
    import _compile_rebuttal_results as CR
    monkeypatch.setattr(CR, "PG", synthetic_pg["out"])
    d = CR.pg_load("m1")
    prov = CR.PG_PROVENANCE["m1"]
    assert len(d) == 6 and prov["n_without_inputs_block"] == 0
    assert prov["pg_version"] == pg_score.PG_VERSION
    assert prov["assay_list_sha256"] == C.assay_list_sha256(synthetic_pg["assays"])
    assert len(prov["structure_set_sha256"]) == 64


# ---------------------------------------------------------------- final_fix_list A5
def test_aggregate_a5_lambda_blind_gate(synthetic_pg, monkeypatch):
    """An Ensembits tag (sidecar checkpoint not 'baseline:') is refused without --primary,
    before any per-assay value is loaded."""
    _run_two_tags(synthetic_pg, monkeypatch)
    side = synthetic_pg["out"] / "per_assay_pg_m2.assays_scored.json"
    d = json.loads(side.read_text()); d["checkpoint"]["variant"] = "ensembits_bm_lamX"
    side.write_text(json.dumps(d))
    orig, calls = AGG.load_tag, []
    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("per-assay data loaded")
    monkeypatch.setattr(AGG, "load_tag", boom)
    with pytest.raises(AGG.BlindError):
        AGG.run(["m1", "m2"], {}, synthetic_pg["assays"], (0.0, 1.0), None,
                out_dir=synthetic_pg["out"], family=["tokenizers_vs_esm2"])
    with pytest.raises(AGG.BlindError):                   # primary not among the tags
        AGG.run(["m1"], {}, synthetic_pg["assays"], (0.0, 1.0), None, primary=["m2"],
                out_dir=synthetic_pg["out"])
    rc = AGG.main(["--method", "m1", "--method", "m2", "--name", "z", "--family",
                   "tokenizers_vs_esm2", "--assay-list", str(synthetic_pg["list"]),
                   "--out-dir", str(synthetic_pg["out"])])
    assert rc == 1 and not calls
    assert not (synthetic_pg["out"] / "pg_d9_z.json").exists()
    monkeypatch.setattr(AGG, "load_tag", orig)
    res = AGG.run(["m1", "m2"], {}, synthetic_pg["assays"], (0.0, 0.5, 1.0), None,
                  primary=["m2"], out_dir=synthetic_pg["out"], n_boot=200)
    assert res["meta"]["ensembits_tags"] == ["m2"] and res["meta"]["primary"] == ["m2"]


def test_aggregate_a5_family_ci_multi_and_appendix(synthetic_pg, monkeypatch):
    from significance_tests import holm
    _run_two_tags(synthetic_pg, monkeypatch)
    assays, grid = synthetic_pg["assays"], (0.0, 0.25, 0.5, 0.75, 1.0)
    with pytest.raises(ValueError):                       # default family needs a primary
        AGG.run(["m1", "m2"], {}, assays, grid, None, out_dir=synthetic_pg["out"])
    with pytest.raises(ValueError):
        AGG.run(["m1", "m2"], {}, assays, grid, None, out_dir=synthetic_pg["out"],
                family=["no_such_item"])
    res = AGG.run(["m1", "m2"], {"m1": "P"}, assays, grid, 0.5, primary=["m1"],
                  out_dir=synthetic_pg["out"], n_boot=300, boot_seed=1)
    ft = res["family_tests"]
    # Q3: tokenizers_vs_nmut added to DEFAULT_FAMILY
    assert [t["comparison"] for t in ft] == ["P + ESM2 (CV alpha) vs ESM2 alone",
                                             "P + ESM2 vs m2 + ESM2 (CV alpha)",
                                             "P + ESM2 vs -n_mut + ESM2 (CV alpha)",
                                             "m2 + ESM2 vs -n_mut + ESM2 (CV alpha)"]
    assert [t["p_holm"] for t in ft] == pytest.approx(holm([t["p_wilcoxon_uniprot"] for t in ft]))
    for t in ft:
        assert t["claim"] == (t["delta"] >= AGG.CLAIM_MIN_DELTA and t["p_holm"] < AGG.CLAIM_ALPHA)
        lo, hi = t["ci95_delta"]
        assert lo <= hi
    # the exploratory tests are still in the JSON, with their own Holm
    assert res["tests"] and all("p_wilcoxon_holm" in t for t in res["tests"])
    d = res["summary"]["P"]["delta_vs_esm2"]
    assert d["mean"] == pytest.approx(np.mean([res["summary"]["P"]["_per_assay"]["blend_cv"][a]
                                               - res["summary"][AGG.ESM2]["_per_assay"]["alone"][a]
                                               for a in assays]))
    # multi-mutant subset: assays with at least one multi-mutant variant
    n_multi = sum(res["per_assay"][a]["_n_multi"] > 0 for a in assays)
    assert res["meta"]["n_multi_mutant_assays"] == n_multi
    assert res["multi_mutant"][AGG.NMUT]["alone"]["n"] == n_multi
    # -n_mut is constant inside every n_mut stratum -> within-n_mut Spearman 0
    assert all(v in (None, 0.0) for v in res["summary"][AGG.NMUT]["_per_assay"]["within_nmut"].values())
    md = AGG.to_markdown(res)
    for s in ("**P** (primary)", "## Pre-specified test family", "## Multi-mutant assays",
              "## Appendix: rank-sum blend and within-n_mut Spearman", "Δ vs ESM2 [95% CI]"):
        assert s in md
    json.dumps(res)


def test_cluster_bootstrap_and_within_nmut():
    vals = {"AAA1_HUMAN_x_1": 0.2, "AAA1_HUMAN_y_2": 0.2, "BBB2_YEAST_z_3": 0.2}
    assert AGG.cluster_bootstrap_ci(vals, n_boot=100) == pytest.approx([0.2, 0.2])
    vals2 = {"AAA1_HUMAN_x_1": 0.0, "AAA1_HUMAN_y_2": 1.0, "BBB2_YEAST_z_3": 0.5,
             "CCC3_ECOLI_w_4": 0.1}
    a = AGG.cluster_bootstrap_ci(vals2, n_boot=500, seed=3)
    assert a == AGG.cluster_bootstrap_ci(vals2, n_boot=500, seed=3) and a[0] < a[1]
    assert AGG.cluster_bootstrap_ci({}, n_boot=10) is None
    rng = np.random.default_rng(0)
    n_mut = np.repeat([1, 2, 3], 20)
    x = rng.normal(size=60)
    y = 10 * n_mut + x                                     # mutation count dominates y
    assert AGG.within_nmut_spearman(x, y, n_mut) == pytest.approx(1.0)
    assert AGG.within_nmut_spearman(-n_mut.astype(float), y, n_mut) == 0.0
    assert AGG.within_nmut_spearman(x[:5], y[:5], n_mut[:5]) is None   # no stratum >= 10
