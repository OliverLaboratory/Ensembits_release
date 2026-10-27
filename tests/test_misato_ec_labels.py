"""Tests of the MISATO EC labels v2 (authors' decision D8, 2026-10-18).

What it does
------------
  * overlap rule of `data_prep/build_misato_ec_labels.py` on a toy polyprotein (a
    Gag-Pol-like entry: precursor, protease, RT, integrase, a non-enzymatic capsid):
    a protease-only chain gets the protease EC only, a few residues of a neighbouring
    product do not count, >= 30 residues or >= 50 % of the chain do; features without a
    Contains component and precursor-only residues carry the entry-level EC;
  * complex labels: a complex without a component-resolved chain is unchanged; a mixed
    complex keeps the EC numbers of its non-polyprotein chains; chains absent from the
    MISATO complex do not contribute;
  * wildcard truncation per depth, identical in `tokenize_misato_and_eval.truncate_ec`,
    `baselines/ec_classify_conv1d_misato.truncate_ec` and the builder's copy;
  * `ec_labels`: the loader refuses a file without the v2 sidecar; result checks;
  * homology fallback (2026-10-19): trigger (EC of >= 2 products of a reviewed polyprotein),
    the 85 % identity / 80 % coverage / 20-residue match rule (85 % passes, 80 % fails), chain EC replaced, complexes
    without SIFTS mapping;
  * domain overrides (step 4c, 2026-10-19): every table entry follows the documented rule
    (domain contained >= 90 %, absent domains < 50 %, removed EC numbers belong to absent
    domains); only the listed complexes change; a stale entry stops the build; on the real
    file each override is applied and reverting the entries gives back the sha256 of the
    labels before step 4c (no other complex changed);
  * if the real misato_ec_v2.json exists: HIV-1 protease complexes carry 3.4.23.16 only
    (incl. those relabelled by the homology fallback: 3O99, 4DJR, 5WLO, 2O4N, 1B6J,
    2RKF, 2RKG, 4M8X, 4M8Y).

Paper
-----
not described in the paper (infrastructure only)

How to run
----------
  python -m pytest tests/test_misato_ec_labels.py -q     # from the codebase root (CPU, seconds)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (CODE_ROOT, CODE_ROOT / 'downstream', CODE_ROOT / 'downstream' / 'baselines',
          CODE_ROOT / 'data_prep'):
    sys.path.insert(0, str(p))

import build_misato_ec_labels as B  # noqa: E402


def _feat(t, s, e, d):
    return {'type': t, 'description': d,
            'location': {'start': {'value': s}, 'end': {'value': e}}}


def _comp(name, ec=()):
    rn = {'fullName': {'value': name}}
    if ec:
        rn['ecNumbers'] = [{'value': x} for x in ec]
    return {'recommendedName': rn}


TOY = {  # Gag-Pol-like: capsid 1-200, protease 201-300, RT 301-800, integrase 801-1000
    'primaryAccession': 'TOY1', 'entryType': 'UniProtKB reviewed (Swiss-Prot)',
    'organism': {'scientificName': 'toy virus'}, 'sequence': {'value': 'A' * 1000, 'length': 1000},
    'proteinDescription': {
        'recommendedName': {'fullName': {'value': 'Gag-Pol polyprotein'}},
        'contains': [_comp('Capsid'), _comp('Protease', ['3.4.23.16']),
                     _comp('Reverse transcriptase', ['2.7.7.49', '3.1.26.13']),
                     _comp('Integrase', ['2.7.7.-', '3.1.-.-'])]},
    'features': [_feat('Chain', 1, 1000, 'Gag-Pol polyprotein'), _feat('Chain', 1, 200, 'Capsid'),
                 _feat('Chain', 201, 300, 'Protease'),
                 _feat('Chain', 301, 800, 'Reverse transcriptase'),
                 _feat('Chain', 801, 1000, 'Integrase')]}

ZYMOGEN = {  # EC on the entry name only (prothrombin-like): not relabelled
    'primaryAccession': 'ZYM1', 'sequence': {'value': 'A' * 600, 'length': 600},
    'proteinDescription': {
        'recommendedName': {'fullName': {'value': 'Prothrombin'},
                            'ecNumbers': [{'value': '3.4.21.5'}]},
        'contains': [_comp('Light chain'), _comp('Heavy chain')]},
    'features': [_feat('Chain', 1, 600, 'Prothrombin'), _feat('Chain', 300, 350, 'Light chain'),
                 _feat('Chain', 351, 600, 'Heavy chain')]}

TOXIN = {  # entry-level EC + one EC component (toxin-like); residues 301-900 precursor-only
    'primaryAccession': 'TOX1', 'sequence': {'value': 'A' * 900, 'length': 900},
    'proteinDescription': {
        'recommendedName': {'fullName': {'value': 'Toxin'}, 'ecNumbers': [{'value': '3.4.22.-'}]},
        'contains': [_comp('Glucosyltransferase', ['2.4.1.-'])]},
    'features': [_feat('Chain', 1, 900, 'Toxin'), _feat('Chain', 1, 300, 'Glucosyltransferase')]}

INTEIN = {  # extein pieces without a component carry the entry-level EC
    'primaryAccession': 'INT1', 'sequence': {'value': 'A' * 900, 'length': 900},
    'proteinDescription': {
        'recommendedName': {'fullName': {'value': 'ATP synthase A'},
                            'ecNumbers': [{'value': '7.1.2.2'}]},
        'contains': [_comp('Endonuclease', ['3.1.-.-'])]},
    'features': [_feat('Chain', 1, 200, 'ATP synthase A, 1st part'),
                 _feat('Chain', 201, 600, 'Endonuclease'),
                 _feat('Chain', 601, 900, 'ATP synthase A, 2nd part')]}

INFO = {e['primaryAccession']: B.entry_products(e) for e in (TOY, ZYMOGEN, TOXIN, INTEIN)}
INFO['PLAIN'] = {'resolvable': False, 'polyprotein': False, 'all_ec': ['1.1.1.1']}


def test_entry_classification():
    toy = INFO['TOY1']
    assert toy['polyprotein'] and toy['component_resolved'] and toy['resolvable']
    assert toy['precursor'] == ['Gag-Pol polyprotein']
    assert [p['name'] for p in toy['products']] == ['Capsid', 'Protease', 'Reverse transcriptase',
                                                    'Integrase']
    assert INFO['ZYM1']['polyprotein'] and not INFO['ZYM1']['component_resolved']
    assert not INFO['ZYM1']['resolvable']
    assert set(INFO['TOY1']['all_ec']) == {'3.4.23.16', '2.7.7.49', '3.1.26.13', '2.7.7.-',
                                          '3.1.-.-'}


@pytest.mark.parametrize('rows, expected', [
    ([('TOY1', 201, 300)], {'3.4.23.16'}),                          # protease only
    ([('TOY1', 190, 310)], {'3.4.23.16'}),                          # 11 + 10 flanking residues
    ([('TOY1', 201, 330)], {'3.4.23.16', '2.7.7.49', '3.1.26.13'}),  # 30 residues of RT count
    ([('TOY1', 201, 329)], {'3.4.23.16'}),                          # 29 residues do not
    ([('TOY1', 1, 200)], set()),                                    # capsid: no EC
    # 20 mapped residues: RT 11 (>= 50 % of the chain), integrase 9 (< 30 and < 50 %)
    ([('TOY1', 790, 809)], {'2.7.7.49', '3.1.26.13'}),
])
def test_overlap_rule(rows, expected):
    ec, _ = B.chain_mature_ec(rows, INFO)
    assert ec == expected


def test_short_peptide_inside_one_product():
    ec, _ = B.chain_mature_ec([('TOY1', 820, 831)], INFO)      # 12-residue peptide in integrase
    assert ec == {'2.7.7.-', '3.1.-.-'}


def test_entry_level_ec_rules():
    # toxin: the glucosyltransferase domain gets only its own EC; the rest is precursor-only
    assert B.chain_mature_ec([('TOX1', 1, 300)], INFO)[0] == {'2.4.1.-'}
    assert B.chain_mature_ec([('TOX1', 400, 700)], INFO)[0] == {'3.4.22.-'}
    # intein entry: the extein pieces carry the entry-level EC, the intein its own
    assert B.chain_mature_ec([('INT1', 1, 200), ('INT1', 601, 900)], INFO)[0] == {'7.1.2.2'}
    assert B.chain_mature_ec([('INT1', 201, 600)], INFO)[0] == {'3.1.-.-'}


def test_non_polyprotein_complex_unchanged():
    src = ['3.4.21.5']
    new, det = B.relabel_complex(src, {'H': [('ZYM1', 351, 600)], 'L': [('ZYM1', 300, 350)]},
                                 {'H': {'ZYM1': {'3.4.21.5'}}, 'L': {'ZYM1': {'3.4.21.5'}}},
                                 None, INFO)
    assert new is None and det is None                              # zymogen: kept as is
    new, _ = B.relabel_complex(['1.1.1.1'], {'A': [('PLAIN', 1, 300)]},
                               {'A': {'PLAIN': {'1.1.1.1'}}}, None, INFO)
    assert new is None


def test_polyprotein_complex_relabelled():
    gagpol = ['2.7.7.-', '2.7.7.49', '3.1.-.-', '3.1.26.13', '3.4.23.16']
    sifts = {'A': [('TOY1', 201, 300)], 'B': [('TOY1', 201, 300)]}
    enz = {c: {'TOY1': set(gagpol)} for c in sifts}
    new, det = B.relabel_complex(gagpol, sifts, enz, {'A', 'B'}, INFO)
    assert new == ['3.4.23.16']
    assert det['removed'] == ['2.7.7.-', '2.7.7.49', '3.1.-.-', '3.1.26.13']
    # mixed complex: a non-polyprotein chain keeps its SIFTS EC numbers
    sifts['C'] = [('PLAIN', 1, 300)]
    enz['C'] = {'PLAIN': {'1.1.1.1'}}
    new, _ = B.relabel_complex(gagpol + ['1.1.1.1'], sifts, enz, {'A', 'B', 'C'}, INFO)
    assert new == ['1.1.1.1', '3.4.23.16']
    # a non-polyprotein chain with an EC also given by the polyprotein keeps it
    enz['C'] = {'PLAIN': {'1.1.1.1', '2.7.7.49'}}
    new, _ = B.relabel_complex(gagpol + ['1.1.1.1'], sifts, enz, {'A', 'B', 'C'}, INFO)
    assert new == ['1.1.1.1', '2.7.7.49', '3.4.23.16']


def test_absent_chain_does_not_contribute():
    gagpol = ['2.7.7.49', '3.1.26.13', '3.4.23.16']
    sifts = {'A': [('TOY1', 201, 300)], 'R': [('TOY1', 301, 800)]}
    enz = {c: {'TOY1': set(gagpol)} for c in sifts}
    new, det = B.relabel_complex(gagpol, sifts, enz, {'A'}, INFO)   # RT chain not in MISATO
    assert new == ['3.4.23.16']
    assert det['absent_cr_chain_ec'] == ['2.7.7.49', '3.1.26.13']
    new, _ = B.relabel_complex(gagpol, sifts, enz, {'A', 'R'}, INFO)
    assert new == ['2.7.7.49', '3.1.26.13', '3.4.23.16']


@pytest.mark.parametrize('ec, by_depth', [
    ('3.4.21.4', ['3', '3.4', '3.4.21']),
    ('3.1.-.-', ['3', '3.1', None]),
    ('3.-.-.-', ['3', None, None]),
    ('2.7.7.-', ['2', '2.7', '2.7.7']),
    ('1.1.1.n12', ['1', '1.1', '1.1.1']),
    ('3.4', ['3', '3.4', None]),
    ('-.-.-.-', [None, None, None]),
])
def test_wildcard_truncation_identical(ec, by_depth):
    import tokenize_misato_and_eval as T
    import ec_classify_conv1d_misato as E
    for f in (T.truncate_ec, E.truncate_ec, B.truncate_ec):
        got = [f(ec, d) for d in (1, 2, 3)]
        assert got == by_depth, f.__module__
        assert not any(g and '-' in g for g in got)


def test_build_label_sets_no_dash_classes():
    import tokenize_misato_and_eval as T
    import ec_classify_conv1d_misato as E
    raw = {'A': ['3.1.-.-'], 'B': ['3.4.23.16', '2.7.7.-'], 'C': []}
    for f in (T.build_label_sets, E.build_label_sets):
        assert f(raw, 3) == {'B': {'3.4.23', '2.7.7'}}               # A has no depth-3 class
        assert f(raw, 2) == {'A': {'3.1'}, 'B': {'3.4', '2.7'}}


def test_ec_labels_loader_and_result_checks(tmp_path):
    import ec_labels as L
    old = tmp_path / 'misato_ec.json'
    old.write_text(json.dumps({'1ABC': ['3.4.23.16']}))
    with pytest.raises(SystemExit):
        L.load_ec_labels(old)                                        # no v2 sidecar
    new = tmp_path / 'misato_ec_v2.json'
    new.write_text(json.dumps({'1ABC': ['3.4.23.16']}))
    Path(str(new) + '.provenance.json').write_text(json.dumps({'schema': L.EC_SCHEMA}))
    labels, rec = L.load_ec_labels(new)
    assert labels == {'1ABC': ['3.4.23.16']} and rec['schema'] == L.EC_SCHEMA
    assert L.ec_result_problems({'ec_labels': rec}, new) == []
    assert L.ec_result_problems({'by_depth': {}}, new)               # old result: refused
    assert L.ec_result_problems({'ec_labels': {**rec, 'sha256': '0' * 64}}, new)
    res = tmp_path / 'r.json'
    res.write_text(json.dumps({'by_depth': {}}))
    assert [p for p, _ in L.stale_ec_outputs([res, tmp_path / 'missing.json'], new)] == [res]


def test_real_labels_hiv_protease():
    from ensembits.paths import MISATO_EC_LABELS
    if not MISATO_EC_LABELS.exists():
        pytest.skip('misato_ec_v2.json not built')
    lab = json.loads(MISATO_EC_LABELS.read_text())
    prov = json.loads(Path(str(MISATO_EC_LABELS) + '.provenance.json').read_text())
    assert prov['schema'] == B.SCHEMA
    for pid in ('2I4V', '1A30'):                                    # HIV-1 protease dimers
        if pid in lab:
            assert lab[pid] == ['3.4.23.16']
    src = json.loads(B.SOURCE_COPY.read_text())
    assert set(src) <= set(lab)  # v3 may have new PIDs from SIFTS
    for pid, v in lab.items():                                       # unchanged unless listed
        if pid not in prov.get('changed', {}) and pid in src:
            assert v == sorted(set(src[pid]))


# ---------------------------------------------------------------- homology fallback
def _rand_seq(n, seed):
    import random
    rng = random.Random(seed)
    return ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(n))


def _toy_reviewed():
    """TOY-like reviewed Gag-Pol with a real (random) sequence: capsid 1-200, protease
    201-300, RT 301-800, integrase 801-1000."""
    seq = _rand_seq(1000, 1)
    e = json.loads(json.dumps(TOY))
    e['primaryAccession'] = 'REV1'
    e['entryType'] = 'UniProtKB reviewed (Swiss-Prot)'
    e['sequence'] = {'value': seq, 'length': 1000}
    return e, seq


def _mutate(s, k, seed):
    import random
    rng = random.Random(seed)
    s = list(s)
    for i in rng.sample(range(len(s)), k):
        s[i] = 'W' if s[i] != 'W' else 'Y'
    return ''.join(s)


def test_homology_trigger_and_match():
    rev, seq = _toy_reviewed()
    entries = {'REV1': rev}
    info = {'REV1': B.entry_products(rev)}
    # trigger: EC numbers of >= 2 distinct mature products (protease + RT), not of one
    assert B.multi_product_refs({'3.4.23.16', '3.1.26.13'}, info) == ['REV1']
    assert B.multi_product_refs({'3.4.23.16', '3.4.23.1'}, info) == []
    refs = B.reference_products(info, entries)
    prot = seq[200:300]
    m = B.homology_match(_mutate(prot, 5, 0), refs)               # 95 % identical
    assert m and m['product'] == 'Protease' and m['identity'] >= 0.85 and m['coverage'] == 1.0
    m = B.homology_match(_mutate(prot, 15, 0), refs)              # 85 %: passes (>= 85 %)
    assert m and m['product'] == 'Protease' and m['identity'] == pytest.approx(0.85)
    assert B.homology_match(_mutate(prot, 20, 0), refs) is None   # 80 %: below 85 %
    assert B.homology_match(prot[:15], refs) is None               # too short
    fb = B.homology_fallback([prot], {'REV1': set(info['REV1']['all_ec'])}, refs, entries, info)
    assert fb['source'] == 'homology_fallback' and fb['ec'] == {'3.4.23.16'}
    assert fb['reference_accessions'] == ['REV1']
    # a chain longer than the product: coverage over the product (protease + 20 RT residues)
    fb = B.homology_fallback([seq[200:320]], {'REV1': set(info['REV1']['all_ec'])}, refs,
                             entries, info)
    assert fb['ec'] == {'3.4.23.16'}                               # 20 RT residues < 30


def test_homology_fallback_replaces_chain_ec():
    rev, seq = _toy_reviewed()
    info = {'REV1': B.entry_products(rev), 'FRAG': {'resolvable': False, 'all_ec': ['3.4.23.1']}}
    sifts = {'A': [('FRAG', 1, 99)], 'B': [('FRAG', 1, 99)]}
    sifts_ec = {'2.7.7.49', '3.1.26.13', '3.4.23.1', '3.4.23.16'}
    enz = {c: {'FRAG': set(sifts_ec)} for c in sifts}
    fb = {c: {'sifts_chain': c, 'present': True, 'ec': {'3.4.23.16'},
              'ref_all_ec': set(info['REV1']['all_ec'])} for c in sifts}
    new, det = B.relabel_complex(['3.1.26.13'], sifts, enz, {'A', 'B'}, info, fb)
    assert new == ['3.4.23.16'] and set(det['homology_chains']) == {'A', 'B'}
    # without a fallback match the complex is unchanged
    assert B.relabel_complex(['3.1.26.13'], sifts, enz, {'A', 'B'}, info, {}) == (None, None)
    # no SIFTS mapping at all (1B6J-like): pieces keyed by chain#index
    fb = {'A#0': {'sifts_chain': None, 'present': True, 'ec': {'3.4.23.16'},
                  'ref_all_ec': set(info['REV1']['all_ec'])}}
    new, _ = B.relabel_complex(['2.7.7.49', '3.1.26.13', '3.4.23.16', '1.1.1.1'], {}, {}, set(),
                               info, fb)
    assert new == ['1.1.1.1', '3.4.23.16']


def test_real_labels_homology_fallback():
    from ensembits.paths import MISATO_EC_LABELS
    if not MISATO_EC_LABELS.exists():
        pytest.skip('misato_ec_v2.json not built')
    lab = json.loads(MISATO_EC_LABELS.read_text())
    prov = json.loads(Path(str(MISATO_EC_LABELS) + '.provenance.json').read_text())
    for pid in ('3O99', '4DJR', '5WLO', '2O4N', '1B6J',            # protease-only chains
                '2RKF', '2RKG', '4M8X', '4M8Y'):                # multi-mutant PR, 87.9 %
        if pid in lab:
            assert lab[pid] == ['3.4.23.16'], pid
            assert prov['changed'][pid]['homology_chains']
            for v in prov['changed'][pid]['homology_chains'].values():
                assert v['source'] == 'homology_fallback'
                assert all(m['identity'] >= 0.85 and m['coverage'] >= 0.8 for m in v['matches'])


# ---------------------------------------------------------------- domain overrides (step 4c)
def test_domain_override_table_follows_the_rule():
    """Every entry: the construct covers >= 90 % of its domain(s) and < 50 % of every absent
    domain; new EC = EC of the contained domain(s), a subset of the old EC; every removed EC
    number belongs to an absent domain; one entry per complex."""
    pids = [o['pid'] for o in B.DOMAIN_OVERRIDES]
    assert len(pids) == len(set(pids)) == 30
    for o in B.DOMAIN_OVERRIDES:
        assert B.check_override(o) == [], o['pid']
        assert o['reason'] == B.DOMAIN_OVERRIDE_REASON and o['chains']
    by = {o['pid']: o for o in B.DOMAIN_OVERRIDES}
    assert by['3SU0']['new_ec'] == ['3.4.21.98']                    # NS3 protease domain
    assert by['4OKS']['new_ec'] == ['3.6.1.15', '3.6.4.13']         # NS3 helicase construct
    assert by['5DUS']['new_ec'] == ['3.2.2.-']                      # MERS macrodomain
    assert by['2Q6F']['new_ec'] == ['3.4.22.-']                     # IBV main protease
    assert by['5I1Q']['new_ec'] == [] and by['6CE6']['new_ec'] == []
    # the rule catches a wrong entry
    bad = json.loads(json.dumps(by['1NJS']))
    bad['construct_uniprot_ranges'] = [[434, 1010]]                 # GART + AIRS
    assert 'AIRS domain not absent' in B.check_override(bad)
    bad = json.loads(json.dumps(by['3MI2'])); bad['new_ec'] = ['2.4.2.10', '4.1.1.23']
    assert B.check_override(bad)


def test_apply_domain_overrides_only_listed_complexes():
    ov = B._ov('9XYZ', 'A', 'P11172', [(223, 480)], old=['2.4.2.10', '4.1.1.23'],
               new=['4.1.1.23'], contained=['ompdc'], absent=['oprt'])
    source = {'9XYZ': ['2.4.2.10', '4.1.1.23'], '1ABC': ['1.1.1.1'], '2DEF': ['3.4.23.16']}
    labels = {'9XYZ': ['2.4.2.10', '4.1.1.23'], '1ABC': ['1.1.1.1'], '2DEF': ['3.4.23.16']}
    changed = {}
    rec = B.apply_domain_overrides(labels, changed, source, [ov])
    assert labels == {'9XYZ': ['4.1.1.23'], '1ABC': ['1.1.1.1'], '2DEF': ['3.4.23.16']}
    assert set(changed) == {'9XYZ'} and changed['9XYZ']['removed'] == ['2.4.2.10']
    assert changed['9XYZ']['domain_override']['domain'][0]['coverage'] == pytest.approx(258 / 260,
                                                                                        abs=1e-4)
    assert rec['n_entries'] == 1 and rec['version'] == B.DOMAIN_OVERRIDE_VERSION
    # a stale entry (label of steps 4 / 4b differs from old_ec) stops the build
    with pytest.raises(SystemExit):
        B.apply_domain_overrides({'9XYZ': ['4.1.1.23']}, {}, source, [ov])
    with pytest.raises(SystemExit):                                  # not a labelled complex
        B.apply_domain_overrides({}, {}, source, [ov])
    with pytest.raises(SystemExit):                                  # duplicate entry
        B.apply_domain_overrides(dict(source), {}, source, [ov, ov])


def test_real_labels_domain_overrides():
    from ensembits.paths import MISATO_EC_LABELS
    if not MISATO_EC_LABELS.exists():
        pytest.skip('misato_ec_v2.json not built')
    lab = json.loads(MISATO_EC_LABELS.read_text())
    prov = json.loads(Path(str(MISATO_EC_LABELS) + '.provenance.json').read_text())
    rec = prov['domain_overrides']
    assert rec['version'] == B.DOMAIN_OVERRIDE_VERSION
    assert [e['pid'] for e in rec['entries']] == [o['pid'] for o in B.DOMAIN_OVERRIDES]
    for o in B.DOMAIN_OVERRIDES:                                     # each override applied
        assert lab[o['pid']] == o['new_ec'], o['pid']
        assert prov['changed'][o['pid']]['domain_override']['new_ec'] == o['new_ec']
    # no other complex changed: reverting the entries gives the labels of steps 4 / 4b
    before = dict(lab)
    for o in B.DOMAIN_OVERRIDES:
        before[o['pid']] = o['old_ec']
    assert B.labels_sha256(before) == rec['labels_sha256_before_overrides']
    assert B.labels_sha256(lab) == prov['output_sha256']
    ov = {o['pid'] for o in B.DOMAIN_OVERRIDES}
    assert not [p for p, d in prov['changed'].items() if 'domain_override' in d and p not in ov]
