"""Multi-label test metrics shared by the EC and GO probes (Ensembits and baselines).

What it does
------------
  * `metrics(p, y)`: top-1 hit, mAP (macro AP), muAP (micro AP), F1@0.5 (macro) and
    muF1@0.5 (micro) over the classes with at least one test positive (`nz`), as
    before, plus the protein-centric `fmax` (CAFA) over all classes of the task.
  * `fmax(p, y)`: CAFA protein-centric maximum F-measure. For each threshold t in
    0.00, 0.01, ..., 1.00, a class is predicted when p >= t; precision is averaged over
    the proteins with at least one predicted class, recall over all proteins with at
    least one true class (proteins without a true class are not scored); F(t) =
    2 pr rc / (pr + rc); Fmax = max_t F(t) (thresholds with no predicted protein are
    skipped). Returns the threshold too.
  * `frequency_floor(train_freq, y)`: the label-only predictor that gives every test
    protein the training frequency of each class (ties ranked by column order, i.e. by
    class id); its top-1 hit, muAP, mAP and Fmax.

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{EC Function Prediction} (the five
metrics; "the same five metrics are used for GO"); Fmax and the frequency floor are
new (benchmark_audit D7, Q-metrics) and not yet described in the paper.

Origin
------
New (2026-10-18). `metrics` is the function that existed twice, in
`downstream/tokenize_misato_and_eval.py` and `downstream/baselines/
ec_classify_conv1d_misato.py` (same computation; the baseline copy omitted
`micro_f1@0.5` from its all-NaN return). Both now import it from here.

Used by the reported model: yes (EC and GO rows).

Inputs / outputs
----------------
none (library module)

How to run
----------
Library module; tests in tests/test_go_labels.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

FMAX_THRESHOLDS = np.round(np.arange(101) * 0.01, 2)


def fmax(p, y, thresholds=FMAX_THRESHOLDS) -> tuple[float, float | None]:
    """(Fmax, threshold) — CAFA protein-centric, over every column of `y`."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y) > 0
    has_true = y.sum(1) > 0
    p, y = p[has_true], y[has_true]
    if y.shape[0] == 0:
        return float('nan'), None
    n_true = y.sum(1)
    best, best_t = 0.0, None
    for t in thresholds:
        pred = p >= t
        n_pred = pred.sum(1)
        tp = (pred & y).sum(1)
        covered = n_pred > 0
        if not covered.any():
            continue
        pr = float(np.mean(tp[covered] / n_pred[covered]))
        rc = float(np.mean(tp / n_true))
        if pr + rc <= 0:
            continue
        f = 2 * pr * rc / (pr + rc)
        if f > best:
            best, best_t = f, float(t)
    return float(best), best_t


def metrics(p, y) -> dict:
    """Multi-label test metrics.

    Paper: appendix.tex \\label{app:downstreams} \\paragraph{EC Function
    Prediction} — top-1 hit, mAP (macro AP), muAP (micro AP), F1@0.5 (macro)
    and muF1@0.5 (micro). The same five metrics are used for GO.
    AP and F1 are computed only over classes with at least one positive in the
    test set (`nz`); the paper does not mention this restriction.
    top-1 hit = fraction of proteins whose highest-scored class is a true label.
    Fmax (CAFA, protein-centric) is over all classes.
    """
    nz = (y.sum(0) > 0)
    if nz.sum() == 0:
        return {'macro_ap': float('nan'), 'micro_ap': float('nan'),
                'top1_hit': float('nan'), 'macro_f1@0.5': float('nan'),
                'micro_f1@0.5': float('nan'), 'fmax': float('nan'),
                'fmax_threshold': None, 'n_classes_in_test': 0}
    macro_ap = float(average_precision_score(y[:, nz], p[:, nz], average='macro'))
    micro_ap = float(average_precision_score(y[:, nz], p[:, nz], average='micro'))
    pred_bin = (p > 0.5).astype(np.int8)
    macro_f1 = float(f1_score(y[:, nz], pred_bin[:, nz], average='macro',
                                zero_division=0))
    micro_f1 = float(f1_score(y[:, nz], pred_bin[:, nz], average='micro',
                                zero_division=0))
    pred_top = p.argmax(-1)
    top1 = float(np.mean([y[i, pred_top[i]] > 0 for i in range(y.shape[0])]))
    f, t = fmax(p, y)
    return {'macro_ap': macro_ap, 'micro_ap': micro_ap,
            'top1_hit': top1, 'macro_f1@0.5': macro_f1,
            'micro_f1@0.5': micro_f1, 'fmax': f, 'fmax_threshold': t,
            'n_classes_in_test': int(nz.sum())}


def frequency_floor(train_freq, y) -> dict:
    """Metrics of the predictor that scores every protein with the training frequencies.

    `train_freq`: (C,) training frequency of each class (column order of `y`); computed
    from labels only. Ties in the top-1 pick go to the first column (np.argmax).
    """
    y = np.asarray(y)
    p = np.tile(np.asarray(train_freq, dtype=np.float64), (y.shape[0], 1))
    m = metrics(p, y)
    return {'top1_hit': m['top1_hit'], 'micro_ap': m['micro_ap'], 'macro_ap': m['macro_ap'],
            'fmax': m['fmax'], 'fmax_threshold': m['fmax_threshold'],
            'n_test': int(y.shape[0]),
            'predictor': 'training frequency of each class (same scores for every protein)'}
