"""Downstream probes that consume pre-built token caches.

Each probe is a runnable CLI (``python -m probes.<name>``). Helpers
``_conv1d_head`` and ``_multilabel`` are shared internals.

Probes available:

- ``probe_rmsf`` — per-residue Cα RMSF (mdCATH or MISATO).
- ``probe_ec`` — EC classification (3 depths × 3 splits).
- ``probe_go`` — GO classification (top-50 terms × 3 splits).
- ``probe_binding_site`` — per-residue binding-site labels.
- ``probe_binding_affinity`` — per-ligand affinity regression.
- ``probe_proteingym`` — DMS Spearman + α-blend with ESM2-650M.
- ``anova_token_dynamics`` — η²(s₁ | token) ANOVA.
- ``build_local_s1_cache`` — produce the per-residue s₁ cache the ANOVA reads.
"""
