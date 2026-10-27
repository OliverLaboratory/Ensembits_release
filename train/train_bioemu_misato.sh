#!/bin/bash
# =============================================================================
# Train the Ensembits tokenizer reported in the paper (BioEmu-CATH + MISATO).
#
# What: runs train/train.py with the hyperparameters of the reported checkpoint
#   (ckpt/bioemu_cath_misato, originally
#   output/dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO).
#
# Paper: `1_method.tex` Tokenizer + SFTD (Eq. eq:sftd); `appendix.tex`
#   app:RVQVAE and app:model-impl (all values below).
#
# Origin: RECONSTRUCTED. The exact command of the reported run was not saved.
#   This is the training template in `rebuttal/reproducibility.md` §1 combined
#   with the sibling launcher
#   `submission_exp/src/rebuttal_analyses/sametrain_baselines/run_quantA.sh`
#   (same corpus and recipe; that run only changes --vq-update to "loss").
#   Every flag that is recorded in the checkpoint's config.json matches it
#   (codebook sizes, widths, layers, queries, heads, EMA 0.99, commitment 0.5,
#   P = 10, batch 4096, lr 1e-3, wd 1e-5, max 1000 epochs, patience 40,
#   eval every 5, seed 42, rect Hungarian, no standardisation, FAPE, K = 16).
#   --warmup-steps, --noise-std and --kmeans-init are not recorded in
#   config.json; they come from the template.
#
# Inputs: the descriptor cache built by data_prep/build_bioemu_misato_descriptor.py
#   (default CD = output/bioemu_cath_misato_e0/cached_desc, the corpus of the
#   retrain: IUPAC dihedrals, fixed MISATO parser, Kabsch FPS, BioEmu by system,
#   split by domain). The trainer refuses a cache whose layout.json records a
#   different dihedral convention.
# Output: $OUT (best.pt, stats.npz, config.json, history.json, training_curve.png)
#
# Usage: RUN=<name> [OUT=...] [SFTD=off] [LAMBDA=<x>] [EXTRA="..."] [DRYRUN=1] \
#          bash train/train_bioemu_misato.sh
#   RUN     required; flat run name without "/", e.g. ensembits_bioemu_misato_e0.
#           OUT defaults to $REPO/output/$RUN (downstream scripts use output/<variant>).
#   OUT     optional override; refused if it is nested below output/ (output/a/b).
#   SFTD    on (default) | off: off drops --random-p-range and every --consistency-*
#           flag, so the trainer takes its single-branch (non-SFTD) path.
#   LAMBDA  --consistency-mse-weight (default 10); not allowed with SFTD=off.
#   EXTRA   extra trainer flags, APPENDED after the fixed defaults (argparse keeps the
#           last value, so EXTRA can still override a default).
#   DRYRUN  1 = print the final command instead of running it.
#   RESUME  1 = continue the run in $OUT from its last.pt (train.py --resume; same code
#           and settings required; append to the old log, e.g. >> log).
# FIXED(code_fix P10, 2026-10-16): old = OUT=output/bioemu_cath_misato_e0/run (nested,
#   broke the downstream writers that use output/<variant>); new = required RUN name,
#   OUT=$REPO/output/$RUN, nested OUT refused. The trainer refuses an OUT that already
#   holds best.pt / config.json (pass EXTRA="--overwrite" to replace it).
# FIXED(code_fix P23, 2026-10-16): old = EXTRA replaced the fixed defaults and SFTD
#   could not be switched off; new = the defaults are always passed and EXTRA is
#   appended; SFTD=off and LAMBDA=<x> switches.
#
# FIXED(code_fix C3, 2026-10-15): points at the retrain corpus (was the old
#   reviewer_3_special_corpus_MISATO cache, legacy +180° file). The fixed defaults
#   are passed explicitly: --dihedral-convention iupac, --hungarian-cost loss,
#   --rvq-residual standard. To rerun the current checkpoint's recipe instead:
#   CD=$REPO/output/reviewer_3_special_corpus_MISATO/cached_desc
#   DESC=esm3desc_dihedral_K16_P10.npy EXTRA="--dihedral-convention legacy_plus180
#   --hungarian-cost l2 --rvq-residual legacy --codebook-update both".
#   (Since P23 EXTRA is appended after the defaults; argparse keeps the last value,
#   so this override still works.)
# FIXED(code_fix E0, 2026-10-20): corpus built, all 8 _rv runs completed.
# =============================================================================
set -u
CODE=$(cd "$(dirname "$0")/.." && pwd)
REPO=${ENSEMBITS_REPO_ROOT:-$(cd "$CODE/../.." && pwd)}   # data/ and output/ (two levels up)
PY=${PY:-python}                                          # set PY to choose the interpreter

CD=${CD:-$REPO/output/bioemu_cath_misato_e0/cached_desc}      # training corpus (256-D, P = 10)
DESC=${DESC:-esm3desc_dihedral_iupac_K16_P10.npy}             # file name records the convention

# FIXED(code_fix P10, 2026-10-16): was OUT=${OUT:-$REPO/output/bioemu_cath_misato_e0/run}
#   (nested, no run name); now a required flat RUN name and OUT=$REPO/output/$RUN.
RUN=${RUN:-}
[ -n "$RUN" ] || { echo "set RUN=<run name> (flat, e.g. RUN=ensembits_bioemu_misato_e0); OUT = \$REPO/output/\$RUN" >&2; exit 1; }
case "$RUN" in
  */*|.|..) echo "RUN=$RUN must be a single folder name (no '/'): downstream scripts use output/<variant>" >&2; exit 1 ;;
esac
OUT=${OUT:-$REPO/output/$RUN}
_out_abs=$(realpath -m "$OUT"); _outroot_abs=$(realpath -m "$REPO/output")
case "$_out_abs" in
  "$_outroot_abs") echo "OUT=$OUT is the output root itself (use output/<name>)" >&2; exit 1 ;;
  "$_outroot_abs"/*/*) echo "OUT=$OUT is nested below $REPO/output (use a flat output/<name>)" >&2; exit 1 ;;
  "$_outroot_abs"/*) ;;
  *) echo "[warn] OUT=$OUT is outside $REPO/output; downstream scripts look for output/<variant>" >&2 ;;
esac

# FIXED(code_fix P23, 2026-10-16): was a fixed SFTD block with lambda = 0.1 and an EXTRA
#   that replaced the defaults below; now SFTD=on|off, LAMBDA=<x>, and EXTRA is appended.
SFTD=${SFTD:-on}
LAMBDA=${LAMBDA:-}
case "$SFTD" in
  on)  SFTD_FLAGS=(--random-p-range 1,10 --consistency-mse-weight "${LAMBDA:-10}" --consistency-distill-fixed-max) ;;
  off) [ -z "$LAMBDA" ] || { echo "LAMBDA=$LAMBDA has no effect with SFTD=off" >&2; exit 1; }
       SFTD_FLAGS=() ;;
  *)   echo "SFTD=$SFTD must be on or off" >&2; exit 1 ;;
esac
# Fixed defaults of the retrain (code_fix §1, §2, C1, C9); always passed.
DEFAULTS=(--dihedral-convention iupac --hungarian-cost loss --rvq-residual standard --codebook-update full)
EXTRA=${EXTRA:-}
DRYRUN=${DRYRUN:-0}
RESUME=${RESUME:-0}

for f in "$CD/$DESC" "$CD/splits.json" "$CD/layout.json"; do
  if [ ! -f "$f" ]; then
    echo "missing $f (build the corpus with data_prep/build_bioemu_misato_descriptor.py)" >&2
    [ "$DRYRUN" = 1 ] || exit 1
  fi
done

CMD=("$PY" -u "$CODE/train/train.py"
  --desc "$CD/$DESC" --splits "$CD/splits.json"
  --out "$OUT" --device "${DEVICE:-cuda:0}"
  --codebook-sizes 2048,128,128 --hidden-dim 256 --latent-dim 128
  --n-encoder-layers 4 --n-decoder-layers 3 --n-queries 8 --n-heads 4
  --no-standardize
  "${SFTD_FLAGS[@]}"
  --rect-hungarian --recon-loss-type fape
  --kmeans-init --noise-std 0.05 --warmup-steps 1000
  --batch-size 4096 --lr 1e-3 --weight-decay 1e-5
  --max-epochs 1000 --patience 40 --eval-every 5 --seed 42
  --desc-dtype float16 --data-on-gpu
  "${DEFAULTS[@]}")
# EXTRA is word-split on purpose (several flags in one string).
# shellcheck disable=SC2206
CMD+=($EXTRA)
# Training audit 2026-10-16: resume a killed run from $OUT/last.pt.
[ "$RESUME" = 1 ] && CMD+=(--resume)

export PYTHONUNBUFFERED=1
if [ "$DRYRUN" = 1 ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
# Flag -> paper:
#   --codebook-sizes 2048,128,128  N_q = 3, M = [2048, 128, 128]      (1_method.tex Residual quantizer)
#   --hidden-dim/--latent-dim       256 / D = 128                      (app:model-impl)
#   --n-encoder-layers/-queries/-heads 4 / 8 / 4                       (app:model-impl)
#   --n-decoder-layers 3            three hidden layers of 256        (app:model-impl)
#   --random-p-range 1,10           P' ~ U{1..10}                      (Eq. eq:sftd, app:model-impl)
#   --consistency-mse-weight 10     lambda = 10                        (app:model-impl; decisions_20261023 R3)
#   --consistency-distill-fixed-max teacher = all 10 frames, stop-grad (Eq. eq:sftd)
#   --rect-hungarian --recon-loss-type fape   Eq. eq:hungarian / eq:fape (w_dih = 3, clamp 10 A defaults)
#   --kmeans-init --noise-std 0.05 --warmup-steps 1000                 (app:RVQVAE)
#   --no-standardize                "Descriptors are used without standardization" (app:model-impl)
#   --desc-dtype float16 --data-on-gpu: memory placement only (from the sibling launcher)
#   --dihedral-convention iupac     standard phi/psi (code_fix §1)
#   --hungarian-cost loss           matching on the per-pair loss, Eq. eq:hungarian (code_fix §2)
#   --rvq-residual standard         rho_l = rho_{l-1} - sg[q_l], all levels commit (code_fix C1)
#   --codebook-update full          EMA / dead-code revival on the full-ensemble branch only (code_fix C9)
