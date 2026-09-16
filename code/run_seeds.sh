#!/bin/bash
# Seed-variance runs of the N=4 main frame (audit, 2026-09-15).
# Trains PSMA N=4 and Static OMA N=4 with --seed 1 and --seed 2, otherwise the
# EXACT flags of record (run_psma_all.sh for PSMA, run_retrain_all.sh for OMA;
# 20 epochs), then evaluates each pair on the full SNR grid at K = 1..4 with the
# audited evaluators (verify_eval.py for PSMA, verify_oma_e40.py for OMA, which
# reuse ser_eval.py's seeds), so every row is paired with data/ser_eval.csv.
# Order: psma_s1, psma_s2, eval PSMA pair, oma_s1, oma_s2, eval OMA pair.
# Never touches an existing checkpoint: aborts if a target directory exists.
set -e
cd ~/ViT
export PYTHONPATH=~/ViT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/tr_env/bin/python
LOG=~/ViT/logs
CK=~/ViT/checkpoints
step() { echo "$(date -u +%FT%TZ) $1"; }
PSMA="--dataset imagenette --img_size 128 --mask prog --var_load --multi_prefix --all_prefix --load_cond --l_s 8 --beta 1 --channel rayleigh --epochs 20 --amp --bs 24 --users 4 --accum 2"
OMA="--dataset imagenette --img_size 128 --channel rayleigh --epochs 20 --amp --bs 24 --users 4 --mask oma --l_s 8 --beta 1"
SNRS="-5 0 5 10 15 20"

for d in swinsc_ov_u4_psma_s1 swinsc_ov_u4_psma_s2 swinsc_ov_u4_oma_s1 swinsc_ov_u4_oma_s2; do
  [ -e "$CK/$d" ] && { echo "REFUSING: $CK/$d already exists"; exit 1; }
done

train() {  # train <name> <seed> <flags...>
  name=$1; seed=$2; shift 2
  step "train $name (seed $seed)"
  $PY scripts/swin_train.py "$@" --seed $seed --out $CK/$name > $LOG/train_${name#swinsc_}.log 2>&1
  grep -q 'done ->' $LOG/train_${name#swinsc_}.log || { echo "FAILED $name"; exit 1; }
  step "done  $name"
}

train swinsc_ov_u4_psma_s1 1 $PSMA
train swinsc_ov_u4_psma_s2 2 $PSMA
step "eval PSMA seeds"
for s in 1 2; do
  $PY scripts/verify_eval.py --ckpt checkpoints/swinsc_ov_u4_psma_s$s --users 4 --snrs $SNRS \
      --out data/verify_seeds_psma_s$s.csv --out_perimage data/verify_seeds_psma_s${s}_perimage.csv \
      > $LOG/verify_seeds_psma_s$s.log 2>&1
  grep -q '^saved' $LOG/verify_seeds_psma_s$s.log || { echo "FAILED eval psma_s$s"; exit 1; }
done
# merge: scheme label psma -> psma_s1 / psma_s2 (verify_eval.py labels every checkpoint "psma")
{ head -1 data/verify_seeds_psma_s1.csv
  for s in 1 2; do tail -n +2 data/verify_seeds_psma_s$s.csv | sed "s/^psma,/psma_s$s,/"; done
} > data/verify_seeds_psma.csv
step "PSMA_SEEDS_DONE ($(($(wc -l < data/verify_seeds_psma.csv) - 1)) rows)"

train swinsc_ov_u4_oma_s1 1 $OMA
train swinsc_ov_u4_oma_s2 2 $OMA
step "eval OMA seeds"
for s in 1 2; do
  $PY scripts/verify_oma_e40.py --ckpt checkpoints/swinsc_ov_u4_oma_s$s --users 4 --scheme oma_s$s --snrs $SNRS \
      --out data/verify_seeds_oma_s$s.csv --out_perimage data/verify_seeds_oma_s${s}_perimage.csv \
      > $LOG/verify_seeds_oma_s$s.log 2>&1
  grep -q '^saved' $LOG/verify_seeds_oma_s$s.log || { echo "FAILED eval oma_s$s"; exit 1; }
done
{ head -1 data/verify_seeds_oma_s1.csv
  for s in 1 2; do tail -n +2 data/verify_seeds_oma_s$s.csv; done
} > data/verify_seeds_oma.csv
step "OMA_SEEDS_DONE ($(($(wc -l < data/verify_seeds_oma.csv) - 1)) rows)"
step "SEEDS_DONE"
