#!/bin/bash
# Queue after the 40-epoch N=4 Static-OMA training (audit, 2026-09-09): wait
# for swin_train.py (checkpoints/swinsc_ov_u4_oma_e40) to finish, then
# evaluate the head at N = K = 4 on the full SNR grid with the seeds of
# ser_eval.py (scripts/verify_oma_e40.py). Modeled on run_e40_eval.sh.
# Nothing under checkpoints/, data/ (except verify_* outputs) or scripts/ is
# modified. The training process is never signalled (kill -0 only probes).
set -e
cd ~/ViT
export PYTHONPATH=~/ViT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/tr_env/bin/python
LOG=~/ViT/logs
CK=~/ViT/checkpoints
TRAIN_LOG=$LOG/train_ov_u4_oma_e40.log
TRAIN_PID=${TRAIN_PID:-3610369}
E40=$CK/swinsc_ov_u4_oma_e40
step() { echo "$(date -u +%FT%TZ) $1"; }

step "waiting for 'done ->' in $TRAIN_LOG (train pid $TRAIN_PID)"
while ! grep -q 'done ->' "$TRAIN_LOG"; do
  if grep -q -E 'Traceback|Error' "$TRAIN_LOG"; then
    step "ABORT: training log contains Traceback/Error"; grep -n -E 'Traceback|Error' "$TRAIN_LOG" | head; exit 1
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 60                                  # let a final flush land
    grep -q 'done ->' "$TRAIN_LOG" && break
    step "ABORT: train pid $TRAIN_PID is gone without 'done ->'"; tail -n 3 "$TRAIN_LOG"; exit 1
  fi
  sleep 120
done
step "training finished: $(tail -n 1 "$TRAIN_LOG")"
[ -f "$E40/tx.pt" ] && [ -f "$E40/rx.pt" ] && [ -f "$E40/config.json" ] || { step "ABORT: $E40 incomplete"; ls -la "$E40"; exit 1; }
sleep 30                                      # let the trainer release the GPU

step "verify_oma_e40.py (N=K=4, SNR grid, reps5)"
$PY scripts/verify_oma_e40.py --ckpt checkpoints/swinsc_ov_u4_oma_e40 --users 4 --scheme oma_e40 \
    --snrs -5 0 5 10 15 20 \
    --out data/verify_oma_e40.csv --out_perimage data/verify_oma_e40_perimage.csv \
    > $LOG/verify_oma_e40.log 2>&1
grep -q '^saved' $LOG/verify_oma_e40.log || { step "FAILED_verify_oma_e40"; tail -n 20 $LOG/verify_oma_e40.log; exit 1; }
step "$(grep '^saved' $LOG/verify_oma_e40.log)"
step "OMA_E40_EVAL_DONE"
