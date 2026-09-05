#!/usr/bin/env bash
# Run the serving-boundary floor legs: one server start per leg, seven in all.
#
# Every server is started by scripts/phase3_serve.py with argv differing ONLY in
# --model, so a difference between legs cannot be a difference in how the server
# was brought up. The launch argv is recorded into each leg and the assembly
# step refuses to combine legs whose commands differ anywhere else.
set -euo pipefail

REPO=/workspace/weight-sync-bench
PY=$REPO/.venv-phase3/bin/python
CLEAN=/workspace/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
LEGS=${LEGS:-/opt/wsb/legs}
PORT=${PORT:-8000}
REPS=${REPS:-20}
LOGDIR=${LOGDIR:-/opt/wsb/serverlogs}

# FlashInfer shells out to `ninja` during warm-up and vLLM's engine-core
# subprocess does not inherit the venv's bin dir, so the engine dies with
# FileNotFoundError: 'ninja' unless it is on PATH explicitly. Measured, not
# precautionary: this is what the first engine start failed on.
export PATH=/opt/wsb/venv-phase3/bin:$PATH
export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$LEGS" "$LOGDIR"

run_leg() {
  local case=$1 layer=$2 model=$3 label=$4
  local log="$LOGDIR/$label.log"
  # Resume: a leg already on disk is not re-measured. Server starts are the
  # scarce thing here, and a re-run after a driver fault should cost only the
  # legs that are actually missing.
  if [ -f "$LEGS/$label.pt" ]; then echo "=== $label: already measured, skipping"; return 0; fi
  echo "=== $label: starting server on $model"
  $PY "$REPO/scripts/phase3_serve.py" --model "$model" --port "$PORT" \
      --broadcast-type filesystem >"$log" 2>&1 &
  local pid=$!

  local waited=0
  until curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 $pid 2>/dev/null; then
      echo "!!! $label: server exited during startup; tail of $log:"; tail -25 "$log"; return 1
    fi
    sleep 5; waited=$((waited+5))
    if [ $waited -ge 600 ]; then echo "!!! $label: server not ready after ${waited}s"; tail -25 "$log"; kill $pid; return 1; fi
  done
  echo "--- $label: ready after ${waited}s"

  local launch
  launch=$(python3 -c "import json,sys; print(json.dumps({'command':['$PY','$REPO/scripts/phase3_serve.py','--model','$model','--port','$PORT','--broadcast-type','filesystem'],'cwd':'$REPO'}))")

  local extra=()
  if [ "$case" != "clean" ]; then extra=(--source-dir "$CLEAN"); fi

  set +e
  $PY -m weight_sync_bench.phase3.serving_floor --served-leg \
      --leg-dir "$LEGS" --case "$case" ${layer:+--layer "$layer"} \
      --base-url "http://localhost:$PORT" --repetitions "$REPS" \
      --prompts 4 --seq-len 32 --seed 0 --broadcast-type filesystem \
      --launch-json "$launch" "${extra[@]}"
  local rc=$?
  set -e

  # Tear the engine down and WAIT for the GPU to actually come back. vLLM's
  # engine core is a separate process that outlives a kill of the launcher, and
  # an engine still holding memory makes the next leg fail startup with
  # "Free memory on device cuda:0 ... is less than desired GPU memory
  # utilization" -- which reads as a configuration error rather than as the
  # previous leg not having exited. Measured: that is how the first restart of
  # this sweep died.
  kill $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  local orphans
  orphans=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader || true)
  [ -n "$orphans" ] && kill -9 $orphans 2>/dev/null || true
  # `if`, not `[ ... ] && break`: under `set -e` that list returns non-zero on
  # the iterations where the GPU is NOT yet free, which exits the whole sweep
  # silently. Measured -- it ended this sweep after one break leg.
  local freed=0
  while [ $freed -lt 120 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -lt 1000 ]; then break; fi
    sleep 5; freed=$((freed+5))
  done
  echo "--- $label: gpu released after ${freed}s"
  return $rc
}

run_leg clean "" "$CLEAN" clean
for layer in 0 13; do
  for case in case1_qkv_head_permute case2_oproj_col_permute case3_norm_permute; do
    run_leg "$case" "$layer" "/opt/wsb/ckpt/$case-layer$layer" "$case@layer$layer"
  done
done
echo "=== all legs done"
