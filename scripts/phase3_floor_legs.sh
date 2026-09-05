#!/usr/bin/env bash
# Run the serving-boundary floor legs: one server start per leg, seven in all.
#
# Every server is started by scripts/phase3_serve.py with argv differing ONLY in
# --model, so a difference between legs cannot be a difference in how the server
# was brought up. The launch argv is recorded into each leg and the assembly
# step refuses to combine legs whose commands differ anywhere else.
set -Eeuo pipefail
# Say so when the sweep ends early. Two runs of this sweep died at a teardown
# with the last leg written and no line after it, which reads as a finished leg
# rather than as an aborted sweep; `-E` carries the trap into `run_leg`.
trap 'status=$?; echo "!!! sweep aborted: line $LINENO exited $status" >&2' ERR

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
# INFO rather than WARNING: vLLM names the attention backend it resolved in its
# startup log and nowhere else reachable from here, and that string is the
# evidence behind the floor's scope condition -- the backend is the one thing
# MATCHED_ENGINE_FLAGS does not pin, and vLLM picks it by compute capability.
# The level is inert numerically; it changes what is written down, not what runs.
export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
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
  # Kill only a process this sweep actually started. Inside a container
  # `nvidia-smi` reports HOST pids, which collide with unrelated pids in this
  # namespace, so killing the list blind kills whatever happens to hold that
  # number here -- including this script. Measured: an unguarded `kill -9` on
  # this list ended the sweep after the clean leg, silently, with the server
  # already shut down cleanly and no line written after it. Matching on the
  # command line makes the kill specific to the engine that outlived its
  # launcher, which is the case this block exists for.
  # One `if`, and no `[ ... ] && ...` short-circuit: under `set -e` a false
  # test as the last command of a loop body ends the whole sweep, silently,
  # exactly where a leg has just been written and nothing looks wrong. Measured
  # twice here. An `if` whose condition is false leaves status 0.
  local orphans opid
  orphans=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader || true)
  for opid in $orphans; do
    if [ "$opid" != "$$" ] && ps -p "$opid" -o args= 2>/dev/null | grep -qE "phase3_serve|vllm|EngineCore"; then
      kill -9 "$opid" 2>/dev/null || true
    fi
  done
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
