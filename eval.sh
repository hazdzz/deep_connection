#!/bin/bash
set -uo pipefail

# ------------------------------- config -------------------------------
# Prefer a checkpoint with the optimizer state stripped out (see strip_ckpt.py).
# The training checkpoint is ~3x larger because of the AdamW moments, and eight
# processes loading it at once is a lot of host RAM and shared-filesystem I/O.
CKPT="${CKPT:-out-finewebedu-large-predyt/ckpt.pt}"
OUT_DIR="${OUT_DIR:-out-finewebedu-large-predyt/eval}"
DTYPE="${DTYPE:-float32}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
NUM_GPUS="${NUM_GPUS:-8}"
DEVICE="${DEVICE:-cuda}"
HYPER_CONN_N="${HYPER_CONN_N:-4}"
INCLUDE_PATH="${INCLUDE_PATH:-./lm_eval_tasks}"   # social_iqa parquet override lives here

LIMIT_ARGS=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    LIMIT_ARGS="--limit ${SMOKE_LIMIT:-50}"
    echo "[smoke] limiting to ${SMOKE_LIMIT:-50} docs per task"
fi

# heaviest task first so the long pole (hellaswag) starts immediately
TASKS=(
    hellaswag
    arc_easy
    boolq
    social_iqa
    arc_challenge
    sciq
    piqa
    winogrande
    openbookqa
)

mkdir -p "$OUT_DIR/logs"
echo "ckpt   = $CKPT"
echo "outdir = $OUT_DIR"
echo "gpus   = $NUM_GPUS"
echo "dtype  = $DTYPE  fewshot = $NUM_FEWSHOT"
echo "tasks  = ${TASKS[*]}"
echo "----------------------------------------------------------------------"

if [[ ! -f "$CKPT" ]]; then
    echo "[fatal] checkpoint not found: $CKPT"
    exit 1
fi
if [[ ! -f "$INCLUDE_PATH/social_iqa.yaml" ]]; then
    echo "[fatal] $INCLUDE_PATH/social_iqa.yaml not found."
    echo "        social_iqa cannot load without it: the upstream HF repo is script-based"
    echo "        and datasets>=4.0 refuses to run dataset scripts."
    exit 1
fi

# --------------------------- dataset warm-up ---------------------------
# Prepare every dataset serially first. Eight processes hitting the HF cache at
# the same time on a shared filesystem is a common source of lock contention and
# half-written caches, and it only ever needs doing once.
echo "[warmup] preparing datasets serially"
python - "$INCLUDE_PATH" "${TASKS[@]}" <<'PY'
import sys
from lm_eval.tasks import TaskManager, get_task_dict
include_path, tasks = sys.argv[1], sys.argv[2:]
get_task_dict(tasks, TaskManager(include_path=include_path))
print(f"[warmup] {len(tasks)} datasets ready")
PY
if [[ $? -ne 0 ]]; then
    echo "[fatal] dataset warm-up failed; fix this before launching the parallel jobs"
    exit 1
fi
echo "----------------------------------------------------------------------"

# ----------------------------- work queue -----------------------------
declare -A GPU_PID     # gpu index -> pid currently running on it
declare -A PID_TASK    # pid -> task name, so failures can be reported by name
FAILED=()

launch () {
    local gpu="$1" task="$2"
    echo "[launch] gpu=$gpu  task=$task"
    CUDA_VISIBLE_DEVICES="$gpu" python eval.py \
        --ckpt "$CKPT" \
        --tasks "$task" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --num_fewshot "$NUM_FEWSHOT" \
        --hyper_conn_n "$HYPER_CONN_N" \
        --include_path "$INCLUDE_PATH" \
        $LIMIT_ARGS \
        --output_path "$OUT_DIR/${task}.json" \
        > "$OUT_DIR/logs/${task}.log" 2>&1 &
    local pid=$!
    GPU_PID[$gpu]=$pid
    PID_TASK[$pid]=$task
}

idx=0
n=${#TASKS[@]}

# fill every GPU once
for (( g=0; g<NUM_GPUS && idx<n; g++ )); do
    launch "$g" "${TASKS[$idx]}"
    (( idx++ ))
done

# feed remaining tasks to GPUs as they free up
while (( idx < n )); do
    wait -n                                  # block until some job ends
    for g in "${!GPU_PID[@]}"; do
        if ! kill -0 "${GPU_PID[$g]}" 2>/dev/null; then
            unset 'GPU_PID[$g]'
            launch "$g" "${TASKS[$idx]}"
            (( idx++ ))
            break
        fi
    done
done

# let the last jobs finish, collecting exit codes by task name
for pid in "${!PID_TASK[@]}"; do
    if ! wait "$pid" 2>/dev/null; then
        FAILED+=("${PID_TASK[$pid]}")
    fi
done

echo "----------------------------------------------------------------------"
if (( ${#FAILED[@]} > 0 )); then
    echo "[warn] these tasks exited non-zero: ${FAILED[*]}"
    for t in "${FAILED[@]}"; do
        echo "       tail -20 $OUT_DIR/logs/${t}.log"
    done
fi
echo "[done] per-task JSON in $OUT_DIR, logs in $OUT_DIR/logs"

# ------------------------------ summary -------------------------------
python - "$OUT_DIR" <<'PY'
import json, os, sys
out_dir = sys.argv[1]
# the metric reported per task (matches the target table)
want = {
    "arc_challenge": "acc_norm",
    "arc_easy":      "acc_norm",
    "boolq":         "acc",
    "hellaswag":     "acc_norm",
    "openbookqa":    "acc_norm",
    "piqa":          "acc_norm",
    "sciq":          "acc",
    "social_iqa":    "acc",
    "winogrande":    "acc",
}
rows, vals = [], []
for task, metric in want.items():
    path = os.path.join(out_dir, f"{task}.json")
    if not os.path.exists(path):
        rows.append((task, metric, "MISSING"))
        continue
    with open(path) as f:
        res = json.load(f)
    d = res.get(task, {})
    val = d.get(f"{metric},none", d.get(metric))
    if isinstance(val, (int, float)):
        rows.append((task, metric, f"{100*val:.2f}"))
        vals.append(val)
    else:
        rows.append((task, metric, "NA"))

print(f"{'task':16s} {'metric':10s} {'value':>8s}")
print("-" * 36)
for task, metric, shown in rows:
    print(f"{task:16s} {metric:10s} {shown:>8s}")
print("-" * 36)
if len(vals) == len(want):
    print(f"{'average':16s} {'':10s} {100*sum(vals)/len(vals):>8.2f}")
else:
    print(f"[warn] {len(want) - len(vals)} of {len(want)} tasks missing; average not shown")
PY