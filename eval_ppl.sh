#!/bin/bash
set -uo pipefail

# ----------------------------------------------------------------------------
# Perplexity-style benchmarks only: WikiText and LAMBADA. The multiple-choice
# suite lives in eval.sh / eval.py and is not touched here.
#
# Both tasks together take a few minutes on one GPU, so this could equally well
# be a single call:
#     python eval_ppl.py --ckpt "$CKPT" --output_path "$OUT_DIR/ppl.json"
# They are split across two GPUs below only so each task gets its own JSON and
# its own log, matching the layout eval.sh produces.
# ----------------------------------------------------------------------------

CKPT="${CKPT:-out-finewebedu-large-cayspheretnorm_lns/ckpt.pt}"
OUT_DIR="${OUT_DIR:-out-finewebedu-large-cayspheretnorm_lns/eval_ppl}"
DTYPE="${DTYPE:-float32}"          # rolling scores sum hundreds of windows; keep fp32
DEVICE="${DEVICE:-cuda}"
HYPER_CONN_N="${HYPER_CONN_N:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"     # empty -> the model's block_size
INCLUDE_PATH="${INCLUDE_PATH:-}"   # neither task needs a YAML override

TASKS=(
    wikitext            # EleutherAI/wikitext_document_level, wikitext-2-raw-v1, test
    lambada_openai      # EleutherAI/lambada_openai, default config, test
)

EXTRA_ARGS=""
[[ -n "$MAX_SEQ_LEN"   ]] && EXTRA_ARGS+=" --max_seq_len $MAX_SEQ_LEN"
[[ -n "$INCLUDE_PATH"  ]] && EXTRA_ARGS+=" --include_path $INCLUDE_PATH"
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA_ARGS+=" --limit ${SMOKE_LIMIT:-5}"
    echo "[smoke] limiting to ${SMOKE_LIMIT:-5} documents per task"
fi

mkdir -p "$OUT_DIR/logs"
echo "ckpt   = $CKPT"
echo "outdir = $OUT_DIR"
echo "dtype  = $DTYPE  window = ${MAX_SEQ_LEN:-block_size}"
echo "tasks  = ${TASKS[*]}"
echo "----------------------------------------------------------------------"

if [[ ! -f "$CKPT" ]]; then
    echo "[fatal] checkpoint not found: $CKPT"
    exit 1
fi

# --------------------------- dataset warm-up ---------------------------
# Prepare both datasets serially first, for the same reason as in eval.sh:
# concurrent processes hitting the HF cache on a shared filesystem is a good way
# to get lock contention and half-written caches.
#
# Note: lm-eval <= 0.4.9 ships wikitext.yaml with dataset_kwargs.trust_remote_code
# set, and datasets >= 4.0 logs a loud "`trust_remote_code` is not supported
# anymore" error for it. That line is harmless - the HF repo has been converted
# to parquet, so the load succeeds anyway.
echo "[warmup] preparing datasets serially"
python - "$INCLUDE_PATH" "${TASKS[@]}" <<'PY'
import sys
from lm_eval.tasks import TaskManager, get_task_dict
include_path, tasks = (sys.argv[1] or None), sys.argv[2:]
get_task_dict(tasks, TaskManager(include_path=include_path))
print(f"[warmup] {len(tasks)} datasets ready")
PY
if [[ $? -ne 0 ]]; then
    echo "[fatal] dataset warm-up failed; fix this before launching"
    exit 1
fi
echo "----------------------------------------------------------------------"

# ------------------------------ run tasks ------------------------------
declare -A PID_TASK
FAILED=()

gpu=0
for task in "${TASKS[@]}"; do
    echo "[launch] gpu=$gpu  task=$task"
    CUDA_VISIBLE_DEVICES="$gpu" python eval_ppl.py \
        --ckpt "$CKPT" \
        --tasks "$task" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --hyper_conn_n "$HYPER_CONN_N" \
        $EXTRA_ARGS \
        --output_path "$OUT_DIR/${task}.json" \
        > "$OUT_DIR/logs/${task}.log" 2>&1 &
    PID_TASK[$!]=$task
    (( gpu++ ))
done

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

# task -> metrics to print. Perplexities are printed as is (lower is better),
# acc as a percentage. Nothing here is averaged: a perplexity and an accuracy
# do not belong in the same mean.
want = {
    "wikitext":       ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "lambada_openai": ["acc", "perplexity"],
}
as_percent = {"acc"}

print(f"{'task':18s} {'metric':18s} {'value':>10s}")
print("-" * 48)
for task, metrics in want.items():
    path = os.path.join(out_dir, f"{task}.json")
    if not os.path.exists(path):
        print(f"{task:18s} {'-':18s} {'MISSING':>10s}")
        continue
    with open(path) as f:
        d = json.load(f).get(task, {})
    for metric in metrics:
        val = d.get(f"{metric},none", d.get(metric))
        if isinstance(val, (int, float)):
            shown = f"{100*val:.2f}" if metric in as_percent else f"{val:.4f}"
        else:
            shown = "NA"
        print(f"{task:18s} {metric:18s} {shown:>10s}")
print("-" * 48)
print("headline numbers: wikitext word_perplexity, lambada_openai acc")
print("state the rolling-window length (block_size) wherever these are reported")
PY
