#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/eval_final_paper_missing.yaml"
OUTPUT_DIR="results_final_paper_missing"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

if ! python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
then
  echo "CUDA check failed; refusing to start a CPU evaluation." >&2
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  hf auth whoami >/dev/null || {
    echo "Hugging Face authentication is required for the gated Llama bases." >&2
    echo "Run: hf auth login" >&2
    exit 1
  }
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli whoami >/dev/null || {
    echo "Hugging Face authentication is required for the gated Llama bases." >&2
    echo "Run: huggingface-cli login" >&2
    exit 1
  }
fi

git -C .. rev-parse HEAD | tee "$OUTPUT_DIR/git-revision.txt"
python --version 2>&1 | tee "$OUTPUT_DIR/python-version.txt"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
python -m pip freeze > "$OUTPUT_DIR/pip-freeze.txt"
cp "$CONFIG" "$OUTPUT_DIR/eval_final_paper_missing.yaml"

run_stage() {
  local stage="$1"
  shift
  echo
  echo "================================================================"
  echo "Starting stage: $stage"
  echo "================================================================"
  python evaluate.py \
    --config "$CONFIG" \
    --skip-existing \
    --no-summary \
    "$@" \
    2>&1 | tee -a "$LOG_DIR/${stage}.log"
}

# InstructIR has not yet been rerun with the immutable final checkpoints.
# Both Promptriever sizes are included so the final table can report 7B and 8B.
run_stage "01_instructir_missing" \
  --models \
    qwen3-embedding-4b \
    promptriever-7b \
    promptriever-8b \
    ru-only-step-180 \
    ru-en-step-300 \
    ru-promptriever-1.7b-step-320 \
    english-only-4b-step-400 \
  --datasets instructir

# The July corrected FollowIR run already contains Promptriever-7B/8B.  Only
# Qwen3-Embedding, 1.7B, and the unified Ru-only/Ru+En revisions are missing.
run_stage "02_followir_missing" \
  --models \
    qwen3-embedding-4b \
    ru-only-step-180 \
    ru-en-step-300 \
    ru-promptriever-1.7b-step-320 \
  --datasets followir_eng

# Missing mFollowIR rows only.  Ru+En step 300 and Promptriever-8B were
# already evaluated with the corrected official protocol on 2026-07-15.
run_stage "03_mfollowir_missing" \
  --models \
    ru-only-step-180 \
    ru-promptriever-1.7b-step-320 \
    pretrained-4b-step-5750 \
    english-only-4b-step-400 \
    attention-only-4b \
    ru-promptriever-0.6b-final \
    qwen3-embedding-4b \
    multilingual-e5-large \
    bge-m3 \
    bm25 \
  --datasets mfollowir_ru

# Recompute only model rows produced by our pipeline.  The published general
# retrieval baselines are unaffected by the mFollowIR/FollowIR qrels fixes.
run_stage "04_rubq_missing" \
  --models \
    qwen3-embedding-4b \
    pretrained-4b-step-5750 \
    ru-only-step-180 \
    ru-en-step-300 \
    ru-promptriever-1.7b-step-320 \
    attention-only-4b \
    english-only-4b-step-400 \
    ru-promptriever-0.6b-final \
  --datasets rumteb_retrieval

run_stage "05_beir_missing" \
  --models \
    qwen3-embedding-4b \
    pretrained-4b-step-5750 \
    ru-only-step-180 \
    ru-en-step-300 \
    ru-promptriever-1.7b-step-320 \
    attention-only-4b \
    english-only-4b-step-400 \
    ru-promptriever-0.6b-final \
  --datasets en_mteb_retrieval

echo
had_errors=0
if grep -R -n -E '\[ERROR\]|Error evaluating|Traceback' "$LOG_DIR"; then
  had_errors=1
fi

ARCHIVE="$SCRIPT_DIR/final-paper-missing-results.tar.gz"
tar -czf "$ARCHIVE" \
  "$OUTPUT_DIR" \
  "$CONFIG" \
  "run_final_paper_missing.sh"

echo "Results: $SCRIPT_DIR/$OUTPUT_DIR"
echo "Logs:    $SCRIPT_DIR/$LOG_DIR"
echo "Archive: $ARCHIVE"
sha256sum "$ARCHIVE"

if (( had_errors )); then
  echo "One or more evaluations reported errors; partial results were archived." >&2
  exit 2
fi

echo "All mandatory final-paper reruns finished without logged errors."
