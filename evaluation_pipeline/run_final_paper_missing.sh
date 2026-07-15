#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/eval_final_paper_missing.yaml"
OUTPUT_DIR="results_final_paper_missing"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
STATUS_FILE="$OUTPUT_DIR/RUN_STATUS.txt"

on_exit() {
  local exit_code=$?
  trap - EXIT
  if (( exit_code == 0 )); then
    printf 'COMPLETE\n' > "$STATUS_FILE"
  else
    printf 'FAILED exit_code=%s\n' "$exit_code" > "$STATUS_FILE"
  fi
  exit "$exit_code"
}
trap on_exit EXIT
printf 'STARTED %s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"

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

# Verify that the canonical Hub repositories contain exactly the adapters
# selected for the paper (Ru-only step 180, Ru+En step 300, 1.7B step 320).
# Immutable revisions prevent future drift; SHA checks also prove that the
# promoted canonical repositories match the original selected checkpoints.
python - <<'PY'
from huggingface_hub import HfApi

expected = {
    ("Vladimirlv/ru-promptriever-qwen3-4b-ru-only", "b7aef87a7acfff2c0366f5671a207d1b264f2b0a"):
        "ce7d06fa3121942b09d9185fa5bfef7811ed6b24eebc239d16b851a0cdbe69c7",
    ("Vladimirlv/ru-promptriever-qwen3-4b", "699ec8b3176ea2f268f165c6754d89606ab3aecb"):
        "9bf6bb659f4335d012cdff6b3c4026c3ffc69cdb20993bc501f55ac1d1384541",
    ("Vladimirlv/ru-promptriever-qwen3-1.7b", "3bda1fae7b8476bd13e0e753043c394e1e751ff8"):
        "b1ffbb4fe4aced8f634a048732274aa342d4e835ce95d56bcd9c60d2ff2084fd",
}

api = HfApi()
for (repo_id, revision), expected_sha256 in expected.items():
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    adapter = next(
        item for item in info.siblings
        if item.rfilename == "adapter_model.safetensors"
    )
    actual = adapter.lfs.sha256
    if actual != expected_sha256:
        raise RuntimeError(
            f"Selected adapter mismatch for {repo_id}@{revision}: "
            f"expected {expected_sha256}, got {actual}"
        )
    print(f"verified selected adapter: {repo_id}@{info.sha} -> {actual}")
PY

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
  printf 'RUNNING %s %s\n' "$stage" "$(date --iso-8601=seconds)" > "$STATUS_FILE"
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
  printf 'FINISHED %s %s\n' "$stage" "$(date --iso-8601=seconds)" > "$STATUS_FILE"
}

# InstructIR has not yet been rerun with the immutable final checkpoints.
# Both Promptriever sizes are included so the final table can report 7B and 8B.
run_stage "01_instructir_missing" \
  --models \
    qwen3-embedding-4b \
    promptriever-7b \
    promptriever-8b \
    pretrained-4b-step-5750 \
    ru-only \
    ru-en \
    ru-promptriever-1.7b-step-320 \
    english-only-4b-step-400 \
  --datasets instructir

# Preserve the completed selected-checkpoint FollowIR runs. If this script is
# resumed on a fresh machine without those artifacts, recreate them once.
selected_followir_complete=1
for model in ru-only ru-en; do
  for task in \
    Robust04InstructionRetrieval \
    Core17InstructionRetrieval \
    News21InstructionRetrieval; do
    if [[ ! -f "$OUTPUT_DIR/predictions/$model/${task}_predictions.json" ]]; then
      selected_followir_complete=0
    fi
  done
done
if (( ! selected_followir_complete )); then
  run_stage "02a_followir_selected_missing" \
    --models ru-only ru-en \
    --datasets followir_eng
fi

# Promptriever-7B/8B and the selected Ru-only/Ru+En predictions are complete.
# Only Qwen3-Embedding and selected 1.7B remain for the harmonized main table.
run_stage "02_followir_missing" \
  --models \
    qwen3-embedding-4b \
    ru-promptriever-1.7b-step-320 \
  --datasets followir_eng

# Missing mFollowIR rows only.  Ru+En step 300 and Promptriever-8B were
# already evaluated with the corrected official protocol on 2026-07-15.
run_stage "03_mfollowir_missing" \
  --models \
    ru-only \
    ru-promptriever-1.7b-step-320 \
    pretrained-4b-step-5750 \
    english-only-4b-step-400 \
    attention-only-4b \
    ru-promptriever-0.6b-final \
    qwen3-embedding-4b \
    multilingual-e5-large \
    bge-m3 \
  --datasets mfollowir_ru

# Recompute only model rows produced by our pipeline.  The published general
# retrieval baselines are unaffected by the mFollowIR/FollowIR qrels fixes.
run_stage "04_rubq_missing" \
  --models \
    qwen3-embedding-4b \
    pretrained-4b-step-5750 \
    ru-only \
    ru-en \
    ru-promptriever-1.7b-step-320 \
    attention-only-4b \
    english-only-4b-step-400 \
    ru-promptriever-0.6b-final \
  --datasets rumteb_retrieval

run_stage "05_beir_missing" \
  --models \
    qwen3-embedding-4b \
    pretrained-4b-step-5750 \
    ru-only \
    ru-en \
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
