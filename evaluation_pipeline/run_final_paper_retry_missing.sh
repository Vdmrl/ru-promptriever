#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/eval_final_paper_missing.yaml"
OUTPUT_DIR="results_final_paper_missing"
LOG_DIR="$OUTPUT_DIR/logs_retry"
STATUS_FILE="$OUTPUT_DIR/RETRY_STATUS.txt"
mkdir -p "$LOG_DIR"

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

python - <<'PY'
import torch
from huggingface_hub import HfApi

assert torch.cuda.is_available(), "CUDA is not available"
print("torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

expected = {
    ("Vladimirlv/ru-promptriever-qwen3-4b-attn", "1c6c958776a685c4a73baff566e4d826bb8a8e35"):
        {"config.json", "model.safetensors.index.json", "tokenizer_config.json"},
    ("Vladimirlv/ru-promptriever-qwen3-0.6b", "8090f4c29f41a41130f59b1549c64b5904de23d2"):
        {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"},
}
api = HfApi()
for (repo_id, revision), required in expected.items():
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    files = {item.rfilename for item in info.siblings}
    missing = required - files
    if missing:
        raise RuntimeError(f"{repo_id}@{revision} is missing {sorted(missing)}")
    print(f"verified retry model: {repo_id}@{info.sha}")
PY

run_stage() {
  local stage="$1"
  shift
  printf 'RUNNING %s %s\n' "$stage" "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  python evaluate.py \
    --config "$CONFIG" \
    --skip-existing \
    --no-summary \
    "$@" \
    2>&1 | tee -a "$LOG_DIR/${stage}.log"
  printf 'FINISHED %s %s\n' "$stage" "$(date --iso-8601=seconds)" > "$STATUS_FILE"
}

# Five failed rows caused by missing mutable MTEB metadata on the wrapper.
run_stage "01_qwen3_embedding" \
  --models qwen3-embedding-4b \
  --datasets instructir followir_eng mfollowir_ru rumteb_retrieval en_mteb_retrieval

# Two failed mFollowIR rows caused by the same metadata issue.
run_stage "02_encoder_mfollowir" \
  --models multilingual-e5-large bge-m3 \
  --datasets mfollowir_ru

# Six failed rows caused by Hub resolution failures in the first run.
run_stage "03_causal_ablations" \
  --models attention-only-4b ru-promptriever-0.6b-final \
  --datasets mfollowir_ru rumteb_retrieval en_mteb_retrieval

if grep -R -n -E '\[ERROR\]|Failed to load model|Error evaluating|Traceback' "$LOG_DIR"; then
  echo "Retry still contains evaluation errors." >&2
  exit 2
fi

ARCHIVE="$SCRIPT_DIR/final-paper-missing-results.tar.gz"
tar -czf "$ARCHIVE" \
  "$OUTPUT_DIR" \
  "$CONFIG" \
  "run_final_paper_missing.sh" \
  "run_final_paper_retry_missing.sh"

echo "Retry completed and archive rebuilt: $ARCHIVE"
sha256sum "$ARCHIVE"
