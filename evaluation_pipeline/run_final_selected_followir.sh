#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/eval_final_paper_missing.yaml"
OUTPUT_DIR="results_final_paper_missing"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available"
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

# Fail before spending GPU time unless default HF downloads resolve to the
# exact paper-selected step-180 and step-300 adapter weights.
python - <<'PY'
from huggingface_hub import HfApi

expected = {
    "Vladimirlv/ru-promptriever-qwen3-4b-ru-only": (
        "ce7d06fa3121942b09d9185fa5bfef7811ed6b24eebc239d16b851a0cdbe69c7"
    ),
    "Vladimirlv/ru-promptriever-qwen3-4b": (
        "9bf6bb659f4335d012cdff6b3c4026c3ffc69cdb20993bc501f55ac1d1384541"
    ),
}

api = HfApi()
for repo_id, expected_sha256 in expected.items():
    info = api.model_info(repo_id, revision="main", files_metadata=True)
    adapter = next(
        sibling
        for sibling in info.siblings
        if sibling.rfilename == "adapter_model.safetensors"
    )
    actual_sha256 = adapter.lfs.sha256
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{repo_id} default adapter mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    print(f"verified default: {repo_id}@{info.sha} -> {actual_sha256}")
PY

git -C .. rev-parse HEAD | tee "$OUTPUT_DIR/git-revision.txt"
cp "$CONFIG" "$OUTPUT_DIR/eval_final_paper_missing.yaml"

python evaluate.py \
  --config "$CONFIG" \
  --models ru-only ru-en \
  --datasets followir_eng \
  --skip-existing \
  --no-summary \
  2>&1 | tee -a "$LOG_DIR/selected_ru_models_followir.log"

if grep -n -E '\[ERROR\]|Error evaluating|Traceback' \
  "$LOG_DIR/selected_ru_models_followir.log"; then
  echo "FollowIR evaluation reported errors." >&2
  exit 2
fi

echo "Selected ru-only and ru-en FollowIR evaluations completed."
echo "Results: $SCRIPT_DIR/$OUTPUT_DIR"
