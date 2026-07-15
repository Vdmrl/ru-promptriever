"""Fail-fast validation before launching an expensive evaluation run."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path

import mteb
import torch
import yaml
from huggingface_hub import get_token


MFOLLOW_REVISION = "09eecbe45c54b4a6dfb8e68e345cae77337768e2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if metadata.version("mteb") != "2.10.5":
        raise RuntimeError(f"Expected mteb==2.10.5, got {metadata.version('mteb')}")
    if str(config.get("device", "cuda:0")).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable: torch={torch.__version__}, torch CUDA={torch.version.cuda}"
        )

    needs_hf_auth = any(
        str(model.get("model_name_or_path", "")).startswith("samaya-ai/")
        for model in config.get("models", [])
    ) or any(
        dataset.get("data_path") == "Vladimirlv/ru-promptriever-dataset-v0.1"
        for dataset in config.get("datasets", [])
    )
    if needs_hf_auth and not get_token():
        raise RuntimeError("Hugging Face authentication required; run `hf auth login`")

    for model in config.get("models", []):
        if model.get("type") == "causal_lm" and not model.get("revision"):
            raise ValueError(f"Model {model.get('name')} has no immutable revision")
        if model.get("type") == "causal_lm" and not model.get("base_revision"):
            raise ValueError(f"Model {model.get('name')} has no base_revision")

    for dataset in config.get("datasets", []):
        if dataset.get("type") == "synthetic_test":
            for field in ("data_path", "revision", "instruction_negative_field"):
                if not dataset.get(field):
                    raise ValueError(
                        f"Synthetic dataset {dataset.get('name')} missing {field}"
                    )
        if dataset.get("type") == "mfollowir":
            if dataset.get("revision") != MFOLLOW_REVISION:
                raise ValueError("mFollowIR config does not pin the official revision")
            task = list(mteb.get_tasks(tasks=["mFollowIR"], languages=["rus"]))
            if len(task) != 1 or task[0].metadata.dataset["revision"] != MFOLLOW_REVISION:
                raise RuntimeError("Installed MTEB does not expose the expected mFollowIR task")
            if task[0].metadata.type != "InstructionReranking":
                raise RuntimeError("mFollowIR is not configured as InstructionReranking")

    print(f"OK: {args.config}")
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(0)}")
    print("All configured revisions and benchmark protocols passed preflight.")


if __name__ == "__main__":
    main()
