"""Evaluation provenance and safe result-cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


TRACKED_PACKAGES = (
    "mteb",
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
    "peft",
    "accelerate",
    "numpy",
    "pytrec-eval",
)


def git_revision(workdir: str | os.PathLike[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_is_dirty(workdir: str | os.PathLike[str]) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=workdir,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def protocol_payload(
    config: dict,
    model_cfg: dict,
    dataset_cfg: dict,
    code_revision: str,
) -> dict[str, Any]:
    return {
        "code_revision": code_revision,
        "global": {
            key: config.get(key)
            for key in (
                "device",
                "dtype",
                "generic_instruction",
                "retrieval_top_k",
            )
        },
        "model": model_cfg,
        "dataset": dataset_cfg,
    }


def protocol_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def find_matching_result(
    output_dir: str | os.PathLike[str], fingerprint: str
) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("protocol_fingerprint") == fingerprint:
            return path
    return None


def write_run_manifest(
    output_dir: str | os.PathLike[str],
    config_path: str,
    config: dict,
    code_revision: str,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    cuda = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except (ImportError, OSError) as exc:
        cuda["error"] = repr(exc)

    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "config": config,
        "git_revision": code_revision,
        "git_dirty": git_is_dirty(Path(__file__).resolve().parents[2]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda": cuda,
        "nvidia_smi": driver,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
