"""I/O helpers for reading and writing JSONL files."""

import json
import os
import glob


def read_jsonl(file_path: str):
    """Lazily yield parsed objects from a JSONL file, skipping blank lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(data, file_path: str) -> None:
    """Write an iterable of objects to a JSONL file, creating directories as needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_jsonl_files(directory: str) -> list[str]:
    """Return a sorted list of all *.jsonl file paths in ``directory``."""
    return sorted(glob.glob(os.path.join(directory, "*.jsonl")))