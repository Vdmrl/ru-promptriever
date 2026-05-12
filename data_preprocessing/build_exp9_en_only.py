"""
Build the English-only ablation dataset for Experiment 9.

Target composition (200k examples):
  - 100k English Synthetic w/ instruction  — from samaya-ai/msmarco-w-instructions
  - 100k English Synthetic w/o instruction — from samaya-ai/msmarco-w-instructions

Purpose: Ablation study to determine whether quality improvements come from
our training strategy (Russian data, instruction mixing) or simply from using
a more capable base model (Qwen3-4B).

Usage:
    python build_exp9_en_only.py --output_dir ./exp9_en_only_dataset --seed 42
"""

import os
import argparse
import random
import math

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_EN_INSTRUCTED = 100_000
NUM_EN_STANDARD = 100_000

# ---------------------------------------------------------------------------
# PyArrow schema (matches RetrieverDataset expected by train.py)
# ---------------------------------------------------------------------------
_PASSAGE_TYPE = pa.list_(
    pa.struct([
        ("docid", pa.string()),
        ("text",  pa.string()),
        ("title", pa.string()),
    ])
)
_PASSAGE_WITH_EXP_TYPE = pa.list_(
    pa.struct([
        ("docid",       pa.string()),
        ("text",        pa.string()),
        ("title",       pa.string()),
        ("explanation", pa.string()),
    ])
)
_TRAIN_SCHEMA = pa.schema([
    ("query_id",          pa.string()),
    ("query",             pa.string()),
    ("positive_passages", _PASSAGE_TYPE),
    ("negative_passages", _PASSAGE_WITH_EXP_TYPE),
    ("only_instruction",  pa.string()),
    ("only_query",        pa.string()),
    ("has_instruction",   pa.bool_()),
    ("new_negatives",     _PASSAGE_WITH_EXP_TYPE),
    ("is_repeated",       pa.bool_()),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_arrow(df: pd.DataFrame) -> pa.Table:
    """Convert a DataFrame with nested passage dicts to a typed Arrow table."""
    arrays = {
        "query_id": pa.array(
            [str(v) if v is not None else "" for v in df["query_id"]], type=pa.string()
        ),
        "query": pa.array(
            [str(v) if v is not None else "" for v in df["query"]], type=pa.string()
        ),
        "positive_passages": pa.array(
            df["positive_passages"].tolist(), type=_PASSAGE_TYPE,
        ),
        "negative_passages": pa.array(
            df["negative_passages"].tolist(), type=_PASSAGE_WITH_EXP_TYPE,
        ),
        "only_instruction": pa.array(
            [str(v) if v is not None else "" for v in df["only_instruction"]], type=pa.string()
        ),
        "only_query": pa.array(
            [str(v) if v is not None else "" for v in df["only_query"]], type=pa.string()
        ),
        "has_instruction": pa.array(
            [bool(v) if v is not None else False for v in df["has_instruction"]], type=pa.bool_()
        ),
        "new_negatives": pa.array(
            df["new_negatives"].tolist(), type=_PASSAGE_WITH_EXP_TYPE,
        ),
        "is_repeated": pa.array(
            [bool(v) if v is not None else False for v in df["is_repeated"]], type=pa.bool_()
        ),
    }
    return pa.table(arrays, schema=_TRAIN_SCHEMA)


def _norm_passage(p) -> dict:
    """Normalize a passage dict to have docid, text, title."""
    if p is None:
        return {"docid": "", "text": "", "title": ""}
    return {
        "docid": str(p.get("docid", p.get("id", "")) or ""),
        "text":  str(p.get("text", "") or ""),
        "title": str(p.get("title", "") or ""),
    }


def _norm_passage_with_exp(p) -> dict:
    """Normalize a passage dict to have docid, text, title, explanation."""
    base = _norm_passage(p)
    base["explanation"] = str(p.get("explanation", "") or "") if p else ""
    return base


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_en_synthetic(num_instructed: int, num_standard: int, seed: int) -> list[dict]:
    """Load subsets from the original English Promptriever dataset."""
    rng = random.Random(seed)
    print(f"  Loading samaya-ai/msmarco-w-instructions (target: {num_instructed} instr + {num_standard} std)...")
    ds = load_dataset("samaya-ai/msmarco-w-instructions", split="train",
                      token=os.environ.get("HF_TOKEN"))

    instructed_idx, standard_idx = [], []
    for i in tqdm(range(len(ds)), desc="  Classifying EN rows"):
        if ds[i].get("has_instruction", False):
            instructed_idx.append(i)
        else:
            standard_idx.append(i)

    print(f"    Available: {len(instructed_idx)} instructed, {len(standard_idx)} standard")

    rng.shuffle(instructed_idx)
    rng.shuffle(standard_idx)
    selected = instructed_idx[:num_instructed] + standard_idx[:num_standard]

    rows = []
    for idx in tqdm(selected, desc="  Sampling EN synthetic"):
        item = ds[idx]
        pos = [_norm_passage(p) for p in (item.get("positive_passages") or [])]
        neg = [_norm_passage_with_exp(p) for p in (item.get("negative_passages") or [])]
        new_neg = [_norm_passage_with_exp(p) for p in (item.get("new_negatives") or [])]
        rows.append({
            "query_id": f"en-synth-{item.get('query_id', str(idx))}",
            "query": item.get("query", ""),
            "positive_passages": pos,
            "negative_passages": neg,
            "only_instruction": item.get("only_instruction", "") or "",
            "only_query": item.get("only_query", item.get("query", "")),
            "has_instruction": bool(item.get("has_instruction", False)),
            "new_negatives": new_neg,
            "is_repeated": False,
        })

    instr_count = sum(1 for r in rows if r["has_instruction"])
    std_count = len(rows) - instr_count
    print(f"  EN Synthetic: {instr_count} instructed + {std_count} standard = {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, output_dir: str, split_name: str, chunk_size: int = 150000):
    """Save DataFrame as sharded parquet files."""
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    total_rows = len(df)
    num_shards = max(1, math.ceil(total_rows / chunk_size))

    for i in range(num_shards):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_rows)
        shard_df = df.iloc[start:end].reset_index(drop=True)

        shard_name = f"{split_name}-{i:05d}-of-{num_shards:05d}.parquet"
        shard_path = os.path.join(data_dir, shard_name)

        arrow_table = _df_to_arrow(shard_df)
        pq.write_table(
            arrow_table, shard_path,
            compression="snappy", version="2.6",
            use_dictionary=False, data_page_version="2.0",
        )
        print(f"    Saved {shard_path} ({len(shard_df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build the English-only ablation dataset for Experiment 9"
    )
    parser.add_argument("--output_dir", default="./exp9_en_only_dataset",
                        help="Directory to save the dataset (default: ./exp9_en_only_dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # ==================================================================
    # Load English Synthetic Data (100k instructed + 100k standard)
    # ==================================================================
    print("=" * 70)
    print("Loading English Synthetic data (100k instr + 100k std)")
    print("=" * 70)

    en_rows = load_en_synthetic(NUM_EN_INSTRUCTED, NUM_EN_STANDARD, args.seed)
    rng.shuffle(en_rows)

    # ==================================================================
    # Save
    # ==================================================================
    print("\n" + "=" * 70)
    print("Saving dataset")
    print("=" * 70)

    df = pd.DataFrame(en_rows)
    save_parquet(df, args.output_dir, "train")
    print(f"\n✓ Dataset saved to {args.output_dir} ({len(df)} rows)")


if __name__ == "__main__":
    main()
