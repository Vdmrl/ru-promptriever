"""
Build a mixed dataset for continue training ru-Promptriever.

Sources:
  1. MIRACL Russian (train + dev) — human-annotated retrieval pairs
  2. Mr. TyDi Russian (train) — human-annotated retrieval pairs
  3. Synthetic instructed subset — from the existing ru-promptriever-dataset
  4. Synthetic standard subset — from the existing ru-promptriever-dataset

Usage:
    python build_continue_training_dataset.py \
        --output_dir ./continue_training_dataset \
        --num_synthetic_instructed 11000 \
        --num_synthetic_standard 6000 \
        --seed 42
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
# PyArrow schema (must match the existing dataset schema for RetrieverDataset)
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
# Load external datasets
# ---------------------------------------------------------------------------

def load_miracl_russian() -> list[dict]:
    """Load MIRACL Russian train + dev splits and convert to our row format."""
    rows = []

    for split in ["train", "dev"]:
        print(f"  Loading MIRACL Russian {split}...")
        ds = load_dataset("miracl/miracl", "ru", split=split, trust_remote_code=True)

        for item in tqdm(ds, desc=f"  MIRACL {split}"):
            query = item["query"]
            query_id = f"miracl-{item['query_id']}"

            pos_passages = item.get("positive_passages", [])
            neg_passages = item.get("negative_passages", [])

            if not pos_passages:
                continue

            # Format positive passages
            formatted_pos = [_norm_passage(p) for p in pos_passages[:1]]

            # Format negative passages (these are human-annotated negatives)
            formatted_neg = [_norm_passage_with_exp(p) for p in neg_passages]

            rows.append({
                "query_id": query_id,
                "query": query,
                "positive_passages": formatted_pos,
                "negative_passages": formatted_neg,
                "only_instruction": "",
                "only_query": query,
                "has_instruction": False,
                "new_negatives": [],
                "is_repeated": False,
            })

    print(f"  MIRACL total: {len(rows)} rows")
    return rows


def load_mrtydi_russian() -> list[dict]:
    """Load Mr. TyDi Russian train split and convert to our row format."""
    rows = []

    print("  Loading Mr. TyDi Russian train...")
    ds = load_dataset("castorini/mr-tydi", "russian", split="train")

    for item in tqdm(ds, desc="  Mr. TyDi train"):
        query = item["query"]
        query_id = f"mrtydi-{item['query_id']}"

        pos_passages = item.get("positive_passages", [])
        neg_passages = item.get("negative_passages", [])

        if not pos_passages:
            continue

        formatted_pos = [_norm_passage(p) for p in pos_passages[:1]]
        formatted_neg = [_norm_passage_with_exp(p) for p in neg_passages]

        rows.append({
            "query_id": query_id,
            "query": query,
            "positive_passages": formatted_pos,
            "negative_passages": formatted_neg,
            "only_instruction": "",
            "only_query": query,
            "has_instruction": False,
            "new_negatives": [],
            "is_repeated": False,
        })

    print(f"  Mr. TyDi total: {len(rows)} rows")
    return rows


def load_synthetic_subset(
    num_instructed: int,
    num_standard: int,
    seed: int = 42,
    from_end: bool = False,
) -> list[dict]:
    """Load a random subset of the existing ru-promptriever synthetic dataset.
    
    If from_end=True, takes the LAST N items after shuffle instead of the first N.
    Useful to get a different subset from a previous run that used from_end=False.
    """
    rows = []
    rng = random.Random(seed)

    print(f"  Loading synthetic dataset (target: {num_instructed} instructed + {num_standard} standard, from_end={from_end})...")
    ds = load_dataset("Vladimirlv/ru-promptriever-dataset", split="train", num_proc=8)

    # Separate instructed and standard rows
    instructed_indices = []
    standard_indices = []
    for i in range(len(ds)):
        if ds[i].get("has_instruction", False):
            instructed_indices.append(i)
        else:
            standard_indices.append(i)

    print(f"    Available: {len(instructed_indices)} instructed, {len(standard_indices)} standard")

    # Sample subsets
    rng.shuffle(instructed_indices)
    rng.shuffle(standard_indices)

    if from_end:
        selected_instructed = instructed_indices[-num_instructed:]
        selected_standard = standard_indices[-num_standard:]
    else:
        selected_instructed = instructed_indices[:num_instructed]
        selected_standard = standard_indices[:num_standard]

    print(f"    Selected: {len(selected_instructed)} instructed, {len(selected_standard)} standard")

    # Convert to our format (already in the right format, just need to extract)
    for idx in tqdm(selected_instructed + selected_standard, desc="  Sampling synthetic"):
        item = ds[idx]

        # Normalize passages to ensure consistent format
        pos = item.get("positive_passages", [])
        neg = item.get("negative_passages", [])
        new_neg = item.get("new_negatives", [])

        formatted_pos = [_norm_passage(p) for p in (pos or [])]
        formatted_neg = [_norm_passage_with_exp(p) for p in (neg or [])]
        formatted_new_neg = [_norm_passage_with_exp(p) for p in (new_neg or [])]

        rows.append({
            "query_id": f"synth-{item.get('query_id', str(idx))}",
            "query": item.get("query", ""),
            "positive_passages": formatted_pos,
            "negative_passages": formatted_neg,
            "only_instruction": item.get("only_instruction", "") or "",
            "only_query": item.get("only_query", ""),
            "has_instruction": bool(item.get("has_instruction", False)),
            "new_negatives": formatted_new_neg,
            "is_repeated": False,
        })

    print(f"  Synthetic total: {len(rows)} rows")
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


def main():
    parser = argparse.ArgumentParser(
        description="Build mixed dataset for continue training ru-Promptriever"
    )
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the mixed dataset")
    parser.add_argument("--num_synthetic_instructed", type=int, default=11000,
                        help="Number of synthetic instructed rows to include (default: 11000)")
    parser.add_argument("--num_synthetic_standard", type=int, default=6000,
                        help="Number of synthetic standard rows to include (default: 6000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--from_end", action="store_true",
                        help="Take synthetic queries from the END of shuffled list instead of start. "
                             "Use this to get a different subset from a previous run with same seed.")
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="Optional HuggingFace repo ID to upload the dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # ==============================
    # Stage 1: Load all data sources
    # ==============================
    print("=" * 60)
    print("Stage 1: Loading data sources")
    print("=" * 60)

    print("\n[1/3] MIRACL Russian:")
    miracl_rows = load_miracl_russian()

    print("\n[2/3] Mr. TyDi Russian:")
    mrtydi_rows = load_mrtydi_russian()

    print("\n[3/3] Synthetic subset:")
    synth_rows = load_synthetic_subset(
        num_instructed=args.num_synthetic_instructed,
        num_standard=args.num_synthetic_standard,
        seed=args.seed,
        from_end=args.from_end,
    )

    # ==============================
    # Stage 2: Combine and shuffle
    # ==============================
    print("\n" + "=" * 60)
    print("Stage 2: Combining and shuffling")
    print("=" * 60)

    all_rows = miracl_rows + mrtydi_rows + synth_rows
    rng.shuffle(all_rows)

    real_count = len(miracl_rows) + len(mrtydi_rows)
    synth_instr_count = sum(1 for r in synth_rows if r["has_instruction"])
    synth_std_count = sum(1 for r in synth_rows if not r["has_instruction"])

    print(f"\n  Real data (MIRACL + Mr. TyDi): {real_count}")
    print(f"  Synthetic instructed:          {synth_instr_count}")
    print(f"  Synthetic standard:            {synth_std_count}")
    print(f"  Total:                         {len(all_rows)}")

    # ==============================
    # Stage 3: Save as parquet
    # ==============================
    print("\n" + "=" * 60)
    print("Stage 3: Saving dataset")
    print("=" * 60)

    df = pd.DataFrame(all_rows)
    save_parquet(df, args.output_dir, "train")

    # Write a simple README
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# ru-Promptriever Continue Training Dataset\n\n")
        f.write("Mixed dataset for continue training ru-Promptriever.\n\n")
        f.write("## Composition\n\n")
        f.write(f"- **MIRACL Russian** (train + dev): {len(miracl_rows)} rows\n")
        f.write(f"- **Mr. TyDi Russian** (train): {len(mrtydi_rows)} rows\n")
        f.write(f"- **Synthetic instructed**: {synth_instr_count} rows\n")
        f.write(f"- **Synthetic standard**: {synth_std_count} rows\n")
        f.write(f"- **Total**: {len(all_rows)} rows\n")

    print(f"\n✓ Dataset saved to {args.output_dir}")

    # ==============================
    # Optional: push to HuggingFace
    # ==============================
    if args.push_to_hub:
        print(f"\nUploading to HuggingFace: {args.push_to_hub}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=args.push_to_hub, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=args.output_dir,
                repo_id=args.push_to_hub,
                repo_type="dataset",
            )
            print("✓ Upload complete!")
        except Exception as e:
            print(f"✗ Upload failed: {e}")


if __name__ == "__main__":
    main()
