"""
Build the "Gold Mix" dataset for Experiment 8.

Target composition (~65,000 examples):
  1. ~11k Russian Real queries       — MIRACL ru (train+dev) + MrTyDi ru (train), ALL available
  2.  22k Russian Synthetic w/ instr — from Vladimirlv/ru-promptriever-dataset
  3.   8k Russian Synthetic w/o instr— from Vladimirlv/ru-promptriever-dataset
  4.  17k English Synthetic w/ instr — from samaya-ai/msmarco-w-instructions
  5.   7k English Synthetic w/o instr— from samaya-ai/msmarco-w-instructions
                                       ──────
                              Total ≈ 65k (exact count depends on MIRACL+MrTyDi availability)

Note: XOR-TyDi and RuBQ were considered but rejected:
  - XOR-TyDi is cross-lingual QA (RU question → EN passage) — no Russian passages
  - RuBQ is KBQA over Wikidata — answers are entity labels, not text passages

Usage:
    python build_exp8_gold_mix.py --output_dir ./exp8_gold_mix_dataset --seed 42
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
NUM_RU_SYNTH_INSTRUCTED = 22_000
NUM_RU_SYNTH_STANDARD = 8_000
NUM_EN_SYNTH_INSTRUCTED = 17_000
NUM_EN_SYNTH_STANDARD = 7_000

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


def _make_real_row(query_id: str, query: str, pos_passages: list, neg_passages: list) -> dict:
    """Create a row dict for a real (non-instructed) retrieval query."""
    return {
        "query_id": query_id,
        "query": query,
        "positive_passages": [_norm_passage(p) for p in pos_passages[:1]],
        "negative_passages": [_norm_passage_with_exp(p) for p in neg_passages],
        "only_instruction": "",
        "only_query": query,
        "has_instruction": False,
        "new_negatives": [],
        "is_repeated": False,
    }


# ---------------------------------------------------------------------------
# 1. Russian Real Data Loaders
# ---------------------------------------------------------------------------

def load_miracl_russian() -> list[dict]:
    """Load MIRACL Russian train + dev splits."""
    rows = []
    for split in ["train", "dev"]:
        print(f"  Loading MIRACL Russian {split}...")
        ds = load_dataset("miracl/miracl", "ru", split=split, trust_remote_code=True,
                          token=os.environ.get("HF_TOKEN"))
        for item in tqdm(ds, desc=f"  MIRACL {split}"):
            pos = item.get("positive_passages", [])
            neg = item.get("negative_passages", [])
            if not pos:
                continue
            rows.append(_make_real_row(
                f"miracl-{item['query_id']}", item["query"], pos, neg
            ))
    print(f"  MIRACL total: {len(rows)} rows")
    return rows


def load_mrtydi_russian() -> list[dict]:
    """Load Mr. TyDi Russian train split."""
    rows = []
    print("  Loading Mr. TyDi Russian train...")
    ds = load_dataset("castorini/mr-tydi", "russian", split="train",
                      token=os.environ.get("HF_TOKEN"))
    for item in tqdm(ds, desc="  Mr. TyDi train"):
        pos = item.get("positive_passages", [])
        neg = item.get("negative_passages", [])
        if not pos:
            continue
        rows.append(_make_real_row(
            f"mrtydi-{item['query_id']}", item["query"], pos, neg
        ))
    print(f"  Mr. TyDi total: {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# 2. Russian Synthetic Data Loader
# ---------------------------------------------------------------------------

def load_ru_synthetic(num_instructed: int, num_standard: int, seed: int) -> list[dict]:
    """Load subsets from existing ru-promptriever-dataset."""
    rng = random.Random(seed)
    print(f"  Loading ru-promptriever-dataset (target: {num_instructed} instr + {num_standard} std)...")
    ds = load_dataset("Vladimirlv/ru-promptriever-dataset", split="train", num_proc=8,
                      token=os.environ.get("HF_TOKEN"))

    instructed_idx, standard_idx = [], []
    for i in range(len(ds)):
        if ds[i].get("has_instruction", False):
            instructed_idx.append(i)
        else:
            standard_idx.append(i)

    print(f"    Available: {len(instructed_idx)} instructed, {len(standard_idx)} standard")

    rng.shuffle(instructed_idx)
    rng.shuffle(standard_idx)
    selected = instructed_idx[:num_instructed] + standard_idx[:num_standard]

    rows = []
    for idx in tqdm(selected, desc="  Sampling RU synthetic"):
        item = ds[idx]
        pos = [_norm_passage(p) for p in (item.get("positive_passages") or [])]
        neg = [_norm_passage_with_exp(p) for p in (item.get("negative_passages") or [])]
        new_neg = [_norm_passage_with_exp(p) for p in (item.get("new_negatives") or [])]
        rows.append({
            "query_id": f"ru-synth-{item.get('query_id', str(idx))}",
            "query": item.get("query", ""),
            "positive_passages": pos,
            "negative_passages": neg,
            "only_instruction": item.get("only_instruction", "") or "",
            "only_query": item.get("only_query", ""),
            "has_instruction": bool(item.get("has_instruction", False)),
            "new_negatives": new_neg,
            "is_repeated": False,
        })

    instr_count = sum(1 for r in rows if r["has_instruction"])
    std_count = len(rows) - instr_count
    print(f"  RU Synthetic: {instr_count} instructed + {std_count} standard = {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# 3. English Synthetic Data Loader
# ---------------------------------------------------------------------------

def load_en_synthetic(num_instructed: int, num_standard: int, seed: int) -> list[dict]:
    """Load subsets from the original English Promptriever dataset (msmarco-w-instructions)."""
    rng = random.Random(seed + 1)  # Different seed from RU to avoid correlation
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
        description="Build the Gold Mix dataset for Experiment 8 (balanced multilingual)"
    )
    parser.add_argument("--output_dir", default="./exp8_gold_mix_dataset",
                        help="Directory to save the dataset (default: ./exp8_gold_mix_dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="Optional HuggingFace repo ID to upload the dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # ==================================================================
    # Stage 1: Load Russian Real Data (all available from MIRACL + MrTyDi)
    # ==================================================================
    print("=" * 70)
    print("Stage 1: Loading Russian REAL data (MIRACL + MrTyDi, all available)")
    print("=" * 70)

    print("\n[1/2] MIRACL Russian:")
    miracl_rows = load_miracl_russian()

    print("\n[2/2] Mr. TyDi Russian:")
    mrtydi_rows = load_mrtydi_russian()

    ru_real = miracl_rows + mrtydi_rows
    rng.shuffle(ru_real)
    print(f"\n  → Total Russian real: {len(ru_real)} rows")

    # ==================================================================
    # Stage 2: Load Russian Synthetic Data (26k instructed + 10k standard)
    # ==================================================================
    print("\n" + "=" * 70)
    print("Stage 2: Loading Russian SYNTHETIC data (26k instr + 10k std)")
    print("=" * 70)

    ru_synth = load_ru_synthetic(NUM_RU_SYNTH_INSTRUCTED, NUM_RU_SYNTH_STANDARD, args.seed)

    # ==================================================================
    # Stage 3: Load English Synthetic Data (20k instructed + 8k standard)
    # ==================================================================
    print("\n" + "=" * 70)
    print("Stage 3: Loading English SYNTHETIC data (20k instr + 8k std)")
    print("=" * 70)

    en_synth = load_en_synthetic(NUM_EN_SYNTH_INSTRUCTED, NUM_EN_SYNTH_STANDARD, args.seed)

    # ==================================================================
    # Stage 4: Combine and shuffle
    # ==================================================================
    print("\n" + "=" * 70)
    print("Stage 4: Combining and shuffling")
    print("=" * 70)

    all_rows = ru_real + ru_synth + en_synth
    rng.shuffle(all_rows)

    # Compute stats
    ru_real_count = len(ru_real)
    ru_instr = sum(1 for r in ru_synth if r["has_instruction"])
    ru_std = sum(1 for r in ru_synth if not r["has_instruction"])
    en_instr = sum(1 for r in en_synth if r["has_instruction"])
    en_std = sum(1 for r in en_synth if not r["has_instruction"])

    print(f"\n  1. Russian Real (MIRACL+MrTyDi):  {ru_real_count:>6}")
    print(f"  2. Russian Synth (instructed):    {ru_instr:>6}")
    print(f"  3. Russian Synth (standard):      {ru_std:>6}")
    print(f"  4. English Synth (instructed):    {en_instr:>6}")
    print(f"  5. English Synth (standard):      {en_std:>6}")
    print(f"  {'─' * 42}")
    print(f"  TOTAL:                            {len(all_rows):>6}")

    # ==================================================================
    # Stage 5: Save as parquet
    # ==================================================================
    print("\n" + "=" * 70)
    print("Stage 5: Saving dataset")
    print("=" * 70)

    df = pd.DataFrame(all_rows)
    save_parquet(df, args.output_dir, "train")

    # Write README
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 8 — Gold Mix Dataset\n\n")
        f.write("Balanced multilingual dataset for training ru-Promptriever.\n\n")
        f.write("## Composition\n\n")
        f.write(f"| Category | Count |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Russian Real (MIRACL ru + MrTyDi ru) | {ru_real_count} |\n")
        f.write(f"| Russian Synthetic (instructed) | {ru_instr} |\n")
        f.write(f"| Russian Synthetic (standard) | {ru_std} |\n")
        f.write(f"| English Synthetic (instructed) | {en_instr} |\n")
        f.write(f"| English Synthetic (standard) | {en_std} |\n")
        f.write(f"| **Total** | **{len(all_rows)}** |\n")
        f.write(f"\n## Notes\n\n")
        f.write("- XOR-TyDi rejected: cross-lingual QA (RU query → EN passage), no RU passages\n")
        f.write("- RuBQ rejected: KBQA over Wikidata, answers are entity labels, not passages\n")
        f.write("- mMarco-ru rejected: machine-translated, low quality\n")

    print(f"\n✓ Dataset saved to {args.output_dir}")

    # ==================================================================
    # Optional: push to HuggingFace
    # ==================================================================
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
