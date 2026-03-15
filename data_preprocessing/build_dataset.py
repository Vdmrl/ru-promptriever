import sys
import os
import argparse
import random
import math

import pandas as pd
from tqdm import tqdm

# Add the current directory (data_preprocessing) to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.io import read_jsonl, get_jsonl_files
from utils.bm25 import BM25Retriever


def normalize_raw_record(record):
    """Convert a raw (unfiltered) record into the flat format expected by the pipeline."""
    idata = record["instruction_data"]
    mdata = record["mining_data"]

    # Collect all instruction negatives (matches_both == False)
    valid_negs = []
    for idx, doc in enumerate(mdata.get("documents", [])):
        if not doc.get("matches_both", False):
            valid_negs.append(
                {
                    "id": f"{record['query_id']}_{idx}",
                    "text": doc["passage"],
                    "title": doc.get("title", ""),
                }
            )

    return {
        "query_id": record["query_id"],
        "original_query": record["original_query"],
        "rewritten_query": idata.get("rewritten_query", record["original_query"]),
        "instruction": idata["instruction"],
        "rewritten_original_positive": {
            "id": str(record["original_positive_id"]),
            "text": idata["rewritten_pos_doc"],
            "title": idata.get("rewritten_pos_title", ""),
        },
        "rewritten_original_negative": {
            "id": str(record["original_negative_id"]),
            "text": idata["rewritten_neg_doc"],
            "title": idata.get("rewritten_neg_title", ""),
        },
        "final_positive": {
            "id": str(record["original_positive_id"]),
            "text": idata["rewritten_pos_doc"],
            "title": idata.get("rewritten_pos_title", ""),
            "source": "rewritten",
        },
        "valid_synthetic_negatives": valid_negs,
    }


def collect_split_doc_pool(records):
    """Collect all unique documents from a set of records into a document pool."""
    pool = {}
    for record in records:
        for key in ["rewritten_original_positive", "rewritten_original_negative"]:
            item = record.get(key)
            if item and item.get("text"):
                doc_id = str(item["id"])
                if doc_id not in pool:
                    pool[doc_id] = {
                        "docid": doc_id,
                        "text": item["text"],
                        "title": item.get("title", ""),
                    }
        # Also add final_positive and synthetic negatives to the pool
        fp = record.get("final_positive")
        if fp and fp.get("text"):
            fp_id = str(fp["id"])
            if fp_id not in pool:
                pool[fp_id] = {
                    "docid": fp_id,
                    "text": fp["text"],
                    "title": fp.get("title", ""),
                }
        for sn in record.get("valid_synthetic_negatives", []):
            sn_id = str(sn["id"])
            if sn_id not in pool:
                pool[sn_id] = {
                    "docid": sn_id,
                    "text": sn["text"],
                    "title": sn.get("title", ""),
                }
    return pool


def build_eval_rows(records, doc_pool, rng):
    """Build rows for val/test splits \u2014 no BM25 negatives, only the document pool."""
    rows = []
    for record in records:
        q_id = record["query_id"]
        orig_q = record["original_query"]
        rewr_q = record.get("rewritten_query", orig_q)
        instruct = record["instruction"]
        is_repeated = record.get("is_repeated", False)

        # Mix 50/50
        chosen_q = orig_q if rng.random() < 0.5 else rewr_q

        # Standard query row (without instruction)
        std_pos_id = str(record["rewritten_original_positive"]["id"])
        std_pos_doc = doc_pool.get(std_pos_id) or {
            "docid": std_pos_id,
            "text": record["rewritten_original_positive"]["text"],
            "title": record["rewritten_original_positive"].get("title", ""),
        }

        rows.append(
            {
                "query_id": str(q_id),
                "query": chosen_q,
                "positive_passages": [std_pos_doc],
                "negative_passages": [],
                "only_instruction": None,
                "only_query": chosen_q,
                "has_instruction": False,
                "new_negatives": [],
                "is_repeated": is_repeated,
            }
        )

        # Instruct query row
        inst_pos_info = record["final_positive"]
        inst_pos_doc = {
            "docid": str(inst_pos_info["id"]),
            "text": inst_pos_info["text"],
            "title": inst_pos_info.get("title", ""),
        }

        valid_synt_negs = []
        for sn in record.get("valid_synthetic_negatives", []):
            valid_synt_negs.append(
                {
                    "docid": str(sn["id"]),
                    "text": sn["text"],
                    "title": sn.get("title", ""),
                    "explanation": "instruction_negative",
                }
            )

        rows.append(
            {
                "query_id": f"{q_id}-instruct",
                "query": f"{chosen_q} {instruct}".strip(),
                "positive_passages": [inst_pos_doc],
                "negative_passages": valid_synt_negs,
                "only_instruction": instruct,
                "only_query": chosen_q,
                "has_instruction": True,
                "new_negatives": valid_synt_negs,
                "is_repeated": is_repeated,
            }
        )

    return rows


def build_train_rows(records, doc_registry, std_negs_batch, inst_negs_batch, rng):
    """Build rows for training split \u2014 with BM25 hard negatives."""
    rows = []
    for idx, record in enumerate(tqdm(records, desc="Assembling train rows")):
        q_id = record["query_id"]
        orig_q = record["original_query"]
        rewr_q = record.get("rewritten_query", orig_q)
        instruct = record["instruction"]
        is_repeated = record.get("is_repeated", False)

        # Mix 50/50
        chosen_q = orig_q if rng.random() < 0.5 else rewr_q

        # Standard query row (without instruction)
        std_pos_id = str(record["rewritten_original_positive"]["id"])
        std_pos_doc = doc_registry.get(std_pos_id) or {
            "docid": std_pos_id,
            "text": record["rewritten_original_positive"]["text"],
            "title": record["rewritten_original_positive"].get("title", ""),
        }

        std_neg_docs = []
        for cand in std_negs_batch[idx]:
            if cand["docid"] != std_pos_id:
                std_neg_docs.append(cand)
            if len(std_neg_docs) >= 30:
                break

        rows.append(
            {
                "query_id": str(q_id),
                "query": chosen_q,
                "positive_passages": [std_pos_doc],
                "negative_passages": std_neg_docs,
                "only_instruction": None,
                "only_query": chosen_q,
                "has_instruction": False,
                "new_negatives": [],
                "is_repeated": is_repeated,
            }
        )

        # Instruct query row
        inst_pos_info = record["final_positive"]
        inst_pos_doc = {
            "docid": str(inst_pos_info["id"]),
            "text": inst_pos_info["text"],
            "title": inst_pos_info.get("title", ""),
        }

        inst_neg_docs = []
        valid_synt_negs = []

        for sn in record.get("valid_synthetic_negatives", []):
            entry = {
                "docid": str(sn["id"]),
                "text": sn["text"],
                "title": sn.get("title", ""),
                "explanation": "instruction_negative",
            }
            inst_neg_docs.append(entry)
            valid_synt_negs.append(entry)

        for cand in inst_negs_batch[idx]:
            if len(inst_neg_docs) >= 30:
                break
            if cand["docid"] != inst_pos_doc["docid"]:
                inst_neg_docs.append(cand)

        rows.append(
            {
                "query_id": f"{q_id}-instruct",
                "query": f"{chosen_q} {instruct}".strip(),
                "positive_passages": [inst_pos_doc],
                "negative_passages": inst_neg_docs,
                "only_instruction": instruct,
                "only_query": chosen_q,
                "has_instruction": True,
                "new_negatives": valid_synt_negs,
                "is_repeated": is_repeated,
            }
        )

    return rows


def save_sharded_parquet(df, output_dir, split_name, chunk_size=150000):
    """Saves DataFrame into multiple parquet shards suitable for Hugging Face datasets."""
    split_dir = os.path.join(output_dir, "data")
    os.makedirs(split_dir, exist_ok=True)
    
    total_rows = len(df)
    if total_rows == 0:
        return []
        
    num_shards = max(1, math.ceil(total_rows / chunk_size))
    
    saved_paths = []
    for i in range(num_shards):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        shard_df = df.iloc[start_idx:end_idx]
        
        shard_name = f"{split_name}-{i:05d}-of-{num_shards:05d}.parquet"
        shard_path = os.path.join(split_dir, shard_name)
        
        shard_df.to_parquet(shard_path, engine="pyarrow")
        saved_paths.append(shard_path)
        
    return saved_paths


def write_hf_repo_files(output_dir):
    """Writes README.md and .gitattributes to the dataset root."""
    gitattr_path = os.path.join(output_dir, ".gitattributes")
    with open(gitattr_path, "w", encoding="utf-8") as f:
        f.write("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
        
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# Promptriever Dataset\n\n")
        f.write("This directory contains a generated promptriever dataset, fully compatible with the Hugging Face `datasets` library structure.\n\n")
        
        f.write("## Data Generation Script (`build_dataset.py`)\n\n")
        
        f.write("The script `build_dataset.py` constructs a final dataset by processing raw JSONL chunks generated by the upstream pipeline and running BM25 hard-negative mining on them. ")
        f.write("It automatically blends original (`original_query`) and synthetic (`rewritten_query`) targets at a 50/50 ratio to limit the impact of OOD artifacts or poor translation strings.\n\n")
        
        f.write("### Usage parameters:\n")
        f.write("- `--filtered_dir`: Path to the directory containing JSONL chunks.\n")
        f.write("- `--output_dir`: Path to save the HuggingFace-compatible dataset output.\n")
        f.write("- `--raw`: Read raw and normalize on the fly (useful for testing directly with raw chunks).\n")
        f.write("- `--val_size`: Number of unique queries allocated to validation.\n")
        f.write("- `--test_size`: Number of unique queries allocated to test.\n")
        f.write("- `--seed`: Random seed controlling reproducibility.\n")
        f.write("- `--chunk_size`: Number of rows per parquet shard (approx 256MB size). Output chunks have `-00000-of-00001` format.\\n\\n")

        f.write("## Structure\\n")
        f.write("- `data/`: Parquet files defining the training, validation, and test sets.\\n")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_dir", required=True)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the dataset repository in HF format",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Read raw (unfiltered) data and normalize it on the fly",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="Number of unique query_ids for validation (default: 1000)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=5000,
        help="Number of unique query_ids for test (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=150000,
        help="Number of rows per parquet shard (approx 256MB)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = get_jsonl_files(args.filtered_dir)

    # ========================
    # Stage 1: Read all records
    # ========================
    print("Reading all records...")
    all_records = []
    skipped = 0

    for file_path in tqdm(input_files, desc="Reading files"):
        for record in read_jsonl(file_path):
            if args.raw:
                if record.get("status") != "success":
                    skipped += 1
                    continue
                record = normalize_raw_record(record)
            all_records.append(record)

    if args.raw:
        print(
            f"Raw mode: kept {len(all_records)} records, skipped {skipped} (non-success)"
        )

    # ========================
    # Stage 2: Group and Split by query_id
    # ========================
    print("Grouping and assigning is_repeated flags...")
    
    qid_to_records = {}
    for r in all_records:
        qid_to_records.setdefault(r["query_id"], []).append(r)
        
    rng = random.Random(args.seed)
    
    unique_qids_list = sorted(qid_to_records.keys())
    print(f"  Unique query_ids: {len(unique_qids_list)}")
    
    rng.shuffle(unique_qids_list)
    
    val_qids = set(unique_qids_list[: args.val_size])
    test_qids = set(unique_qids_list[args.val_size : args.val_size + args.test_size])
    train_qids = set(unique_qids_list[args.val_size + args.test_size :])

    # Assign is_repeated
    for qid, recs in qid_to_records.items():
        chosen_idx = rng.randrange(len(recs))
        for i, r in enumerate(recs):
            r["is_repeated"] = (i != chosen_idx)
            
    # Partition records
    train_records = [r for r in all_records if r["query_id"] in train_qids]
    val_records = [r for r in all_records if r["query_id"] in val_qids]
    test_records = [r for r in all_records if r["query_id"] in test_qids]

    print(f"  Train records: {len(train_records)} ({len(train_qids)} distinct query_ids)")
    print(f"  Val records:   {len(val_records)} ({len(val_qids)} distinct query_ids)")
    print(f"  Test records:  {len(test_records)} ({len(test_qids)} distinct query_ids)")

    # ========================
    # Stage 3: Build val/test (no BM25)
    # ========================
    print("\nBuilding val/test splits...")

    val_doc_pool = collect_split_doc_pool(val_records)
    test_doc_pool = collect_split_doc_pool(test_records)

    print(f"  Val document pool: {len(val_doc_pool)} docs")
    print(f"  Test document pool: {len(test_doc_pool)} docs")

    val_rows = build_eval_rows(val_records, val_doc_pool, rng)
    test_rows = build_eval_rows(test_records, test_doc_pool, rng)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    # ========================
    # Stage 4: Build train (with BM25)
    # ========================
    print("\nBuilding train split with BM25 negatives...")

    # Build doc registry from train records only
    doc_registry = {}
    for record in train_records:
        for key in ["rewritten_original_positive", "rewritten_original_negative"]:
            item = record.get(key)
            if item and item.get("text"):
                doc_id = str(item["id"])
                if doc_id not in doc_registry:
                    doc_registry[doc_id] = {
                        "docid": doc_id,
                        "text": item["text"],
                        "title": item.get("title", ""),
                    }

    bm25_corpus_docs = list(doc_registry.values())
    print(f"  Train corpus size: {len(bm25_corpus_docs)}")

    retriever = BM25Retriever()
    retriever.index(bm25_corpus_docs)

    std_queries = [r["original_query"] for r in train_records]
    inst_queries = [
        f"{r.get('rewritten_query', r['original_query'])} {r['instruction']}".strip()
        for r in train_records
    ]

    print("  Mining BM25 negatives...")
    std_negs_batch = retriever.search(std_queries, k=35)
    inst_negs_batch = retriever.search(inst_queries, k=35)

    train_rows = build_train_rows(
        train_records, doc_registry, std_negs_batch, inst_negs_batch, rng
    )

    train_df = pd.DataFrame(train_rows)

    # ========================
    # Stage 5: Sharding & export
    # ========================
    print("\\nSaving datasets...")
    
    print("  -> Exporting Final Dataset")
    save_sharded_parquet(train_df, args.output_dir, "train", args.chunk_size)
    save_sharded_parquet(val_df, args.output_dir, "val", args.chunk_size)
    save_sharded_parquet(test_df, args.output_dir, "test", args.chunk_size)
    write_hf_repo_files(args.output_dir)

    # Save document pools as separate parquet files (useful for indexing)
    print("  -> Exporting Document Pools")
    pd.DataFrame(list(val_doc_pool.values())).to_parquet(
        os.path.join(args.output_dir, "val_corpus.parquet"), engine="pyarrow"
    )
    pd.DataFrame(list(test_doc_pool.values())).to_parquet(
        os.path.join(args.output_dir, "test_corpus.parquet"), engine="pyarrow"
    )



    # ========================
    # Summary
    # ========================
    print("\n=== Summary ===")
    print(f"Train: {len(train_df)} rows ({len(train_records)} distinct queries)")
    print(
        f"Val:   {len(val_df)} rows ({len(val_records)} distinct queries, {len(val_doc_pool)} docs in pool)"
    )
    print(
        f"Test:  {len(test_df)} rows ({len(test_records)} distinct queries, {len(test_doc_pool)} docs in pool)"
    )
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} rows generated (incl. instruct variations).")


if __name__ == "__main__":
    main()
