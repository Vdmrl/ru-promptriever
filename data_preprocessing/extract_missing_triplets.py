"""
extract_missing_triplets.py

Identifies query IDs that were deleted by the filter stage and have not been
successfully regenerated in any other pipeline run, then extracts the
corresponding triple lines from the source TSV files for re-processing.

Output: a new triples TSV containing exactly one entry per missing query, ready
to be passed back to the data_generation pipeline.
"""

import os
import json
import glob
from tqdm import tqdm

def main():
    # Paths are resolved relative to this script's location (data_preprocessing/).
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    filtered_dir = os.path.join(current_dir, "data", "output_filtered")
    
    # Source TSV files to scan for missing query triplets.
    input_triples_paths = [
        os.path.join(parent_dir, "data_generation", "data", "input", "triples.train.ids.small.tsv"),
        os.path.join(parent_dir, "data_generation", "data", "input", "triples.new_unique_aug.tsv")
    ]
    # Destination file for the extracted triplets.
    output_triples_path = os.path.join(parent_dir, "data_generation", "data", "input", "triples.train.ids.filtered.tsv")
    
    deleted_queries_file = os.path.join(filtered_dir, "deleted_queries.jsonl")
    
    # Step 1: Collect all query_ids that completed both generation and filtering.
    successful_queries = set()
    filtered_files = glob.glob(os.path.join(filtered_dir, "run_paraphrasing_*_filtered.jsonl"))
    print(f"Found {len(filtered_files)} filtered files. Collecting successful queries...")
    for ff in filtered_files:
        with open(ff, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # A record present in a filtered output file has at least one valid negative.
                successful_queries.add(str(data['query_id']))
    
    # Step 2: Collect all query_ids that were discarded by the filter.
    deleted_queries = set()
    if os.path.exists(deleted_queries_file):
        with open(deleted_queries_file, 'r', encoding='utf-8') as f:
            print(f"Reading deleted queries from {deleted_queries_file}...")
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                deleted_queries.add(str(data['query_id']))
    else:
        print(f"Warning: {deleted_queries_file} not found.")

    # Step 3: Queries that are deleted but never recovered in any subsequent run.
    missing_queries = deleted_queries - successful_queries
    
    print(f"Total deleted queries: {len(deleted_queries)}")
    print(f"Total successful queries: {len(successful_queries)}")
    print(f"Total queries to re-run (deleted but not successful in any other run): {len(missing_queries)}")
    
    if not missing_queries:
        print("No queries left to re-run! Exiting.")
        return

    # Step 4: Extract exactly one triplet per missing query (first occurrence wins).
    valid_input_paths = [p for p in input_triples_paths if os.path.exists(p)]
    if not valid_input_paths:
        print(f"\nERROR: No input triples files found. Checked:")
        for p in input_triples_paths:
            print(f" - {p}")
        return
        
    print(f"\nExtracting triplets for {len(missing_queries)} missing queries from {len(valid_input_paths)} files...")
    written_queries = set()
    
    with open(output_triples_path, 'w', encoding='utf-8') as fout:
        for file_path in valid_input_paths:
            if not missing_queries:
                break
                
            print(f"\nProcessing {os.path.basename(file_path)}...")
            # Count total lines in current file to show progress bar
            total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore'))
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fin:
                for line in tqdm(fin, total=total_lines, desc=f"Scanning"):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        qid = parts[0]
                        if qid in missing_queries:
                            fout.write(line)
                            # Remove the query so each ID is written only once.
                            missing_queries.remove(qid)
                            written_queries.add(qid)
                            
                    # Early-exit once all missing queries have been found.
                    if not missing_queries:
                        print("Found all missing queries! Stopping extraction early.")
                        break

    print(f"\nSuccessfully wrote {len(written_queries)} triplets to {output_triples_path}")
    if missing_queries:
        print(f"Warning: Could not find triplets for {len(missing_queries)} queries in the provided files.")

if __name__ == "__main__":
    main()
