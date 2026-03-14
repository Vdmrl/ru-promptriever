import os
import json
import glob
from tqdm import tqdm

def main():
    # Pathing relative to the script's location (data_preprocessing/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    filtered_dir = os.path.join(current_dir, "data", "output_filtered")
    
    # Paths to the large files from which we extract
    input_triples_paths = [
        os.path.join(parent_dir, "data_generation", "data", "input", "triples.train.ids.small.tsv"),
        os.path.join(parent_dir, "data_generation", "data", "input", "triples.new_unique_aug.tsv")
    ]
    # Where to save the output file
    output_triples_path = os.path.join(parent_dir, "data_generation", "data", "input", "triples.train.ids.filtered.tsv")
    
    deleted_queries_file = os.path.join(filtered_dir, "deleted_queries.jsonl")
    
    # 1. Collect all query_ids that have been successfully generated & filtered
    successful_queries = set()
    filtered_files = glob.glob(os.path.join(filtered_dir, "run_paraphrasing_*_filtered.jsonl"))
    print(f"Found {len(filtered_files)} filtered files. Collecting successful queries...")
    for ff in filtered_files:
        with open(ff, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # Valid negative exists -> query successfully generated
                successful_queries.add(str(data['query_id']))
    
    # 2. Collect all query_ids that failed filtering
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

    # 3. Compute missing queries (those that were deleted AND not recovered in another run)
    missing_queries = deleted_queries - successful_queries
    
    print(f"Total deleted queries: {len(deleted_queries)}")
    print(f"Total successful queries: {len(successful_queries)}")
    print(f"Total queries to re-run (deleted but not successful in any other run): {len(missing_queries)}")
    
    if not missing_queries:
        print("No queries left to re-run! Exiting.")
        return

    # 4. Extract exactly one triplet per missing query
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
                            # Remove from missing_queries so we only extract it ONCE
                            missing_queries.remove(qid)
                            written_queries.add(qid)
                            
                    # Early exit if we have found all needed queries
                    if not missing_queries:
                        print("Found all missing queries! Stopping extraction early.")
                        break

    print(f"\nSuccessfully wrote {len(written_queries)} triplets to {output_triples_path}")
    if missing_queries:
        print(f"Warning: Could not find triplets for {len(missing_queries)} queries in the provided files.")

if __name__ == "__main__":
    main()
