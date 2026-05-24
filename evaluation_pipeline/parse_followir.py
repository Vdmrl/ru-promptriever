import os
import json
import glob
import pandas as pd

def find_scores(data, target_key):
    """Recursively search for a key in a nested dictionary/list."""
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]
        for k, v in data.items():
            result = find_scores(v, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_scores(item, target_key)
            if result is not None:
                return result
    return None

def main():
    results_dir = "./results_followir_eng"
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return
    
    files = glob.glob(os.path.join(results_dir, "*.json"))
    if not files:
        print(f"No JSON files found in {results_dir}")
        return

    records = []
    
    for file in files:
        model_name = os.path.basename(file).split("__")[0]
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # The structure is usually data["mteb"] -> list of task results
            mteb_data = data.get("mteb", [])
            if not mteb_data:
                # Sometimes it might just be the raw dict
                mteb_data = [data] if "scores" in data else data

            for task_data in mteb_data:
                task_name = task_data.get("task_name", "UnknownTask")
                if "Robust04" in task_name:
                    short_name = "Robust04"
                elif "Core17" in task_name:
                    short_name = "Core17"
                elif "News21" in task_name:
                    short_name = "News21"
                else:
                    # In case the task name is hidden deeper
                    task_str = str(task_data)
                    if "Robust04InstructionRetrieval" in task_str: short_name = "Robust04"
                    elif "Core17InstructionRetrieval" in task_str: short_name = "Core17"
                    elif "News21InstructionRetrieval" in task_str: short_name = "News21"
                    else: continue

                # Recursively extract the 'og' dict and 'p-MRR'
                og_scores = find_scores(task_data, "og")
                p_mrr = find_scores(task_data, "p-MRR")
                
                # Sometimes it's lowercase
                if p_mrr is None:
                    p_mrr = find_scores(task_data, "p_mrr")
                
                if og_scores is not None and p_mrr is not None:
                    if short_name == "Robust04" or short_name == "Core17":
                        # MAP at 1000
                        base_metric = og_scores.get("map_at_1000", 0.0) * 100
                    else: # News21
                        # nDCG at 5
                        base_metric = og_scores.get("ndcg_at_5", 0.0) * 100
                    
                    records.append({
                        "Model": model_name,
                        "Task": short_name,
                        "Base Metric (MAP/nDCG)": round(base_metric, 2),
                        "p-MRR": round(p_mrr * 100, 2)
                    })
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    if not records:
        print("No valid FollowIR results found in the JSON files.")
        return

    df = pd.DataFrame(records)
    
    # Pivot table for better viewing: Models as rows, Tasks as columns
    pivot_df = df.pivot(index="Model", columns="Task")
    
    # Flatten multi-index columns
    pivot_df.columns = [f"{col[1]} {col[0]}" for col in pivot_df.columns]
    
    # Sort columns logically
    cols = []
    for task in ["Robust04", "Core17", "News21"]:
        cols.extend([f"{task} Base Metric (MAP/nDCG)", f"{task} p-MRR"])
    
    # Only keep columns that actually exist
    cols = [c for c in cols if c in pivot_df.columns]
    pivot_df = pivot_df[cols]
    
    print("\n" + "="*80)
    print("FOLLOWIR ENGLISH RESULTS".center(80))
    print("="*80)
    print(pivot_df.to_markdown())
    print("="*80)

if __name__ == "__main__":
    main()
