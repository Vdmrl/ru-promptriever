import json
import wandb
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, required=True, help="Path to trainer_state.json")
    parser.add_argument("--project", type=str, default="ru-promptriever", help="Wandb project name")
    parser.add_argument("--name", type=str, required=True, help="Wandb run name")
    args = parser.parse_args()

    if not os.path.exists(args.state_path):
        print(f"Error: {args.state_path} not found.")
        return

    with open(args.state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        print("No log history found in trainer_state.json")
        return

    print(f"Initializing wandb run: {args.name}")
    wandb.init(project=args.project, name=args.name)

    print(f"Uploading {len(log_history)} log entries...")
    for log in log_history:
        # Filter out internal trainer keys if needed, but logging everything is usually fine
        step = log.pop("step", None)
        if step is not None:
            wandb.log(log, step=step)
        else:
            wandb.log(log)
            
    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    main()
