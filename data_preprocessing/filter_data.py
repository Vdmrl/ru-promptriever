import sys
import os
import time
import json
import argparse
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.processor import FilterProcessor
from utils.scheduler import get_allowed_threads
from utils.io import read_jsonl, get_jsonl_files

# Shared lock to serialize writes across thread-pool callback threads.
file_write_lock = threading.Lock()
stats_lock = threading.Lock()


def append_line(obj, path):
    """Append a single JSON object as a line to a file (thread-safe)."""
    line = json.dumps(obj, ensure_ascii=False)
    with file_write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based data filtering for Promptriever"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/input",
        help="Directory with input JSONL files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output_filtered",
        help="Directory for filtered JSONL output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to LLM config YAML",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable short reasoning before DA/NET answer (slower but more accurate)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to process per file (for testing)",
    )
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p

    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Shared log files aggregated across all processed input chunks.
    deleted_queries_path = os.path.join(output_dir, "deleted_queries.jsonl")
    deleted_negatives_path = os.path.join(output_dir, "deleted_negatives.jsonl")

    processor = FilterProcessor(config_path=args.config, use_reasoning=args.reasoning)

    input_files = get_jsonl_files(input_dir)
    if not input_files:
        print(f"[Main] No JSONL files found in {input_dir}")
        return

    print(f"[Main] Found {len(input_files)} JSONL file(s) to process")
    print(f"[Main] Reasoning mode: {'ON' if args.reasoning else 'OFF'}")
    print(f"[Main] Deleted-queries log : {deleted_queries_path}")
    print(f"[Main] Deleted-negatives log: {deleted_negatives_path}")

    for in_file in input_files:
        base_name = os.path.splitext(os.path.basename(in_file))[0]
        # Mirror input filename with a _filtered suffix.
        out_file = os.path.join(output_dir, f"{base_name}_filtered.jsonl")

        # Idempotency guard: skip this file if it was fully processed in a prior run.
        done_marker = out_file + ".done"
        if os.path.exists(done_marker) and os.path.exists(out_file):
            print(f"[Main] Skipping completed file: {base_name}")
            continue

        # Count total lines upfront for the progress bar denominator.
        with open(in_file, "r", encoding="utf-8") as f:
            total_records = sum(1 for _ in f)
        print(f"\n[Main] Processing {base_name}.jsonl ({total_records} records)")

        # Persist the number of fully completed records so later invocations
        # can skip already-processed entries without reprocessing the entire file.
        offset_file = out_file + ".offset"
        offset = 0
        if os.path.exists(offset_file):
            with open(offset_file, "r") as f:
                try:
                    offset = int(f.read().strip())
                except ValueError:
                    pass

        # Mutable counters accessed from callback closures; using a list
        # allows mutation from inner functions without `nonlocal`.
        stats = {
            "kept": 0,
            "discarded": 0,
            "failed": 0,
            "skipped_status": 0,
        }
        completed_count = [0]  # Wrapped in a list to permit mutation from closures.

        executor = ThreadPoolExecutor(max_workers=9)
        futures = []

        reader = read_jsonl(in_file)
        if offset > 0:
            print(f"[Main] Resuming from record {offset}...")
            reader = itertools.islice(reader, offset, None)

        effective_total = total_records - offset
        if args.limit is not None:
            effective_total = min(effective_total, args.limit)

        submitted_count = 0
        pbar = tqdm(total=effective_total, desc="Filtering", unit="rec")

        try:
            for record in reader:
                if args.limit is not None and submitted_count >= args.limit:
                    break

                # Records that failed during generation can be skipped immediately
                # without an LLM call — the filter has nothing to check.
                if record.get("status") != "success":
                    with stats_lock:
                        stats["skipped_status"] += 1
                        completed_count[0] += 1
                    with open(offset_file, "w") as f_off:
                        f_off.write(str(offset + submitted_count + 1))
                    submitted_count += 1
                    pbar.update(1)
                    continue

                # Block submission until the active thread count drops below the
                # scheduler's time-based cap.
                while True:
                    allowed_threads = get_allowed_threads()
                    active_futures = [fut for fut in futures if not fut.done()]
                    futures = active_futures
                    if len(futures) < allowed_threads:
                        break
                    time.sleep(0.1)

                # Capture the loop variable in a new scope so each closure
                # references an independent snapshot rather than the shared variable.
                snap_submitted = submitted_count

                def make_callback(snap):
                    def callback(fut):
                        try:
                            result = fut.result()
                            status = result.get("_filter_status", "failed")

                            if status == "kept":
                                # Remove internal bookkeeping fields before persisting.
                                filtered_out = result.pop("_filtered_out_negatives", [])
                                result.pop("_filter_status", None)

                                # Append the accepted record to the output chunk.
                                append_line(result, out_file)

                                # Log negatives that were filtered out for analysis.
                                for neg in filtered_out:
                                    append_line(neg, deleted_negatives_path)

                                with stats_lock:
                                    stats["kept"] += 1

                            elif status == "discarded":
                                # Persist the full discarded record (including checked
                                # positives and reasoning) for post-hoc analysis.
                                result.pop("_filter_status", None)
                                append_line(result, deleted_queries_path)
                                with stats_lock:
                                    stats["discarded"] += 1

                            else:  # failed
                                tqdm.write(
                                    f"[ERR] QID: {result.get('query_id')} "
                                    f"-> {result.get('error')}"
                                )
                                with stats_lock:
                                    stats["failed"] += 1

                        except Exception as e:
                            tqdm.write(f"[FATAL] callback snap={snap}: {e}")
                            with stats_lock:
                                stats["failed"] += 1
                        finally:
                            with stats_lock:
                                completed_count[0] += 1
                            with open(offset_file, "w") as f_off:
                                f_off.write(str(offset + completed_count[0]))
                            pbar.update(1)

                    return callback

                future = executor.submit(processor.process_sample, record)
                future.add_done_callback(make_callback(snap_submitted))
                futures.append(future)
                submitted_count += 1

                pbar.set_description(
                    f"Active: {len([f for f in futures if not f.done()])}/{allowed_threads}"
                )

        except KeyboardInterrupt:
            tqdm.write("\n[Main] Interrupted. Waiting for active tasks...")

        pbar.close()
        executor.shutdown(wait=True)

        # Persist final offset
        with open(offset_file, "w") as f_off:
            f_off.write(str(offset + submitted_count))

        print(
            f"\n[Main] {base_name} — "
            f"kept: {stats['kept']}, "
            f"discarded: {stats['discarded']}, "
            f"failed: {stats['failed']}, "
            f"skipped (non-success): {stats['skipped_status']}"
        )

        # Mark the file complete if all input records have been submitted.
        remaining = total_records - (offset + submitted_count)
        if remaining <= 0 or (args.limit is not None and submitted_count >= args.limit):
            if os.path.exists(offset_file):
                os.remove(offset_file)
            with open(done_marker, "w") as f:
                f.write("done")
            print(f"[Main] Completed: {out_file}")
        else:
            print(f"[Main] Partial ({remaining} records remain). Re-run to resume.")

    print("\n[Main] Pipeline finished.")


if __name__ == "__main__":
    main()
