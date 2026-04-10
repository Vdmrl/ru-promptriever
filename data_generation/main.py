import os
import time
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils.data_loader import DataLoader
from utils.processor import Processor
from utils.scheduler import get_allowed_threads

# Shared lock to serialize writes across worker threads.
file_write_lock = threading.Lock()


def save_result(result, output_file):
    json_str = json.dumps(result, ensure_ascii=False)
    with file_write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json_str + "\n")


def main():
    parser = argparse.ArgumentParser(description="GigaChat Promptriever Generator")
    parser.add_argument("--limit", type=int, default=100, help="docs to process")
    parser.add_argument("--offset", type=int, default=0, help="skip n docs")
    parser.add_argument("--output", type=str, default="data/output/generated_data.jsonl", help="output path")
    parser.add_argument("--config", type=str, default="config.yaml", help="config path")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_INPUT_DIR = os.path.join(BASE_DIR, "data", "input")

    loader = DataLoader(
        collection_path=os.path.join(DATA_INPUT_DIR, "russian_collection.tsv"),
        queries_path=os.path.join(DATA_INPUT_DIR, "russian_queries.train.tsv"),
        triples_path=os.path.join(DATA_INPUT_DIR, "triples.train.ids.small.tsv")
    )
    loader.load_memory()

    processor = Processor(config_path=args.config)

    # Upper bound matches the weekend-mode cap from the scheduler (9 threads).
    executor = ThreadPoolExecutor(max_workers=9)
    futures = []

    data_gen = loader.yield_triples(offset=args.offset, limit=args.limit)

    processed_count = 0
    print(f"\n[Main] Starting Pipeline. Target: {args.limit} docs. Offset: {args.offset}")
    print("[Main] Press Ctrl+C to stop gracefully.\n")

    pbar = tqdm(total=args.limit, desc="Processing", unit="doc")

    try:
        while processed_count < args.limit:
            # Enforce the time-based thread cap (lower during Moscow business hours).
            allowed_threads = get_allowed_threads()

            # Prune completed futures to prevent unbounded list growth.
            active_futures = [f for f in futures if not f.done()]
            futures = active_futures
            active_count = len(futures)

            # Submit new tasks up to the allowed concurrency ceiling.
            while active_count < allowed_threads and processed_count < args.limit:
                try:
                    sample = next(data_gen)


                    future = executor.submit(processor.process_sample, sample)

                    def callback(fut):
                        try:
                            res = fut.result()
                            save_result(res, args.output)

                            if res['status'] != 'success':
                                # tqdm.write() is thread-safe and preserves the progress bar.
                                tqdm.write(f"[ERR] QID: {res.get('query_id')} -> {res.get('error')}")

                        except Exception as e:
                            tqdm.write(f"[FATAL] System error in callback: {e}")
                        finally:
                            pbar.update(1)

                    future.add_done_callback(callback)

                    futures.append(future)
                    active_count += 1
                    processed_count += 1

                except StopIteration:
                    tqdm.write("[Main] Data source exhausted (EOF).")
                    processed_count = args.limit
                    break

            pbar.set_description(f"Threads: {active_count}/{allowed_threads}")
            time.sleep(0.05)

    except KeyboardInterrupt:
        tqdm.write("\n[Main] Interrupted by user. Waiting for active tasks to finish...")

    pbar.close()
    executor.shutdown(wait=True)
    print("[Main] Pipeline finished.")


if __name__ == "__main__":
    main()