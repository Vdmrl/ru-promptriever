import os
import sys
import gc
import csv

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)


class DataLoader:
    def __init__(self, collection_path: str, queries_path: str, triples_path: str):
        self.collection_path = collection_path
        self.queries_path = queries_path
        self.triples_path = triples_path

        # In-memory lookup tables populated by load_memory().
        self.collection: dict[int, str] = {}  # passage_id -> passage_text
        self.queries: dict[int, str] = {}     # query_id   -> query_text

    def load_memory(self):
        """Load queries and the document collection entirely into RAM."""
        print(f"[Loader] Loading QUERIES from {os.path.basename(self.queries_path)}...")
        with open(self.queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.queries[int(parts[0])] = parts[1]

        print(f"[Loader] Loading COLLECTION from {os.path.basename(self.collection_path)}...")
        print("[Loader] WARNING: This is the memory-intensive step.")

        try:
            with open(self.collection_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.collection[int(parts[0])] = parts[1]
        except MemoryError:
            print("[CRITICAL] Out of Memory while loading collection.")
            sys.exit(1)

        print(f"[Loader] Done. Loaded {len(self.collection)} docs and {len(self.queries)} queries.")

        # Release any unused memory after the bulk load.
        gc.collect()

    def yield_triples(self, offset=0, limit=None):
        """Stream triples from disk, resolving IDs to text from memory.

        Deduplicates on query_id so each query appears at most once. The
        ``offset`` and ``limit`` parameters operate on the deduplicated stream,
        enabling independent parallel workers to cover disjoint query ranges.
        """
        unique_queries_count = 0
        yielded = 0
        seen_queries = set()

        print(f"[Loader] Reading triples from {os.path.basename(self.triples_path)}...")

        with open(self.triples_path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit is not None and yielded >= limit:
                    break

                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                try:
                    qid = int(parts[0])

                    # Skip duplicate appearances of the same query_id.
                    if qid in seen_queries:
                        continue

                    seen_queries.add(qid)

                    # Offset is counted against unique query positions, not raw line numbers.
                    if unique_queries_count < offset:
                        unique_queries_count += 1
                        continue

                    pid = int(parts[1])
                    nid = int(parts[2])

                    if qid in self.queries and pid in self.collection and nid in self.collection:
                        yield (
                            qid,
                            self.queries[qid],
                            pid,
                            self.collection[pid],
                            nid,
                            self.collection[nid]
                        )
                        yielded += 1
                        unique_queries_count += 1

                except ValueError:
                    continue