"""
Data utilities for Promptriever training.

RetrieverDataset  — loads parquet, returns (query, positive, negatives) per row.
RetrieverCollator — tokenizes and flattens passages into a single batch for GradCache.
"""

import random
from dataclasses import dataclass
from typing import List

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class RetrieverDataset(Dataset):
    """
    PyTorch Dataset for bi-encoder retriever training from a parquet file.

    Each row is expected to contain:
      - positive_passages  (list[dict])  — relevant passages ({docid, text, title})
      - negative_passages  (list[dict])  — hard negatives + instruction negatives
      - has_instruction    (bool)        — whether the row carries an instruction
      - only_query         (str)         — the raw query without instruction
      - only_instruction   (str | None)  — the instruction text (if any)

    For instruction-augmented rows the query is reassembled as
        "{only_query} {only_instruction}"
    following the original Promptriever paper convention (instruction appended).
    """

    def __init__(
        self,
        data_path: str,
        num_negatives: int = 7,
        seed: int = 42,
    ):
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

        # Resolve hf:// URIs to local cached paths via hf_hub_download.
        # Format: hf://datasets/<user>/<repo>/<filepath>
        local_path = self._resolve_data_path(data_path)

        # HF Datasets uses memory-mapping (Apache Arrow) — ~0 RAM overhead.
        self.dataset = load_dataset(
            "parquet", data_files={"train": local_path}, split="train"
        )

    @staticmethod
    def _resolve_data_path(data_path: str) -> str:
        """Download from HF Hub if hf:// URI, otherwise return as-is."""
        if data_path.startswith("hf://datasets/"):
            # hf://datasets/Vladimirlv/repo-name/path/to/file.parquet
            remainder = data_path[len("hf://datasets/") :]
            parts = remainder.split("/", 2)  # [user, repo, filepath]
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid hf:// URI: {data_path}. "
                    f"Expected hf://datasets/<user>/<repo>/<filepath>"
                )
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = parts[2]
            print(f"[data] Downloading {filename} from {repo_id}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
            )
            print(f"[data] Cached at: {local_path}")
            return local_path
        return data_path

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _format_passage(doc: dict) -> str:
        """Concatenate title and body of a passage document."""
        title = doc.get("title", "")
        text = doc.get("text", "")
        if title:
            return f"{title}. {text}"
        return text

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]

        # --- Query ---
        # Reassemble in Promptriever format: {query} {instruction}
        if row.get("has_instruction") and row.get("only_instruction"):
            query = f"{row['only_query']} {row['only_instruction']}"
        else:
            query = row["only_query"]

        # --- Positive passage ---
        pos_list = row.get("positive_passages", [])
        if pos_list is not None and len(pos_list) > 0:
            positive = self._format_passage(pos_list[0])
        else:
            positive = ""

        # --- Negative passages ---
        neg_list = row.get("negative_passages", [])
        if neg_list is None:
            neg_list = []
        negatives = [self._format_passage(n) for n in neg_list]

        # Truncate or pad to a fixed number of negatives
        if len(negatives) > self.num_negatives:
            negatives = self.rng.sample(negatives, self.num_negatives)
        while len(negatives) < self.num_negatives:
            negatives.append("")

        return {
            "query": query,
            "positive": positive,
            "negatives": negatives,
        }


@dataclass
class RetrieverCollator:
    """
    Collator for bi-encoder retriever batches.

    Tokenizes queries and passages separately.
    Passages are flattened into a single list per batch:
      [Pos_1, Neg_1_1, ..., Neg_1_N, Pos_2, Neg_2_1, ..., Neg_2_N, ...]

    Returns a dict with:
      - queries:       {input_ids, attention_mask}
      - passages:      {input_ids, attention_mask}
      - num_negatives: int  (used by the loss function)
    """

    tokenizer: PreTrainedTokenizerBase
    max_len_query: int = 304
    max_len_passage: int = 256

    def __call__(self, batch: List[dict]) -> dict:
        queries = [item["query"] for item in batch]

        passages = []
        for item in batch:
            passages.append(item["positive"])
            passages.extend(item["negatives"])

        q_tokens = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_len_query,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        p_tokens = self.tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_len_passage,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        num_negatives = len(batch[0]["negatives"]) if batch else 0

        return {
            "queries": q_tokens,
            "passages": p_tokens,
            "num_negatives": num_negatives,
        }
