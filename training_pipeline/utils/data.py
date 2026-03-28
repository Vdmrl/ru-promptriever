"""
Data utilities for Promptriever training.

RetrieverDataset  — loads parquet, returns (query, positive, negatives) per row.
RetrieverCollator — tokenizes and flattens passages into a single batch for GradCache.
"""

import os
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
        split: str = "train",
        num_negatives: int = 7,
        num_instruct_negatives: int = 3,
        instruct_only: bool = False,
        use_repeated: bool = False,
        seed: int = 42,
    ):
        self.num_negatives = num_negatives
        self.num_instruct_negatives = num_instruct_negatives
        self.num_hard_negatives = num_negatives - num_instruct_negatives
        self.rng = random.Random(seed)

        # Load data: support HF repo IDs, hf:// URIs, and local file paths.
        self.dataset = self._load_data(data_path, split)

        # Filter deduplicates by default (unless --use_repeated is passed and is_repeated exists in data)
        if not use_repeated and "is_repeated" in self.dataset.column_names:
            before_repeats = len(self.dataset)
            self.dataset = self.dataset.filter(lambda x: not x["is_repeated"])
            print(f"[data] Removed repeated queries: {before_repeats} → {len(self.dataset)} rows")

        # Filter to instruction-augmented rows only (halves the dataset).
        if instruct_only:
            before = len(self.dataset)
            self.dataset = self.dataset.filter(lambda x: x.get("has_instruction", False))
            print(f"[data] instruct_only filter: {before} → {len(self.dataset)} rows")

    @staticmethod
    def _load_data(data_path: str, split: str = "train"):
        """Load dataset from HF repo ID, hf:// URI, or local file path."""
        # 1. Direct HF repo ID (e.g. "Vladimirlv/ru-promptriever-dataset")
        if (
            "/" in data_path
            and not data_path.startswith("hf://")
            and not os.path.exists(data_path)
        ):
            print(f"[data] Loading HF dataset '{data_path}' split='{split}'...")
            return load_dataset(data_path, split=split)

        # 2. Legacy hf:// URI (e.g. "hf://datasets/user/repo/file.parquet")
        if data_path.startswith("hf://datasets/"):
            remainder = data_path[len("hf://datasets/"):]
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
            data_path = local_path

        # 3. Local file path
        return load_dataset(
            "parquet", data_files={"train": data_path}, split="train"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _format_passage(doc) -> str:
        """Concatenate title and body if dict, or return directly if string."""
        if isinstance(doc, str):
            return doc
        title = doc.get("title", "")
        text = doc.get("text", "")
        if title:
            return f"{title}. {text}"
        return text

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]

        # --- Query ---
        # Read pre-assembled query if available (the new approach), else fallback to old concatenation
        if "query" in row:
             query = row["query"]
        else:
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

        # --- Negative passages (typed sampling) ---
        neg_list = row.get("negative_passages", [])
        if neg_list is None:
            neg_list = []

        # negative_passages exclusively contains hard negatives
        hard_negs = [self._format_passage(n) for n in neg_list]

        # Instruction negatives (new_negatives) are ONLY used for rows WITH instruction.
        # For has_instruction=false rows, instruct-negatives are query-positive
        # (they were designed to be relevant to the query but violate the instruction).
        # Using them as negatives without instruction would corrupt training.
        has_instruction = row.get("has_instruction", False)

        if has_instruction:
            instruct_negs_raw = row.get("new_negatives", []) or []
            instruct_negs = [self._format_passage(n) for n in instruct_negs_raw]
            n_inst = min(self.num_instruct_negatives, len(instruct_negs))
            sampled_inst = self.rng.sample(instruct_negs, n_inst) if n_inst > 0 else []
        else:
            instruct_negs = []
            sampled_inst = []

        # Fill the rest from hard negatives
        n_hard_needed = self.num_negatives - len(sampled_inst)
        n_hard = min(n_hard_needed, len(hard_negs))
        sampled_hard = self.rng.sample(hard_negs, n_hard) if n_hard > 0 else []

        # If still short, fill from remaining hard negatives
        deficit = self.num_negatives - len(sampled_inst) - len(sampled_hard)
        if deficit > 0 and has_instruction:
            remaining_inst = [x for x in instruct_negs if x not in sampled_inst]
            remaining_hard = [x for x in hard_negs if x not in sampled_hard]
            remaining = remaining_hard + remaining_inst
            fill = remaining[:deficit]
            sampled_hard.extend(fill)
        elif deficit > 0:
            remaining_hard = [x for x in hard_negs if x not in sampled_hard]
            fill = remaining_hard[:deficit]
            sampled_hard.extend(fill)

        negatives = sampled_inst + sampled_hard

        # Pad if still short
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
