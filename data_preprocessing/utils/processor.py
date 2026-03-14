import threading
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_core.messages import SystemMessage, HumanMessage

from utils.llm_init import create_llm_instance
from utils.prompts import get_positive_prompt, get_negative_prompt, get_system_prompt


class ParseError(ValueError):
    """Raised when the LLM response cannot be parsed as ДА/НЕТ."""


class FilterProcessor:
    def __init__(self, config_path: str, use_reasoning: bool = False):
        self.config_path = config_path
        self.use_reasoning = use_reasoning
        self._thread_local = threading.local()

        self.system_prompt = get_system_prompt(use_reasoning)
        self.positive_prompt_template = get_positive_prompt(use_reasoning)
        self.negative_prompt_template = get_negative_prompt(use_reasoning)

    def _get_llm(self):
        if not hasattr(self._thread_local, "llm"):
            self._thread_local.llm = create_llm_instance(config_path=self.config_path)
        return self._thread_local.llm

    @staticmethod
    def _parse_answer(text: str) -> tuple[bool, str]:
        """
        Extract DA/NET from LLM response.
        For reasoning mode the answer is expected on the last non-empty line;
        everything before it is the reasoning.
        Returns (answer: bool, reasoning: str).
        """
        raw = text.strip()
        cleaned = raw.upper()

        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        answer = None

        if lines:
            last_line = lines[-1].upper()
            if "ДА" in last_line and "НЕТ" not in last_line:
                answer = True
            elif "НЕТ" in last_line:
                answer = False

            if answer is not None:
                reasoning = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""
                return answer, reasoning

        # Fallback: scan full text
        if "ДА" in cleaned and "НЕТ" not in cleaned:
            return True, raw
        if "НЕТ" in cleaned:
            return False, raw

        raise ParseError(f"Cannot parse LLM response as ДА/НЕТ: {text!r:.200}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        retry=retry_if_exception_type((Exception, ParseError)),
        reraise=True,
    )
    def _check_relevance(
        self, llm, query, instruction, document, prompt_type="positive"
    ):
        """
        Send a relevance-check prompt to the LLM and return (answer, reasoning).
        prompt_type: "positive" checks if doc matches query+instruction;
                     "negative" checks if doc violates the instruction (correct hard-neg).
        """
        if prompt_type == "positive":
            user_text = self.positive_prompt_template.format(
                query=query, instruction=instruction, document=document
            )
        else:
            user_text = self.negative_prompt_template.format(
                query=query, instruction=instruction, document=document
            )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_text),
        ]
        response = llm.invoke(messages)
        return self._parse_answer(response.content)

    def process_sample(self, record):
        """
        Filter a single record. Returns a dict with a top-level '_filter_status' key:
          - '_filter_status': 'kept'       — record passed; saved to filtered output
          - '_filter_status': 'discarded'  — no valid positive found; whole query dropped
          - '_filter_status': 'failed'     — LLM error after retries

        Kept records also carry '_filtered_out_negatives': list of negatives that did NOT
        pass the check (for analysis / deleted_negatives log).

        Filtering logic:
          Stage 1 — Positive: check rewritten_pos_doc vs rewritten_query+instruction.
                    If fails, try synthetic positive (matches_both=true).
                    If neither passes, mark as 'discarded'.
          Stage 2 — Negatives: for each instruction negative (matches_both=false),
                    check it truly violates the instruction; keep only confirmed negatives.
        """
        try:
            llm = self._get_llm()
        except Exception as e:
            return {
                "_filter_status": "failed",
                "query_id": record.get("query_id"),
                "error": f"LLM Init: {e}",
            }

        idata = record["instruction_data"]
        mdata = record["mining_data"]
        query = idata["rewritten_query"]
        instruction = idata["instruction"]

        # Build a minimal snippet used for deleted_queries log
        query_meta = {
            "query_id": record["query_id"],
            "original_query": record["original_query"],
            "rewritten_query": query,
            "instruction": instruction,
        }

        # === Stage 1: Positive document check ===
        final_pos_doc = None
        final_pos_source = None
        final_pos_id = None
        final_pos_title = ""
        # For deleted_queries log: track what was checked and why it failed
        checked_positives = []

        try:
            pos_relevant, pos_reasoning = self._check_relevance(
                llm,
                query,
                instruction,
                idata["rewritten_pos_doc"],
                prompt_type="positive",
            )
            checked_positives.append(
                {
                    "source": "rewritten",
                    "title": idata.get("rewritten_pos_title", ""),
                    "text": idata["rewritten_pos_doc"],
                    "llm_answer": pos_relevant,
                    "llm_reasoning": pos_reasoning,
                }
            )

            if pos_relevant:
                final_pos_doc = idata["rewritten_pos_doc"]
                final_pos_source = "rewritten"
                final_pos_id = str(record["original_positive_id"])
                final_pos_title = idata.get("rewritten_pos_title", "")
            else:
                # Fallback: try synthetic positive (matches_both=true)
                synth_pos = next(
                    (d for d in mdata.get("documents", []) if d.get("matches_both")),
                    None,
                )
                if synth_pos:
                    synth_relevant, synth_reasoning = self._check_relevance(
                        llm,
                        query,
                        instruction,
                        synth_pos["passage"],
                        prompt_type="positive",
                    )
                    checked_positives.append(
                        {
                            "source": "synthetic",
                            "title": synth_pos.get("title", ""),
                            "text": synth_pos["passage"],
                            "llm_answer": synth_relevant,
                            "llm_reasoning": synth_reasoning,
                        }
                    )
                    if synth_relevant:
                        final_pos_doc = synth_pos["passage"]
                        final_pos_source = "synthetic"
                        final_pos_id = f"{record['query_id']}_synth"
                        final_pos_title = synth_pos.get("title", "")

            if final_pos_doc is None:
                return {
                    "_filter_status": "discarded",
                    **query_meta,
                    "checked_positives": checked_positives,
                }

        except Exception as e:
            err_str = str(e)
            if "Connection" in err_str or "Errno" in err_str:
                if hasattr(self._thread_local, "llm"):
                    del self._thread_local.llm
            return {
                "_filter_status": "failed",
                "query_id": record.get("query_id"),
                "error": f"Positive check: {err_str}",
            }

        # === Stage 2: Negative documents check ===
        valid_negs = []
        filtered_out_negs = []

        try:
            synth_negs = [
                d for d in mdata.get("documents", []) if not d.get("matches_both")
            ]

            for neg_idx, neg_doc in enumerate(synth_negs):
                is_correct_negative, reasoning = self._check_relevance(
                    llm, query, instruction, neg_doc["passage"], prompt_type="negative"
                )

                if is_correct_negative:
                    valid_negs.append(
                        {
                            "id": f"{record['query_id']}_{neg_idx}",
                            "text": neg_doc["passage"],
                            "title": neg_doc.get("title", ""),
                        }
                    )
                else:
                    # Record what was dropped for analysis
                    filtered_out_negs.append(
                        {
                            "query_id": record["query_id"],
                            "rewritten_query": query,
                            "instruction": instruction,
                            "neg_index": neg_idx,
                            "error_type": neg_doc.get("error_type", ""),
                            "title": neg_doc.get("title", ""),
                            "text": neg_doc["passage"],
                            "llm_reasoning": reasoning,
                        }
                    )

        except Exception as e:
            err_str = str(e)
            if "Connection" in err_str or "Errno" in err_str:
                if hasattr(self._thread_local, "llm"):
                    del self._thread_local.llm
            return {
                "_filter_status": "failed",
                "query_id": record.get("query_id"),
                "error": f"Negative check: {err_str}",
            }

        # === Build filtered record ===
        return {
            "_filter_status": "kept",
            "_filtered_out_negatives": filtered_out_negs,
            "query_id": record["query_id"],
            "original_query": record["original_query"],
            "rewritten_query": idata["rewritten_query"],
            "instruction": idata["instruction"],
            "final_positive": {
                "id": final_pos_id,
                "text": final_pos_doc,
                "title": final_pos_title,
                "source": final_pos_source,
            },
            "valid_synthetic_negatives": valid_negs,
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
        }
