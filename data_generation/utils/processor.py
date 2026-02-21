import time
import threading
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.llm_init import create_llm_instance
from utils.prompts import (
    SYSTEM_PROMPT,
    INSTRUCTION_GENERATION_PROMPT,
    INSTRUCTION_NEGATIVE_MINING_PROMPT,
    InstructionGenOutput,
    InstructionNegativeMiningOutput,
    get_random_generation_params
)


class Processor:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.instruct_parser = PydanticOutputParser(pydantic_object=InstructionGenOutput)
        self.mining_parser = PydanticOutputParser(pydantic_object=InstructionNegativeMiningOutput)
        self._thread_local = threading.local()

    def _get_llm(self):
        if not hasattr(self._thread_local, "llm"):
            # Initialize once per thread. Temperature is now loaded from config.
            self._thread_local.llm = create_llm_instance(
                config_path=self.config_path
            )
        return self._thread_local.llm

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _generate_instruction(self, llm, query, pos_doc, neg_doc, pos_id, neg_id):
        params = get_random_generation_params()

        # No manual formatting here, inputs map directly to prompt placeholders
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", INSTRUCTION_GENERATION_PROMPT)
        ]).partial(format_instructions=self.instruct_parser.get_format_instructions())

        chain = prompt | llm | self.instruct_parser

        return chain.invoke({
            "query": query,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
            "pos_id": pos_id,
            "neg_id": neg_id,
            **params
        })

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _generate_negatives(self, llm, query, instruction):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", INSTRUCTION_NEGATIVE_MINING_PROMPT)
        ]).partial(format_instructions=self.mining_parser.get_format_instructions())

        chain = prompt | llm | self.mining_parser

        return chain.invoke({
            "query": query,
            "instruction": instruction
        })

    def process_sample(self, sample):
        try:
            llm = self._get_llm()
        except Exception as e:
            return {"query_id": sample[0], "status": "failed", "error": f"LLM Init: {e}"}

        qid, q_text, pid, p_text, nid, n_text = sample

        result = {
            "query_id": qid,
            "original_query": q_text,
            "original_positive_id": pid,
            "original_negative_id": nid,
            "status": "failed",
            "error": None
        }

        try:
            # Rewrite docs & Generate instruction
            instruct_out = self._generate_instruction(llm, q_text, p_text, n_text, pid, nid)

            # Mining using the CLEAN rewritten query from Step 1
            mining_out = self._generate_negatives(
                llm,
                instruct_out.rewritten_query,
                instruct_out.instruction
            )

            result["status"] = "success"
            result["instruction_data"] = instruct_out.dict()
            result["mining_data"] = mining_out.dict()
            result["generated_instruction_text"] = instruct_out.instruction

        except Exception as e:
            # If socket dies, clear it so next retry makes a new one
            err_str = str(e)
            if "Connection" in err_str or "Errno" in err_str:
                if hasattr(self._thread_local, "llm"):
                    del self._thread_local.llm
            result["error"] = err_str

        return result