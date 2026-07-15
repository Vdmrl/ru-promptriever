from enum import Enum
import unittest

from evaluation_pipeline.models.prompt_utils import (
    _batch_texts,
    apply_role_prefix,
    materialize_texts,
    resolve_prompt_name,
)


class FakePromptType(Enum):
    query = "query"
    document = "document"


class ResolvePromptNameTest(unittest.TestCase):
    def test_legacy_names_are_preserved(self):
        self.assertEqual(resolve_prompt_name("query"), "query")
        self.assertEqual(resolve_prompt_name("passage"), "passage")

    def test_mteb_prompt_types_are_mapped(self):
        self.assertEqual(resolve_prompt_name(prompt_type=FakePromptType.query), "query")
        self.assertEqual(
            resolve_prompt_name(prompt_type=FakePromptType.document), "passage"
        )

    def test_explicit_prompt_name_takes_precedence(self):
        self.assertEqual(
            resolve_prompt_name("query", FakePromptType.document), "query"
        )

    def test_unknown_role_stays_unset(self):
        self.assertIsNone(resolve_prompt_name(prompt_type="other"))

    def test_official_promptriever_double_space_is_preserved(self):
        self.assertEqual(
            apply_role_prefix(["  text  "], "query", query_prefix="query:  "),
            ["query:  text"],
        )
        self.assertEqual(
            apply_role_prefix(
                ["  document  "], "passage", passage_prefix="passage:  "
            ),
            ["passage:  document"],
        )

    def test_text_batch_is_not_stringified(self):
        self.assertEqual(
            _batch_texts({"text": ["first", "second"], "id": ["1", "2"]}),
            ["first", "second"],
        )

    def test_single_batch_string_is_not_split_into_characters(self):
        self.assertEqual(_batch_texts({"text": "whole sentence"}), ["whole sentence"])

    def test_plain_iterable_materialization(self):
        self.assertEqual(materialize_texts(iter(["a", "b"])), ["a", "b"])

    def test_legacy_document_separator_can_be_reconstructed(self):
        batch = {
            "text": ["Title body", "body only"],
            "title": ["Title", ""],
            "body": ["body", "body only"],
        }
        self.assertEqual(
            _batch_texts(batch, document_title_separator=". "),
            ["Title. body", "body only"],
        )


if __name__ == "__main__":
    unittest.main()
