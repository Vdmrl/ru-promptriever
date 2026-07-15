from enum import Enum
import unittest

from evaluation_pipeline.models.prompt_utils import apply_role_prefix, resolve_prompt_name


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


if __name__ == "__main__":
    unittest.main()
