import json
import tempfile
import unittest
from pathlib import Path

from evaluation_pipeline.utils.run_manifest import (
    find_matching_result,
    protocol_fingerprint,
)


class RunManifestTest(unittest.TestCase):
    def test_fingerprint_is_order_independent(self):
        self.assertEqual(
            protocol_fingerprint({"a": 1, "b": {"x": 2}}),
            protocol_fingerprint({"b": {"x": 2}, "a": 1}),
        )

    def test_old_result_without_fingerprint_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps({"model": "m"}), encoding="utf-8")
            self.assertIsNone(find_matching_result(tmp, "expected"))

    def test_matching_result_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                json.dumps({"protocol_fingerprint": "expected"}), encoding="utf-8"
            )
            self.assertEqual(find_matching_result(tmp, "expected"), path)


if __name__ == "__main__":
    unittest.main()
