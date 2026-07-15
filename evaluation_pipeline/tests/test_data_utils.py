import json
import tempfile
import unittest
from pathlib import Path

from evaluation_pipeline.utils.data_utils import format_results_table, load_all_results


class DataUtilsTest(unittest.TestCase):
    def test_mteb_pmrr_is_scaled_and_normalized(self):
        table = format_results_table(
            [
                {
                    "model": "model",
                    "dataset": "mfollowir_ru",
                    "results": {
                        "mteb": [
                            {
                                "task_name": "mFollowIR",
                                "scores": {"test": [{"p-MRR": 0.1854}]},
                            }
                        ]
                    },
                }
            ]
        )
        self.assertIn("mFollowIR.p_mrr", table)
        self.assertIn("18.54", table)

    def test_manifest_is_not_loaded_as_evaluation_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_manifest.json").write_text("{}", encoding="utf-8")
            (root / "result.json").write_text(
                json.dumps({"model": "m", "dataset": "d", "results": {}}),
                encoding="utf-8",
            )
            self.assertEqual(len(load_all_results(tmp)), 1)


if __name__ == "__main__":
    unittest.main()
