import unittest
from pathlib import Path

import yaml


class EvaluationConfigTest(unittest.TestCase):
    def test_mfollowir_paper_models_use_native_document_preprocessing(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "eval_mfollowir_significance.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        models = {model["name"]: model for model in config["models"]}

        self.assertEqual(
            models["ru-promptriever-qwen3-4b"]["document_title_separator"],
            ". ",
        )
        self.assertNotIn(
            "document_title_separator", models["promptriever-llama3.1-8b"]
        )

    def test_instruction_datasets_have_immutable_protocol_fields(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        errors = []
        for path in config_dir.glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for dataset in config.get("datasets", []):
                if dataset.get("type") == "synthetic_test":
                    for field in (
                        "data_path",
                        "revision",
                        "instruction_negative_field",
                    ):
                        if not dataset.get(field):
                            errors.append(f"{path.name}:{dataset.get('name')} missing {field}")
                if dataset.get("type") == "mfollowir":
                    if dataset.get("revision") != (
                        "09eecbe45c54b4a6dfb8e68e345cae77337768e2"
                    ):
                        errors.append(
                            f"{path.name}:{dataset.get('name')} has wrong revision"
                        )
                    if not dataset.get("save_predictions"):
                        errors.append(
                            f"{path.name}:{dataset.get('name')} must save predictions"
                        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
