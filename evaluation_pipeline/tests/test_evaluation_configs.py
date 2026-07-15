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

    def test_rebuttal_ru_models_use_native_document_preprocessing(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        expected = {
            "eval_followir_significance.yaml": {"ru-only-paper", "ru-en-paper"},
            "eval_step1500_full.yaml": {"4b-pretrain-step-1500"},
        }
        errors = []
        for filename, model_names in expected.items():
            config = yaml.safe_load(
                (config_dir / filename).read_text(encoding="utf-8")
            )
            models = {model["name"]: model for model in config["models"]}
            for model_name in model_names:
                if models[model_name].get("document_title_separator") != ". ":
                    errors.append(f"{filename}:{model_name}")
        self.assertEqual(errors, [])

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

    def test_final_paper_config_pins_every_hub_model(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "eval_final_paper_missing.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        errors = []
        for model in config["models"]:
            if model["type"] == "bm25":
                continue
            if not model.get("revision"):
                errors.append(f"{model['name']} has no revision")
            if model["type"] == "causal_lm" and not model.get("base_revision"):
                errors.append(f"{model['name']} has no base_revision")
        self.assertEqual(errors, [])

    def test_final_ru_models_use_one_native_format(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "eval_final_paper_missing.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        ru_models = [
            model
            for model in config["models"]
            if model["type"] == "causal_lm"
            and not model["name"].startswith("promptriever-")
        ]
        self.assertTrue(ru_models)
        self.assertEqual(
            [
                model["name"]
                for model in ru_models
                if model.get("document_title_separator") != ". "
                or model.get("append_eos") is not True
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
