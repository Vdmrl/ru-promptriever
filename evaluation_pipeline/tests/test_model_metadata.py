import unittest

from models.encoder_retriever import EncoderRetriever
from models.qwen3_embedding_retriever import Qwen3EmbeddingRetriever


class ModelMetadataTest(unittest.TestCase):
    def test_encoder_metadata_is_mutable_for_mteb_prediction_export(self):
        model = EncoderRetriever.__new__(EncoderRetriever)
        marker = object()
        model.mteb_model_meta = marker
        self.assertIs(model.mteb_model_meta, marker)

    def test_qwen_metadata_is_mutable_for_mteb_prediction_export(self):
        model = Qwen3EmbeddingRetriever.__new__(Qwen3EmbeddingRetriever)
        marker = object()
        model.mteb_model_meta = marker
        self.assertIs(model.mteb_model_meta, marker)


if __name__ == "__main__":
    unittest.main()
