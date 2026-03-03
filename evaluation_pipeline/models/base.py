"""
Abstract base class for retrieval models.

All model wrappers implement the MTEB model protocol:
    encode(sentences, batch_size, **kwargs) -> np.ndarray
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseRetriever(ABC):
    """Abstract base retriever compatible with MTEB evaluation."""

    @abstractmethod
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """Encode a list of sentences into dense vectors.

        Args:
            sentences: List of text strings to encode.
            batch_size: Encoding batch size.

        Returns:
            np.ndarray of shape [len(sentences), hidden_dim].
        """
        ...
