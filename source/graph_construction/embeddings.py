"""
The shared sentence encoder.

One encoder is used everywhere in Nexus: fact embeddings for the label
extraction traversal, entity names during entity linking, and entity names and
hyperedge descriptions when the vector indexes are built. Keep one instance and
pass it around rather than constructing a second model elsewhere, so that every
vector in the pipeline lives in the same space.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 (384-dim). CPU is fine for corpora of this size."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 512,
        ).astype(np.float32)


def load_embedder(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    **kwargs,
) -> MiniLMEmbedder:
    """Load the shared encoder. `device` is named here so that callers can pin
    it to the same card as the rest of a step."""
    return MiniLMEmbedder(model_name, device=device, **kwargs)
