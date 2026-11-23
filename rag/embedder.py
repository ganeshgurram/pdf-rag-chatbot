"""Stage 3 - Embedding.

Input : list[str] (chunks) or a single query string
Output: float32 numpy array, shape (n, 384), L2-normalised

Normalised vectors make inner product == cosine similarity, which is what the
FAISS IndexFlatIP in the vector store computes.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings


class Embedder:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedding_model, device="cpu")
        self.tokenizer = self.model.tokenizer
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return vecs.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        # Asymmetric retrieval: short query vs long passage -> bge wants the
        # instruction prefix on the query only.
        vec = self.model.encode(
            [self.cfg.query_instruction + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype("float32")
