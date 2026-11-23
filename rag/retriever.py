"""Stages 5 + 6 - Hybrid retrieval and cross-encoder reranking.

Input : user query (string)
Output: list[RetrievedChunk] - the `final_top_k` most relevant chunks, best first

Flow:
  dense (FAISS, top 20) --+
                          +--> Reciprocal Rank Fusion --> cross-encoder rerank --> top 6
  sparse (BM25, top 20) --+
"""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .chunker import Chunk
from .config import Settings
from .embedder import Embedder
from .vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float          # reranker score (or RRF score when reranker is off)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> dict[int, float]:
    """score(d) = sum over rankings of 1 / (k + rank). Rank-based, so the very
    different score scales of cosine and BM25 never need to be normalised."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(self, cfg: Settings, embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.reranker = CrossEncoder(cfg.reranker_model, device="cpu") if cfg.use_reranker else None

    def retrieve(self, store: VectorStore, query: str) -> list[RetrievedChunk]:
        if len(store) == 0:
            return []

        dense = store.dense_search(self.embedder.embed_query(query), self.cfg.dense_top_k)
        sparse = store.sparse_search(query, self.cfg.sparse_top_k)
        fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]], self.cfg.rrf_k)
        candidates = sorted(fused, key=fused.get, reverse=True)

        if self.reranker is None:
            return [RetrievedChunk(store.chunks[i], fused[i]) for i in candidates[: self.cfg.final_top_k]]

        pairs = [(query, store.chunks[i].text) for i in candidates]
        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [RetrievedChunk(store.chunks[i], float(s)) for i, s in ranked[: self.cfg.final_top_k]]
