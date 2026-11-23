"""Stage 4 - Indexing / storage.

Input : chunks + their embeddings
Output: persisted index on disk (FAISS vectors + chunk metadata + BM25 corpus)

Two indexes are kept side by side:
  * FAISS IndexFlatIP - exact cosine search over dense vectors (semantic match)
  * BM25Okapi         - classic keyword scoring (exact terms, IDs, numbers, names)
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk

_WORD_RE = re.compile(r"\w+")


def bm25_tokenize(text: str) -> list[str]:
    """Lower-cased word tokens. BM25 needs whole words, not sub-word pieces."""
    return _WORD_RE.findall(text.lower())


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    # ---- build -------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self._bm25 = None  # rebuilt lazily

    @property
    def bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._bm25 = BM25Okapi([bm25_tokenize(c.text) for c in self.chunks])
        return self._bm25

    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- search ------------------------------------------------------------
    def dense_search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        scores, ids = self.index.search(qvec, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(bm25_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    # ---- persistence -------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore | None":
        vpath, cpath = directory / "vectors.faiss", directory / "chunks.json"
        if not (vpath.exists() and cpath.exists()):
            return None
        index = faiss.read_index(str(vpath))
        store = cls(index.d)
        store.index = index
        with open(cpath, encoding="utf-8") as f:
            store.chunks = [Chunk(**d) for d in json.load(f)]
        return store
