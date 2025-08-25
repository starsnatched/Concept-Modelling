import faiss
import numpy as np
from typing import Any

class ConceptStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.meta: list[dict[str, Any]] = []

    def insert(self, vectors: np.ndarray, metas: list[dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.meta.extend(metas)

    def lookup(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
        query = np.asarray(query, dtype=np.float32)
        distances, indices = self.index.search(query, k)
        retrieved = [[self.meta[i] for i in idx] for idx in indices]
        return distances, indices, retrieved
