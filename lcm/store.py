import numpy as np
from typing import Any

class ConceptStore:
    def __init__(self, dim: int):
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.meta: list[dict[str, Any]] = []

    def insert(self, vectors: np.ndarray, metas: list[dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.size:
            self.vectors = np.vstack([self.vectors, vectors])
        else:
            self.vectors = vectors
        self.meta.extend(metas)

    def lookup(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
        query = np.asarray(query, dtype=np.float32)
        if not self.vectors.size:
            distances = np.full((query.shape[0], k), np.inf, dtype=np.float32)
            indices = np.full((query.shape[0], k), -1, dtype=int)
            retrieved = [[{} for _ in range(k)] for _ in range(query.shape[0])]
            return distances, indices, retrieved
        dists = ((self.vectors[None, :, :] - query[:, None, :]) ** 2).sum(axis=2)
        indices = np.argsort(dists, axis=1)[:, :k]
        distances = np.take_along_axis(dists, indices, axis=1)
        retrieved = [[self.meta[i] for i in idx] for idx in indices]
        return distances, indices, retrieved
