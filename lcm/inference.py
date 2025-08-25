import numpy as np
import torch
from .encoder import StreamingEncoder
from .segmenter import Segmenter
from .rvq import ResidualVectorQuantizer
from .store import ConceptStore
from .denoiser import ConceptDenoiser

class StreamingInference:
    def __init__(self, hidden_dim: int = 512, codebook_size: int = 1024, levels: int = 2, latent_dim: int = 512, latent_len: int = 128):
        self.encoder = StreamingEncoder(hidden_dim)
        self.segmenter = Segmenter(hidden_dim)
        self.rvq = ResidualVectorQuantizer(codebook_size, hidden_dim, levels)
        self.store = ConceptStore(hidden_dim)
        self.denoiser = ConceptDenoiser(latent_dim, latent_len)
        self.state: torch.Tensor | None = None

    def process(self, data: bytes, steps: int = 2, k: int = 4):
        byte_tensor = torch.tensor(list(data), dtype=torch.long).unsqueeze(0)
        features, self.state = self.encoder(byte_tensor, self.state)
        segments = self.segmenter.segment(features[0])
        vectors = []
        metas = []
        for start, end in segments:
            pooled = self.segmenter.pool(features[0], start, end).unsqueeze(0)
            quant, codes = self.rvq(pooled)
            vectors.append(quant.detach().cpu().numpy())
            metas.append({'span': (start, end), 'codes': codes.squeeze(0).tolist()})
        if vectors:
            stacked = np.vstack(vectors)
            self.store.insert(stacked, metas)
            _, indices, _ = self.store.lookup(stacked, k)
            retrieved_vectors = stacked[indices].reshape(1, -1, stacked.shape[1])
            retrieved_torch = torch.tensor(retrieved_vectors, dtype=torch.float32)
        else:
            retrieved_torch = None
        self.denoiser(features, retrieved_torch, steps)
        return segments, metas
