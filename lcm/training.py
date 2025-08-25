import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from .encoder import StreamingEncoder
from .segmenter import Segmenter
from .rvq import ResidualVectorQuantizer

class CorpusDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 128):
        self.data = Path(path).read_bytes()
        self.seq_len = seq_len
    def __len__(self) -> int:
        return max(len(self.data) - self.seq_len, 0)
    def __getitem__(self, idx: int) -> torch.Tensor:
        chunk = self.data[idx:idx + self.seq_len]
        return torch.tensor(list(chunk), dtype=torch.long)

def build_models() -> tuple[StreamingEncoder, Segmenter, ResidualVectorQuantizer]:
    encoder = StreamingEncoder()
    segmenter = Segmenter()
    rvq = ResidualVectorQuantizer(1024, 512, 2)
    return encoder, segmenter, rvq

def train(corpus_path: str, epochs: int = 1, batch_size: int = 32, seq_len: int = 128) -> None:
    dataset = CorpusDataset(corpus_path, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder, segmenter, rvq = build_models()
    params = list(encoder.parameters()) + list(segmenter.parameters()) + list(rvq.parameters())
    optimizer = torch.optim.Adam(params)
    for _ in range(epochs):
        for batch in loader:
            features, _ = encoder(batch)
            loss = torch.tensor(0.0)
            for i in range(batch.size(0)):
                segs = segmenter.segment(features[i])
                pooled = [segmenter.pool(features[i], s, e) for s, e in segs]
                if not pooled:
                    continue
                pooled_tensor = torch.stack(pooled)
                quant, _ = rvq(pooled_tensor)
                loss = loss + torch.nn.functional.mse_loss(quant, pooled_tensor)
            if loss.item() == 0.0:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
