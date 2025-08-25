import torch
from .inference import StreamingInference
from .io import load_models
from .utils import get_device


def build_pipeline(path: str, device: torch.device | None = None) -> StreamingInference:
    device = device or get_device()
    pipeline = StreamingInference(device=device)
    encoder, segmenter, rvq = load_models(path, device)
    pipeline.encoder = encoder
    pipeline.segmenter = segmenter
    pipeline.rvq = rvq
    return pipeline


def chat(path: str) -> None:
    pipeline = build_pipeline(path)
    while True:
        try:
            line = input(" > ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        _, _, out = pipeline.process(line.encode())
        print(out.decode(errors="ignore"))
