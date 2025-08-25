import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from lcm.inference import StreamingInference

def decode_segments(data: bytes, segments: list[tuple[int, int]]) -> list[str]:
    return [data[s:e].decode('utf-8', errors='ignore') for s, e in segments]

def test_process_runs() -> None:
    pipeline = StreamingInference()
    data = b"hello world"
    segments, metas, output = pipeline.process(data)
    decoded = decode_segments(data, segments)
    assert isinstance(segments, list)
    assert isinstance(output, bytes)
    assert len(output) > 0
    assert len(segments) == len(metas)
    assert len(decoded) == len(segments)
    print("Segments:", segments)
    print("Metas:", metas)
    print("Decoded:", decoded)
    print("Output bytes length:", len(output))

if __name__ == "__main__":
    test_process_runs()
    print("Test completed successfully.")