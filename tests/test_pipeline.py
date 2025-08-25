from lcm.inference import StreamingInference

def test_process_runs():
    pipeline = StreamingInference()
    data = b"hello world"
    segments, metas = pipeline.process(data)
    assert isinstance(segments, list)
    assert len(segments) == len(metas)
