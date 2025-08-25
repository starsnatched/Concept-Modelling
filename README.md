# Concept Modelling

## Overview

Concept Modelling provides a modular pipeline for byte-level language processing. It encodes streams, segments spans, quantizes embeddings, retrieves related concepts, plans responses, and realizes byte outputs.

## Modules

- `lcm.encoder.StreamingEncoder`
- `lcm.segmenter.Segmenter`
- `lcm.rvq.ResidualVectorQuantizer`
- `lcm.store.ConceptStore`
- `lcm.inference.StreamingInference`
- `lcm.training`
- `lcm.planner.ConceptPlanner`
- `lcm.realizer.ByteDiffusionDecoder`
- `lcm.realizer.SegmentCTCRealizer`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from lcm.inference import StreamingInference
pipeline = StreamingInference()
segments, metadata, output = pipeline.process(b"example text")
```

## Training

```python
from lcm.training import train
train("corpus.txt", epochs=1, model_dir="models")
```

## Chat

```python
from lcm.chat import chat
chat("models")
```

## Testing

```bash
pytest -q
```
The test prints segments, metadata, decoded text, and output length.
