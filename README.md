# Concept Modelling

## Overview

Concept Modelling provides a modular pipeline for byte-level language processing. It encodes streams, segments spans, quantizes embeddings, retrieves related concepts, and denoises representations.

## Modules

- `lcm.encoder.StreamingEncoder`
- `lcm.segmenter.Segmenter`
- `lcm.rvq.ResidualVectorQuantizer`
- `lcm.store.ConceptStore`
- `lcm.denoiser.ConceptDenoiser`
- `lcm.inference.StreamingInference`
- `lcm.training`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from lcm.inference import StreamingInference
pipeline = StreamingInference()
segments, metadata = pipeline.process(b"example text")
```

## Training

```python
from lcm.training import train
train("corpus.txt", epochs=1)
```

## Testing

```bash
pytest -q
```
The test prints segments, metadata, and decoded text.
