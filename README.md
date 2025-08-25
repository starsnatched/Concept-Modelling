# Concept Modelling

## Overview

Concept Modelling implements a modular pipeline for streaming language processing without sequence limits. It encodes byte streams, segments concepts, quantizes representations, retrieves from an external store, and denoises with latent reasoning.

## Modules

- `lcm.encoder.StreamingEncoder` processes byte sequences with a recurrent backbone.
- `lcm.segmenter.Segmenter` infers span boundaries and pools segment features.
- `lcm.rvq.ResidualVectorQuantizer` compresses embeddings using residual vector quantization.
- `lcm.store.ConceptStore` persists concept vectors with approximate nearest neighbor search.
- `lcm.denoiser.ConceptDenoiser` refines local and retrieved representations through attention-based latent updates.
- `lcm.inference.StreamingInference` coordinates the full pipeline for online processing.

## Usage

```python
from lcm.inference import StreamingInference
pipeline = StreamingInference()
segments, metadata = pipeline.process(b"example text")
```

## Testing

`pytest` runs the pipeline sanity check.
