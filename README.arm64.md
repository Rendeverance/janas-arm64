# Janas on ARM64: NEON kernels and Windows on ARM

An ARM64 port of [Janas](https://github.com/prabanta-dev/janas), the C engine by
Maurizio Cammalleri that runs large mixture-of-experts models by streaming the
experts from disk. This repository holds the port as it was written, on a
Snapdragon X mini PC (8 cores, 32 GB), between 14 and 16 October 2026.

Janas itself is Maurizio's work. The ARM64 work here is mine, and goes
upstream as small pull requests on his terms.

## What is here

The default branch, `arm64`, is upstream's Release 2026-10-14 plus:

- NEON and dot-product kernels for every quantised format Janas reads, single
  and batched, with the integer sums exact and the float steps in the order of
  the scalar references (`src/llm/quant_neon.h`, `quant_neon_k.h`).
- An I8MM tile (two rows by two vectors with `smmla`) for prompt processing
  (`src/llm/quant_neon_mm.h`).
- NEON for attention, DeltaNet, the norms, the router and the per-layer
  embeddings (`src/llm/attention_neon.h` and the model files).
- What Windows on ARM64 needs to build and run: the target CPU taken from the
  compiler, CPU features and memory from Windows, and a fix in `janas_free`.

Measured on that machine with Qwen3.6-35B-A3B (Q4_0, converted with
`gguf2jns`): about 32 tokens a second writing and 79 reading a 64-token prompt,
from 3.5 and 6.7 with the plain C build.

## Status

Work in progress. The quantised dot products are bit exact with the scalar
references and go upstream first. The float kernels for attention, DeltaNet and
the rest still reduce in a different order from the AVX2 path in a few places,
so their results can differ in the last bit; they are being fixed before they
are offered upstream.

## Licence

GNU General Public License, version 3 or later, as Janas (see `LICENSE`).
