# Janas on ARM64: NEON kernels and Windows on ARM

An ARM64 port of [Janas](https://github.com/prabanta-dev/janas), the C engine by
Maurizio Cammalleri that runs large mixture-of-experts models by streaming the
experts from disk. This repository holds the port as it was written, on a
Snapdragon X mini PC (8 cores, 32 GB), between 14 and 16 October 2026.

Janas itself is Maurizio's work, and its own README is here as
[README.janas.md](README.janas.md). The ARM64 work here is mine, and goes
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

## Numbers

On that machine with Qwen3.6-35B-A3B (Q4_0, converted with `gguf2jns`) and a
64-token prompt: 29 to 32 tokens a second writing and 76 to 79 reading, up from
3.5 and 6.7 with the plain C build. llama.cpp's CPU build writes 22 to 24 on
the same machine and model.

## Building

Linux on ARM64:

    ./build.sh release test

Windows on ARM64, with the llvm-mingw toolchain on the path:

    JANAS_OS=windows CC=aarch64-w64-mingw32-clang AR=llvm-ar \
    WINDRES=aarch64-w64-mingw32-windres ./build.sh release test

## Status

Work in progress. The quantised dot products are bit exact with the scalar
references and are offered upstream first
([prabanta-dev/janas#32](https://github.com/prabanta-dev/janas/pull/32)), with
the `janas_free` fix
([#31](https://github.com/prabanta-dev/janas/pull/31)). The float kernels for
attention, DeltaNet and the rest still reduce in a different order from the
AVX2 path in a few places, so their results can differ in the last bit; they
are being fixed before they are offered upstream.

## Licence

GNU General Public License, version 3 or later, as Janas (see `LICENSE`).
