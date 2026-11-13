# llm-training-kernels

From-scratch Transformer LM training stack with a fused FlashAttention-2 Triton kernel and profiling-driven optimization.

> **Status: work in progress.** The code and tests are in place; GPU benchmarks are still being measured.
> No performance numbers are listed until they come from the scripts in this repo.

## What's here

- `llmtk/model`: Transformer LM built from scratch (Linear, Embedding, RMSNorm, SwiGLU, RoPE, causal multi-head attention)
- `llmtk/tokenizer`: byte-level BPE with a parallel, heap-based trainer
- `llmtk/optim`: AdamW, cosine LR schedule with warmup, global gradient clipping
- `llmtk/kernels`: FlashAttention-2, as a tiled PyTorch reference and in Triton (fused forward kernel, plus backward kernels for dK/dV and dQ); see [docs/flash_attention_backward.md](docs/flash_attention_backward.md)
- `train.py`, `scripts/encode_dataset.py`, `benchmarks/`

## Quickstart

```bash
pip install -e '.[dev]'
pytest -q            # Triton tests run only on a CUDA GPU
```

## Acknowledgements

Built as a self-study project following Stanford CS336 (Language Modeling from Scratch). All implementations in this repository are my own.
