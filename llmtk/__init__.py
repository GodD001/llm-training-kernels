"""llmtk: from-scratch Transformer LM training stack with FlashAttention-2 kernels.

Triton kernels live in `llmtk.kernels` and are imported only on demand (Triton is Linux/CUDA only).
"""

__version__ = "0.1.0"
