# Copyright © 2025 Manus AI

"""
MLX Speculative Decoding

A high-performance speculative decoding implementation for MLX, providing
vLLM-like capabilities for Apple Silicon with support for concurrent requests
and multiple models.
"""

__version__ = "0.1.0"

from .core import SpeculativeEngine
from .models import BatchedKVCache, load_model_pair
from .server import SpeculativeServer
from .utils import generate, batch_generate, stream_generate

__all__ = [
    "SpeculativeEngine",
    "BatchedKVCache", 
    "load_model_pair",
    "SpeculativeServer",
    "generate",
    "batch_generate", 
    "stream_generate",
]
