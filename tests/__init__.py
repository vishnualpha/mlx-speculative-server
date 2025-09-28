# Copyright © 2025 Manus AI

"""
Test suite for MLX Speculative Decoding

This test suite provides comprehensive testing for the speculative decoding
implementation, including unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import mlx_speculative
test_dir = Path(__file__).parent
project_dir = test_dir.parent
sys.path.insert(0, str(project_dir))

# Test configuration
TEST_CONFIG = {
    "test_model_path": os.environ.get("TEST_MODEL_PATH", "microsoft/Phi-3-mini-4k-instruct"),
    "test_draft_model_path": os.environ.get("TEST_DRAFT_MODEL_PATH"),
    "max_test_tokens": int(os.environ.get("MAX_TEST_TOKENS", "50")),
    "test_timeout": float(os.environ.get("TEST_TIMEOUT", "30.0")),
    "skip_slow_tests": os.environ.get("SKIP_SLOW_TESTS", "false").lower() == "true",
    "skip_model_tests": os.environ.get("SKIP_MODEL_TESTS", "false").lower() == "true",
}

# Test prompts for consistent testing
TEST_PROMPTS = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot.",
    "What are the benefits of renewable energy?",
]

BATCH_TEST_PROMPTS = [
    "What is machine learning?",
    "How does photosynthesis work?",
    "Explain the theory of relativity.",
    "What is the meaning of life?",
    "How do computers work?",
    "What is artificial intelligence?",
    "Describe the water cycle.",
    "What causes climate change?",
]
