# Copyright © 2025 Manus AI

"""
Test utilities and fixtures for MLX Speculative Decoding tests.
"""

import tempfile
import json
import os
from typing import Dict, Any, List, Optional
from unittest.mock import Mock
import mlx.core as mx

from mlx_speculative.models import ModelConfig
from mlx_speculative.core import SpeculativeEngine


class MockModel:
    """Mock model for testing purposes."""
    
    def __init__(
        self,
        num_layers: int = 12,
        head_dim: int = 64,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        vocab_size: int = 32000,
        hidden_size: int = 768,
    ):
        self.layers = [Mock() for _ in range(num_layers)]
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Mock parameters for memory estimation
        param_size = hidden_size * vocab_size  # Embedding layer
        param_size += num_layers * hidden_size * hidden_size * 4  # Transformer layers
        
        self._parameters = [Mock(nbytes=param_size // 10) for _ in range(10)]
    
    def parameters(self):
        """Return mock parameters."""
        return self._parameters
    
    def __call__(self, tokens, cache=None):
        """Mock forward pass."""
        batch_size, seq_len = tokens.shape
        return mx.random.normal((batch_size, seq_len, self.vocab_size))


class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        
        # Mock detokenizer
        self.detokenizer = Mock()
        self.detokenizer.reset = Mock()
        self.detokenizer.add_token = Mock()
        self.detokenizer.finalize = Mock()
        self.detokenizer.last_segment = "mock segment"
    
    def encode(self, text: str) -> List[int]:
        """Mock encoding - return simple token sequence."""
        # Simple mock: return token IDs based on text length
        return list(range(2, min(len(text) + 2, 50)))
    
    def decode(self, tokens: List[int]) -> str:
        """Mock decoding - return simple text."""
        return f"Generated text with {len(tokens)} tokens"
    
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        """Mock chat template application."""
        if isinstance(messages, list) and len(messages) > 0:
            content = messages[-1].get("content", "")
            formatted = f"<|user|>\n{content}\n<|assistant|>\n"
            return formatted if not tokenize else self.encode(formatted)
        return ""


def create_mock_engine(
    target_layers: int = 32,
    draft_layers: int = 16,
    num_draft_tokens: int = 4,
) -> SpeculativeEngine:
    """Create a mock speculative engine for testing."""
    target_model = MockModel(num_layers=target_layers)
    draft_model = MockModel(num_layers=draft_layers)
    
    return SpeculativeEngine(
        target_model=target_model,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
    )


def create_test_config(
    model_path: str = "/path/to/test/model",
    draft_model_path: Optional[str] = None,
    **kwargs
) -> ModelConfig:
    """Create a test model configuration."""
    return ModelConfig(
        model_path=model_path,
        draft_model_path=draft_model_path,
        **kwargs
    )


def create_temp_config_file(config_data: Dict[str, Any]) -> str:
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        return f.name


def cleanup_temp_file(file_path: str):
    """Clean up a temporary file."""
    if os.path.exists(file_path):
        os.unlink(file_path)


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_prompts(count: int = 10, min_length: int = 5, max_length: int = 50) -> List[str]:
        """Generate test prompts."""
        prompts = []
        base_prompts = [
            "What is the meaning of",
            "How does",
            "Explain the concept of",
            "Tell me about",
            "What are the benefits of",
            "How can I",
            "What is the difference between",
            "Why is",
            "When should I",
            "Where can I find",
        ]
        
        topics = [
            "artificial intelligence",
            "machine learning",
            "quantum computing",
            "renewable energy",
            "space exploration",
            "biotechnology",
            "climate change",
            "cryptocurrency",
            "virtual reality",
            "robotics",
        ]
        
        for i in range(count):
            base = base_prompts[i % len(base_prompts)]
            topic = topics[i % len(topics)]
            prompt = f"{base} {topic}?"
            prompts.append(prompt)
        
        return prompts
    
    @staticmethod
    def generate_token_sequences(
        batch_size: int = 4,
        seq_len: int = 20,
        vocab_size: int = 32000
    ) -> mx.array:
        """Generate test token sequences."""
        return mx.random.randint(0, vocab_size, (batch_size, seq_len))
    
    @staticmethod
    def generate_logits(
        batch_size: int = 4,
        seq_len: int = 20,
        vocab_size: int = 32000
    ) -> mx.array:
        """Generate test logits."""
        return mx.random.normal((batch_size, seq_len, vocab_size))


class PerformanceTimer:
    """Utility for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def start(self):
        """Start timing."""
        import time
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing."""
        import time
        self.end_time = time.time()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class MockServerClient:
    """Mock client for testing server endpoints."""
    
    def __init__(self):
        self.responses = {}
        self.request_history = []
    
    def set_response(self, endpoint: str, method: str, response_data: Dict[str, Any]):
        """Set a mock response for an endpoint."""
        key = f"{method.upper()} {endpoint}"
        self.responses[key] = response_data
    
    def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a mock request."""
        key = f"{method.upper()} {endpoint}"
        
        # Record request
        self.request_history.append({
            "method": method,
            "endpoint": endpoint,
            "kwargs": kwargs,
        })
        
        # Return mock response
        if key in self.responses:
            return self.responses[key]
        else:
            return {"status": "error", "message": f"No mock response for {key}"}
    
    def get(self, endpoint: str, **kwargs):
        """Mock GET request."""
        return self.request("GET", endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs):
        """Mock POST request."""
        return self.request("POST", endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs):
        """Mock DELETE request."""
        return self.request("DELETE", endpoint, **kwargs)


def assert_performance_target(
    actual_throughput: float,
    target_throughput: float,
    tolerance: float = 0.1
):
    """Assert that performance meets target with tolerance."""
    min_acceptable = target_throughput * (1 - tolerance)
    assert actual_throughput >= min_acceptable, (
        f"Performance below target: {actual_throughput:.1f} < {min_acceptable:.1f} "
        f"(target: {target_throughput:.1f} ±{tolerance*100:.1f}%)"
    )


def assert_memory_usage(
    actual_memory_mb: float,
    max_memory_mb: float
):
    """Assert that memory usage is within limits."""
    assert actual_memory_mb <= max_memory_mb, (
        f"Memory usage exceeds limit: {actual_memory_mb:.1f}MB > {max_memory_mb:.1f}MB"
    )


def create_sample_model_config() -> Dict[str, Any]:
    """Create a sample model configuration for testing."""
    return {
        "models": {
            "llama-7b": "/path/to/llama-7b",
            "llama-13b": "/path/to/llama-13b",
            "phi-3-mini": "/path/to/phi-3-mini",
        },
        "groups": {
            "llama-family": {
                "models": ["llama-7b", "llama-13b"],
                "default_model": "llama-7b",
                "description": "Llama model family",
                "tags": ["llama", "meta"],
            }
        },
        "settings": {
            "max_models": 10,
            "memory_limit_gb": 16.0,
            "auto_unload": True,
            "load_balancing": True,
            "default_model": "llama-7b",
        }
    }


class TestMetrics:
    """Collect and analyze test metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, name: str, value: float, unit: str = ""):
        """Record a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "unit": unit,
            "timestamp": __import__("time").time(),
        })
    
    def get_average(self, name: str) -> Optional[float]:
        """Get average value for a metric."""
        if name not in self.metrics:
            return None
        
        values = [m["value"] for m in self.metrics[name]]
        return sum(values) / len(values) if values else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for name, measurements in self.metrics.items():
            values = [m["value"] for m in measurements]
            if values:
                summary[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "unit": measurements[0].get("unit", ""),
                }
        
        return summary
    
    def print_summary(self):
        """Print metrics summary."""
        summary = self.get_summary()
        
        print("\nTest Metrics Summary:")
        print("-" * 50)
        
        for name, stats in summary.items():
            unit = f" {stats['unit']}" if stats['unit'] else ""
            print(f"{name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Average: {stats['average']:.3f}{unit}")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}{unit}")
            print()


# Global test metrics instance
test_metrics = TestMetrics()
