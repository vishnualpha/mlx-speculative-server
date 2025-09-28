# Copyright © 2025 Manus AI

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import mlx.core as mx

from mlx_speculative.core import SpeculativeEngine
from mlx_speculative.models import BatchedKVCache
from mlx_speculative.sample_utils import top_p_sampling, create_sampler
from tests import TEST_CONFIG


class TestBatchedKVCache(unittest.TestCase):
    """Test the BatchedKVCache implementation."""
    
    def setUp(self):
        self.head_dim = 64
        self.n_kv_heads = 8
        self.batch_size = 4
        self.cache = BatchedKVCache(self.head_dim, self.n_kv_heads, self.batch_size)
    
    def test_initialization(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.head_dim, self.head_dim)
        self.assertEqual(self.cache.n_kv_heads, self.n_kv_heads)
        self.assertEqual(self.cache.batch_size, self.batch_size)
        self.assertEqual(self.cache.offset, 0)
        self.assertIsNone(self.cache.keys)
        self.assertIsNone(self.cache.values)
    
    def test_update_and_fetch(self):
        """Test updating and fetching from cache."""
        seq_len = 10
        keys = mx.random.normal((self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        values = mx.random.normal((self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        
        # First update
        all_keys, all_values = self.cache.update_and_fetch(keys, values)
        
        self.assertEqual(all_keys.shape, (self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        self.assertEqual(all_values.shape, (self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        self.assertEqual(self.cache.offset, seq_len)
        
        # Second update
        new_seq_len = 5
        new_keys = mx.random.normal((self.batch_size, self.n_kv_heads, new_seq_len, self.head_dim))
        new_values = mx.random.normal((self.batch_size, self.n_kv_heads, new_seq_len, self.head_dim))
        
        all_keys, all_values = self.cache.update_and_fetch(new_keys, new_values)
        
        expected_total_len = seq_len + new_seq_len
        self.assertEqual(all_keys.shape[2], expected_total_len)
        self.assertEqual(all_values.shape[2], expected_total_len)
        self.assertEqual(self.cache.offset, expected_total_len)
    
    def test_memory_limit(self):
        """Test memory limit functionality."""
        cache_with_limit = BatchedKVCache(
            self.head_dim, self.n_kv_heads, self.batch_size, max_size=100
        )
        
        # Add data that exceeds limit
        large_seq_len = 150
        keys = mx.random.normal((self.batch_size, self.n_kv_heads, large_seq_len, self.head_dim))
        values = mx.random.normal((self.batch_size, self.n_kv_heads, large_seq_len, self.head_dim))
        
        all_keys, all_values = cache_with_limit.update_and_fetch(keys, values)
        
        # Should be limited by max_size
        self.assertLessEqual(all_keys.shape[2], 100)
        self.assertLessEqual(cache_with_limit.offset, 100)
    
    def test_trim(self):
        """Test trimming functionality."""
        seq_len = 20
        keys = mx.random.normal((self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        values = mx.random.normal((self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        
        self.cache.update_and_fetch(keys, values)
        original_offset = self.cache.offset
        
        # Trim 5 tokens
        self.cache.trim(5)
        self.assertEqual(self.cache.offset, original_offset - 5)
    
    def test_reset(self):
        """Test reset functionality."""
        seq_len = 10
        keys = mx.random.normal((self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        values = mx.random.normal((self.batch_size, self.n_kv_heads, seq_len, self.head_dim))
        
        self.cache.update_and_fetch(keys, values)
        self.cache.reset()
        
        self.assertEqual(self.cache.offset, 0)
        self.assertIsNone(self.cache.keys)
        self.assertIsNone(self.cache.values)


class TestSamplingUtils(unittest.TestCase):
    """Test sampling utility functions."""
    
    def test_top_p_sampling(self):
        """Test top-p sampling."""
        # Create test logits
        vocab_size = 1000
        batch_size = 2
        logits = mx.random.normal((batch_size, vocab_size))
        
        # Test with different top_p values
        for top_p in [0.1, 0.5, 0.9, 1.0]:
            tokens = top_p_sampling(logits, top_p)
            self.assertEqual(tokens.shape, (batch_size,))
            self.assertTrue(mx.all(tokens >= 0))
            self.assertTrue(mx.all(tokens < vocab_size))
    
    def test_top_p_sampling_single(self):
        """Test top-p sampling with single sequence."""
        vocab_size = 100
        logits = mx.random.normal((vocab_size,))
        
        token = top_p_sampling(logits, 0.5)
        self.assertEqual(token.shape, ())
        self.assertTrue(0 <= token < vocab_size)
    
    def test_create_sampler(self):
        """Test sampler creation."""
        vocab_size = 50
        logits = mx.random.normal((vocab_size,))
        
        # Test greedy sampler
        greedy_sampler = create_sampler(temperature=0.0)
        token = greedy_sampler(logits)
        expected_token = mx.argmax(logits)
        self.assertEqual(token.item(), expected_token.item())
        
        # Test temperature sampler
        temp_sampler = create_sampler(temperature=1.0)
        token = temp_sampler(logits)
        self.assertTrue(0 <= token < vocab_size)
        
        # Test top-p sampler
        top_p_sampler = create_sampler(temperature=1.0, top_p=0.5)
        token = top_p_sampler(logits)
        self.assertTrue(0 <= token < vocab_size)


class TestSpeculativeEngine(unittest.TestCase):
    """Test the SpeculativeEngine class."""
    
    def setUp(self):
        """Set up mock models for testing."""
        # Create mock target model
        self.target_model = Mock()
        self.target_model.layers = [Mock() for _ in range(12)]  # 12 layers
        
        # Create mock draft model
        self.draft_model = Mock()
        self.draft_model.layers = [Mock() for _ in range(6)]  # 6 layers
        
        # Mock model attributes
        for model in [self.target_model, self.draft_model]:
            model.n_kv_heads = 8
            model.head_dim = 64
            model.vocab_size = 32000
        
        self.engine = SpeculativeEngine(
            target_model=self.target_model,
            draft_model=self.draft_model,
            num_draft_tokens=4,
        )
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.target_model, self.target_model)
        self.assertEqual(self.engine.draft_model, self.draft_model)
        self.assertEqual(self.engine.num_draft_tokens, 4)
        self.assertIsInstance(self.engine.stats, dict)
    
    def test_get_kv_heads(self):
        """Test KV heads extraction."""
        kv_heads = self.engine._get_kv_heads(self.target_model)
        self.assertEqual(len(kv_heads), 12)  # Number of layers
        self.assertTrue(all(heads == 8 for heads in kv_heads))
    
    def test_create_batched_cache(self):
        """Test batched cache creation."""
        batch_size = 4
        cache = self.engine._create_batched_cache(self.target_model, batch_size)
        
        self.assertEqual(len(cache), 12)  # Number of layers
        for layer_cache in cache:
            self.assertIsInstance(layer_cache, BatchedKVCache)
            self.assertEqual(layer_cache.batch_size, batch_size)
            self.assertEqual(layer_cache.n_kv_heads, 8)
            self.assertEqual(layer_cache.head_dim, 64)
    
    @patch('mlx_speculative.core.mx.random.bernoulli')
    def test_verify_draft_tokens(self, mock_bernoulli):
        """Test draft token verification."""
        # Mock acceptance decisions
        mock_bernoulli.return_value = mx.array([True, True, False, False])
        
        batch_size = 2
        num_draft = 4
        vocab_size = 100
        
        draft_tokens = mx.random.randint(0, vocab_size, (batch_size, num_draft))
        target_probs = mx.random.uniform(0, 1, (batch_size, num_draft, vocab_size))
        draft_probs = mx.random.uniform(0, 1, (batch_size, num_draft, vocab_size))
        
        accepted_tokens, num_accepted = self.engine._verify_draft_tokens(
            draft_tokens, target_probs, draft_probs
        )
        
        # Should accept first 2 tokens based on mock
        self.assertEqual(num_accepted, 2)
        self.assertEqual(accepted_tokens.shape, (batch_size, 2))
    
    def test_auto_draft_model_creation(self):
        """Test automatic draft model creation."""
        # Test with no draft model provided
        engine_auto = SpeculativeEngine(
            target_model=self.target_model,
            draft_model=None,
            auto_draft=True,
        )
        
        # Should use target model as draft (simplified implementation)
        self.assertEqual(engine_auto.draft_model, self.target_model)


class TestModelUtils(unittest.TestCase):
    """Test model utility functions."""
    
    @unittest.skipIf(TEST_CONFIG["skip_model_tests"], "Model tests disabled")
    def test_model_info_extraction(self):
        """Test model information extraction."""
        from mlx_speculative.models import get_model_info
        
        # Create a mock model with attributes
        mock_model = Mock()
        mock_model.parameters.return_value = [
            mx.zeros((100, 200)),  # 20k parameters
            mx.zeros((50, 100)),   # 5k parameters
        ]
        mock_model.layers = [Mock() for _ in range(6)]
        mock_model.head_dim = 64
        mock_model.n_heads = 12
        mock_model.n_kv_heads = 4
        mock_model.vocab_size = 32000
        
        info = get_model_info(mock_model)
        
        self.assertEqual(info["num_layers"], 6)
        self.assertEqual(info["head_dim"], 64)
        self.assertEqual(info["n_heads"], 12)
        self.assertEqual(info["n_kv_heads"], 4)
        self.assertEqual(info["vocab_size"], 32000)
        self.assertTrue(info["has_layers"])


if __name__ == "__main__":
    unittest.main()
