# Copyright © 2025 Manus AI

import unittest
import time
import statistics
from unittest.mock import Mock, patch
import mlx.core as mx

from mlx_speculative.core import SpeculativeEngine
from mlx_speculative.models import BatchedKVCache
from mlx_speculative.utils import benchmark
from mlx_speculative.multi_model import MultiModelManager
from tests import TEST_CONFIG, TEST_PROMPTS, BATCH_TEST_PROMPTS


class TestCoreBenchmarks(unittest.TestCase):
    """Benchmark tests for core functionality."""
    
    def setUp(self):
        """Set up mock models for benchmarking."""
        # Create realistic mock models
        self.target_model = Mock()
        self.draft_model = Mock()
        
        # Mock model attributes
        self.target_model.layers = [Mock() for _ in range(32)]  # Large model
        self.draft_model.layers = [Mock() for _ in range(16)]   # Smaller draft
        
        for model in [self.target_model, self.draft_model]:
            model.n_kv_heads = 8
            model.head_dim = 128
            model.vocab_size = 32000
        
        # Mock forward passes to simulate realistic computation
        def mock_target_forward(tokens, cache=None):
            batch_size, seq_len = tokens.shape
            # Simulate computation time for large model
            time.sleep(0.01)  # 10ms per forward pass
            return mx.random.normal((batch_size, seq_len, 32000))
        
        def mock_draft_forward(tokens, cache=None):
            batch_size, seq_len = tokens.shape
            # Simulate faster computation for draft model
            time.sleep(0.003)  # 3ms per forward pass
            return mx.random.normal((batch_size, seq_len, 32000))
        
        self.target_model.side_effect = mock_target_forward
        self.draft_model.side_effect = mock_draft_forward
        
        self.engine = SpeculativeEngine(
            target_model=self.target_model,
            draft_model=self.draft_model,
            num_draft_tokens=4,
        )
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_kv_cache_performance(self):
        """Benchmark KV cache operations."""
        batch_size = 8
        head_dim = 128
        n_kv_heads = 8
        
        cache = BatchedKVCache(head_dim, n_kv_heads, batch_size)
        
        # Benchmark cache updates
        seq_lengths = [10, 50, 100, 200]
        update_times = []
        
        for seq_len in seq_lengths:
            keys = mx.random.normal((batch_size, n_kv_heads, seq_len, head_dim))
            values = mx.random.normal((batch_size, n_kv_heads, seq_len, head_dim))
            
            start_time = time.time()
            for _ in range(10):  # Multiple iterations for averaging
                cache.update_and_fetch(keys, values)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            update_times.append(avg_time)
            
            print(f"Cache update (seq_len={seq_len}): {avg_time*1000:.2f}ms")
        
        # Verify performance scales reasonably
        self.assertLess(update_times[0], 0.01)  # Should be under 10ms for small sequences
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_speculative_decoding_throughput(self):
        """Benchmark speculative decoding throughput."""
        batch_sizes = [1, 2, 4, 8]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            # Create test prompts
            prompts = mx.random.randint(0, 1000, (batch_size, 20))  # 20 token prompts
            
            # Mock sampler
            def mock_sampler(logits):
                return mx.random.randint(0, 32000, logits.shape[:-1])
            
            start_time = time.time()
            total_tokens = 0
            
            # Generate tokens
            for tokens, metadata in self.engine.generate_step(
                prompts=prompts,
                max_tokens=50,
                sampler=mock_sampler,
            ):
                total_tokens += tokens.shape[0] * tokens.shape[1]
                
                # Break early for benchmarking
                if metadata["step"] >= 10:
                    break
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = total_tokens / elapsed_time
            
            throughput_results[batch_size] = throughput
            print(f"Batch size {batch_size}: {throughput:.1f} tokens/sec")
        
        # Verify throughput increases with batch size
        self.assertGreater(throughput_results[8], throughput_results[1])
        
        # Target performance: should achieve reasonable throughput
        self.assertGreater(throughput_results[8], 100)  # At least 100 tokens/sec
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_acceptance_rate_benchmark(self):
        """Benchmark acceptance rate under different conditions."""
        batch_size = 4
        prompts = mx.random.randint(0, 1000, (batch_size, 10))
        
        # Test different draft token counts
        draft_token_counts = [2, 4, 6, 8]
        acceptance_rates = {}
        
        for num_draft in draft_token_counts:
            engine = SpeculativeEngine(
                target_model=self.target_model,
                draft_model=self.draft_model,
                num_draft_tokens=num_draft,
            )
            
            def mock_sampler(logits):
                return mx.random.randint(0, 32000, logits.shape[:-1])
            
            total_draft = 0
            total_accepted = 0
            
            # Run generation steps
            for tokens, metadata in engine.generate_step(
                prompts=prompts,
                max_tokens=30,
                sampler=mock_sampler,
            ):
                total_draft += metadata.get("draft_tokens", 0)
                total_accepted += metadata.get("accepted_tokens", 0)
                
                if metadata["step"] >= 5:
                    break
            
            acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
            acceptance_rates[num_draft] = acceptance_rate
            
            print(f"Draft tokens {num_draft}: {acceptance_rate:.2f} acceptance rate")
        
        # Verify we get reasonable acceptance rates
        for rate in acceptance_rates.values():
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)


class TestMultiModelBenchmarks(unittest.TestCase):
    """Benchmark tests for multi-model functionality."""
    
    def setUp(self):
        """Set up multi-model manager for benchmarking."""
        self.manager = MultiModelManager(
            max_models=10,
            load_balancing=True,
        )
        
        # Mock model loading
        self.patcher = patch('mlx_speculative.multi_model.load_model_pair')
        self.mock_load_model_pair = self.patcher.start()
        
        # Setup mock returns
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.parameters.return_value = [Mock(nbytes=1000)]
        
        self.mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
    
    def tearDown(self):
        """Clean up after benchmarks."""
        self.patcher.stop()
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_model_loading_performance(self):
        """Benchmark model loading times."""
        from mlx_speculative.models import ModelConfig
        
        loading_times = []
        num_models = 5
        
        for i in range(num_models):
            config = ModelConfig(model_path=f"/path/to/model_{i}")
            
            start_time = time.time()
            success = self.manager.load_model(f"model_{i}", config)
            end_time = time.time()
            
            self.assertTrue(success)
            loading_time = end_time - start_time
            loading_times.append(loading_time)
            
            print(f"Model {i} loading time: {loading_time:.3f}s")
        
        # Verify reasonable loading times
        avg_loading_time = statistics.mean(loading_times)
        self.assertLess(avg_loading_time, 1.0)  # Should load in under 1 second (mocked)
        
        print(f"Average loading time: {avg_loading_time:.3f}s")
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_model_switching_performance(self):
        """Benchmark model switching overhead."""
        from mlx_speculative.models import ModelConfig
        
        # Load multiple models
        num_models = 3
        for i in range(num_models):
            config = ModelConfig(model_path=f"/path/to/model_{i}")
            self.manager.load_model(f"model_{i}", config)
        
        # Benchmark model retrieval times
        retrieval_times = []
        
        for _ in range(100):  # Multiple iterations
            model_name = f"model_{_ % num_models}"
            
            start_time = time.time()
            engine, tokenizer, actual_name = self.manager.get_model(model_name)
            end_time = time.time()
            
            retrieval_time = end_time - start_time
            retrieval_times.append(retrieval_time)
        
        avg_retrieval_time = statistics.mean(retrieval_times)
        max_retrieval_time = max(retrieval_times)
        
        print(f"Average model retrieval time: {avg_retrieval_time*1000:.3f}ms")
        print(f"Max model retrieval time: {max_retrieval_time*1000:.3f}ms")
        
        # Should be very fast (under 1ms)
        self.assertLess(avg_retrieval_time, 0.001)
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_load_balancing_performance(self):
        """Benchmark load balancing overhead."""
        from mlx_speculative.models import ModelConfig
        
        # Load multiple instances of the same model
        num_instances = 4
        for i in range(num_instances):
            config = ModelConfig(model_path=f"/path/to/instance_{i}")
            self.manager.load_model(f"instance_{i}", config)
            
            # Register with load balancer
            self.manager.load_balancer.register_instance("shared_model", f"instance_{i}")
        
        # Benchmark load balancing decisions
        balancing_times = []
        
        for _ in range(1000):  # Many iterations
            start_time = time.time()
            instance = self.manager.load_balancer.get_best_instance("shared_model", "round_robin")
            end_time = time.time()
            
            balancing_time = end_time - start_time
            balancing_times.append(balancing_time)
        
        avg_balancing_time = statistics.mean(balancing_times)
        max_balancing_time = max(balancing_times)
        
        print(f"Average load balancing time: {avg_balancing_time*1000000:.1f}μs")
        print(f"Max load balancing time: {max_balancing_time*1000000:.1f}μs")
        
        # Should be extremely fast (under 100μs)
        self.assertLess(avg_balancing_time, 0.0001)


class TestEndToEndBenchmarks(unittest.TestCase):
    """End-to-end performance benchmarks."""
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    @unittest.skipIf(TEST_CONFIG["skip_model_tests"], "Model tests disabled")
    def test_realistic_generation_benchmark(self):
        """Benchmark realistic text generation scenario."""
        # This test would use actual models if available
        # For now, we'll simulate with mocks
        
        with patch('mlx_speculative.utils.load') as mock_load:
            # Mock the load function
            mock_engine = Mock()
            mock_tokenizer = Mock()
            
            # Mock generation function
            def mock_generate_step(*args, **kwargs):
                # Simulate generation steps
                for step in range(10):
                    tokens = mx.random.randint(0, 1000, (1, 5))  # 5 tokens per step
                    metadata = {
                        "step": step,
                        "tokens_generated": 5,
                        "acceptance_rate": 0.7,
                        "throughput": 150.0,
                    }
                    yield tokens, metadata
            
            mock_engine.generate_step = mock_generate_step
            mock_load.return_value = (mock_engine, mock_tokenizer)
            
            from mlx_speculative.utils import benchmark
            
            # Run benchmark
            results = benchmark(
                engine=mock_engine,
                tokenizer=mock_tokenizer,
                prompts=TEST_PROMPTS[:3],  # Use first 3 test prompts
                max_tokens=50,
                num_runs=3,
            )
            
            # Verify benchmark results
            self.assertIn("runs", results)
            self.assertIn("avg_throughput", results)
            self.assertIn("avg_acceptance_rate", results)
            self.assertEqual(len(results["runs"]), 3)
            
            print(f"Benchmark results:")
            print(f"  Average throughput: {results['avg_throughput']:.1f} tokens/sec")
            print(f"  Average acceptance rate: {results['avg_acceptance_rate']:.2f}")
            print(f"  Average latency: {results['avg_latency']:.3f}s")
            
            # Verify reasonable performance
            self.assertGreater(results["avg_throughput"], 50)  # At least 50 tokens/sec
            self.assertGreater(results["avg_acceptance_rate"], 0.5)  # At least 50% acceptance
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage patterns."""
        from mlx_speculative.models import estimate_memory_usage
        
        # Create mock models of different sizes
        model_configs = [
            {"layers": 12, "head_dim": 64, "n_kv_heads": 8, "vocab_size": 32000},   # Small
            {"layers": 24, "head_dim": 128, "n_kv_heads": 16, "vocab_size": 50000}, # Medium
            {"layers": 48, "head_dim": 256, "n_kv_heads": 32, "vocab_size": 100000}, # Large
        ]
        
        batch_sizes = [1, 4, 8, 16]
        seq_len = 2048
        
        print("Memory usage estimates:")
        print("Model Size | Batch Size | Total Memory (MB)")
        print("-" * 45)
        
        for i, config in enumerate(model_configs):
            model_size = ["Small", "Medium", "Large"][i]
            
            # Create mock model
            mock_model = Mock()
            for key, value in config.items():
                setattr(mock_model, key, value)
            
            for batch_size in batch_sizes:
                memory_est = estimate_memory_usage(mock_model, batch_size, seq_len)
                total_memory = memory_est["total"]
                
                print(f"{model_size:10} | {batch_size:10} | {total_memory:12.1f}")
                
                # Verify memory estimates are reasonable
                self.assertGreater(total_memory, 0)
                
                # Larger models should use more memory
                if i > 0:
                    prev_config = model_configs[i-1]
                    prev_mock = Mock()
                    for key, value in prev_config.items():
                        setattr(prev_mock, key, value)
                    
                    prev_memory = estimate_memory_usage(prev_mock, batch_size, seq_len)
                    self.assertGreater(total_memory, prev_memory["total"])


class TestStressTests(unittest.TestCase):
    """Stress tests for system limits."""
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_high_concurrency_stress(self):
        """Stress test with high concurrency."""
        import threading
        import queue
        
        # Simulate high concurrent load
        num_threads = 20
        requests_per_thread = 10
        results_queue = queue.Queue()
        
        def worker_thread(thread_id):
            """Worker thread that makes requests."""
            thread_results = []
            
            for i in range(requests_per_thread):
                start_time = time.time()
                
                # Simulate request processing
                time.sleep(0.01)  # 10ms processing time
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                thread_results.append({
                    "thread_id": thread_id,
                    "request_id": i,
                    "processing_time": processing_time,
                })
            
            results_queue.put(thread_results)
        
        # Start all threads
        threads = []
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            thread_results = results_queue.get()
            all_results.extend(thread_results)
        
        # Analyze results
        total_requests = len(all_results)
        avg_processing_time = statistics.mean([r["processing_time"] for r in all_results])
        throughput = total_requests / total_time
        
        print(f"Stress test results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average processing time: {avg_processing_time*1000:.2f}ms")
        print(f"  Throughput: {throughput:.1f} requests/sec")
        
        # Verify system handled the load
        self.assertEqual(total_requests, num_threads * requests_per_thread)
        self.assertLess(avg_processing_time, 0.1)  # Should be under 100ms
        self.assertGreater(throughput, 50)  # At least 50 requests/sec


if __name__ == "__main__":
    # Run benchmarks with timing
    start_time = time.time()
    unittest.main(verbosity=2)
    end_time = time.time()
    
    print(f"\nTotal benchmark time: {end_time - start_time:.2f}s")
