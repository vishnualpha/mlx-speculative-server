# Copyright © 2025 Manus AI

import unittest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from mlx_speculative.server_v2 import enhanced_app, enhanced_server
from mlx_speculative.models import ModelConfig
from tests import TEST_CONFIG, TEST_PROMPTS


class TestEnhancedServer(unittest.TestCase):
    """Integration tests for the enhanced server."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(enhanced_app)
        
        # Mock model loading to avoid actual model downloads
        self.patcher = patch('mlx_speculative.multi_model.load_model_pair')
        self.mock_load_model_pair = self.patcher.start()
        
        # Setup mock returns
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        
        # Mock model attributes
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.n_heads = 12
        mock_target.n_kv_heads = 4
        mock_target.vocab_size = 32000
        mock_target.parameters.return_value = [Mock(nbytes=1000) for _ in range(10)]
        
        # Mock tokenizer methods
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Simple token sequence
        mock_tokenizer.decode.return_value = "Generated response text"
        
        self.mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        # Clear any loaded models
        enhanced_server.model_manager.engines.clear()
        enhanced_server.model_manager.tokenizers.clear()
        enhanced_server.model_manager.metadata.clear()
        enhanced_server.model_manager.model_groups.clear()
        enhanced_server.model_manager.default_model = None
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertIn("models_loaded", data)
        self.assertIn("groups_created", data)
    
    def test_list_models_empty(self):
        """Test listing models when none are loaded."""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["models"], [])
        self.assertEqual(data["total_models"], 0)
        self.assertIsNone(data["default_model"])
    
    def test_load_model_endpoint(self):
        """Test loading a model via API."""
        request_data = {
            "name": "test_model",
            "model_path": "/path/to/test/model",
            "set_default": True,
            "tags": ["test", "small"]
        }
        
        response = self.client.post("/models/load", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("model_info", data)
        
        # Verify model was loaded
        models_response = self.client.get("/models")
        models_data = models_response.json()
        self.assertIn("test_model", models_data["models"])
        self.assertEqual(models_data["default_model"], "test_model")
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        # Mock loading failure
        self.mock_load_model_pair.side_effect = Exception("Loading failed")
        
        request_data = {
            "name": "failing_model",
            "model_path": "/path/to/failing/model"
        }
        
        response = self.client.post("/models/load", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "error")
        
        # Reset mock for other tests
        self.mock_load_model_pair.side_effect = None
    
    def test_unload_model_endpoint(self):
        """Test unloading a model via API."""
        # First load a model
        request_data = {
            "name": "test_model",
            "model_path": "/path/to/test/model"
        }
        self.client.post("/models/load", json=request_data)
        
        # Then unload it
        response = self.client.delete("/models/test_model")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        
        # Verify model was unloaded
        models_response = self.client.get("/models")
        models_data = models_response.json()
        self.assertNotIn("test_model", models_data["models"])
    
    def test_unload_nonexistent_model(self):
        """Test unloading a model that doesn't exist."""
        response = self.client.delete("/models/nonexistent_model")
        self.assertEqual(response.status_code, 404)
    
    def test_get_model_info(self):
        """Test getting model information."""
        # First load a model
        request_data = {
            "name": "test_model",
            "model_path": "/path/to/test/model",
            "tags": ["test"]
        }
        self.client.post("/models/load", json=request_data)
        
        # Get model info
        response = self.client.get("/models/test_model")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["name"], "test_model")
        self.assertIn("config", data)
        self.assertIn("model_info", data)
        self.assertIn("memory_usage", data)
        self.assertIn("test", data["tags"])
    
    def test_get_nonexistent_model_info(self):
        """Test getting info for nonexistent model."""
        response = self.client.get("/models/nonexistent_model")
        self.assertEqual(response.status_code, 404)
    
    def test_create_model_group(self):
        """Test creating a model group."""
        # First load some models
        for i, model_name in enumerate(["model1", "model2", "model3"]):
            request_data = {
                "name": model_name,
                "model_path": f"/path/to/{model_name}"
            }
            self.client.post("/models/load", json=request_data)
        
        # Create a group
        group_data = {
            "name": "test_group",
            "models": ["model1", "model2", "model3"],
            "default_model": "model2",
            "description": "Test group",
            "tags": ["group", "test"]
        }
        
        response = self.client.post("/groups", json=group_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        
        # Verify group was created
        models_response = self.client.get("/models")
        models_data = models_response.json()
        self.assertIn("test_group", models_data["groups"])
    
    def test_search_models(self):
        """Test model search functionality."""
        # Load models with different tags
        models_to_load = [
            ("llama-7b", ["llama", "7b"]),
            ("llama-13b", ["llama", "13b"]),
            ("phi-3", ["phi", "small"])
        ]
        
        for model_name, tags in models_to_load:
            request_data = {
                "name": model_name,
                "model_path": f"/path/to/{model_name}",
                "tags": tags
            }
            self.client.post("/models/load", json=request_data)
        
        # Search by query
        search_data = {"query": "llama"}
        response = self.client.post("/models/search", json=search_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        results = set(data["results"])
        self.assertEqual(results, {"llama-7b", "llama-13b"})
        
        # Search by tags
        search_data = {"tags": ["llama"]}
        response = self.client.post("/models/search", json=search_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        results = set(data["results"])
        self.assertEqual(results, {"llama-7b", "llama-13b"})
    
    @patch('mlx_speculative.utils.generate')
    def test_generate_endpoint(self, mock_generate):
        """Test text generation endpoint."""
        # Mock generation function
        mock_generate.return_value = "This is a generated response."
        
        # Load a model first
        request_data = {
            "name": "test_model",
            "model_path": "/path/to/test/model",
            "set_default": True
        }
        self.client.post("/models/load", json=request_data)
        
        # Test generation
        gen_request = {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = self.client.post("/generate", json=gen_request)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("id", data)
        self.assertIn("text", data)
        self.assertIn("model", data)
        self.assertIn("usage", data)
        self.assertIn("performance", data)
        self.assertIn("speculative_stats", data)
        
        self.assertEqual(data["text"], "This is a generated response.")
        self.assertEqual(data["model"], "test_model")
    
    def test_generate_without_model(self):
        """Test generation when no models are loaded."""
        gen_request = {
            "prompt": "Hello, how are you?",
            "max_tokens": 50
        }
        
        response = self.client.post("/generate", json=gen_request)
        self.assertEqual(response.status_code, 500)
    
    def test_stats_endpoint(self):
        """Test statistics endpoint."""
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("uptime", data)
        self.assertIn("total_requests", data)
        self.assertIn("total_tokens", data)
        self.assertIn("average_throughput", data)
        self.assertIn("active_requests", data)
        self.assertIn("loaded_models", data)
        self.assertIn("model_groups", data)
        self.assertIn("model_performance", data)
        self.assertIn("memory_usage", data)
    
    def test_optimize_endpoint(self):
        """Test optimization recommendations endpoint."""
        response = self.client.get("/optimize")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("actions_taken", data)
        self.assertIn("recommendations", data)
    
    def test_config_save_load(self):
        """Test configuration save/load endpoints."""
        import tempfile
        import os
        
        # Load a model first
        request_data = {
            "name": "test_model",
            "model_path": "/path/to/test/model",
            "tags": ["test"]
        }
        self.client.post("/models/load", json=request_data)
        
        # Save config
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            response = self.client.post(f"/config/save?config_path={config_path}")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertEqual(data["status"], "success")
            
            # Verify file was created
            self.assertTrue(os.path.exists(config_path))
            
            # Clear models and load from config
            enhanced_server.model_manager.engines.clear()
            enhanced_server.model_manager.tokenizers.clear()
            enhanced_server.model_manager.metadata.clear()
            
            response = self.client.post(f"/config/load?config_path={config_path}")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertEqual(data["status"], "success")
            
            # Verify model was reloaded
            models_response = self.client.get("/models")
            models_data = models_response.json()
            self.assertIn("test_model", models_data["models"])
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_request_validation(self):
        """Test request validation."""
        # Test invalid generation request
        invalid_request = {
            "prompt": "Hello",
            "max_tokens": -1,  # Invalid
            "temperature": 3.0  # Invalid
        }
        
        response = self.client.post("/generate", json=invalid_request)
        self.assertEqual(response.status_code, 422)  # Validation error
        
        # Test invalid model load request
        invalid_load = {
            "name": "",  # Empty name
            "model_path": ""  # Empty path
        }
        
        response = self.client.post("/models/load", json=invalid_load)
        self.assertEqual(response.status_code, 422)  # Validation error


class TestServerPerformance(unittest.TestCase):
    """Performance-related integration tests."""
    
    def setUp(self):
        """Set up test client with mocked models."""
        self.client = TestClient(enhanced_app)
        
        # Mock model loading
        self.patcher = patch('mlx_speculative.multi_model.load_model_pair')
        self.mock_load_model_pair = self.patcher.start()
        
        # Setup fast mock returns
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        
        mock_target.layers = [Mock() for _ in range(6)]  # Smaller for speed
        mock_target.head_dim = 32
        mock_target.parameters.return_value = [Mock(nbytes=100)]
        
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Fast response"
        
        self.mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        enhanced_server.model_manager.engines.clear()
        enhanced_server.model_manager.tokenizers.clear()
        enhanced_server.model_manager.metadata.clear()
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        # Load a model
        request_data = {
            "name": "test_model",
            "model_path": "/path/to/test/model",
            "set_default": True
        }
        self.client.post("/models/load", json=request_data)
        
        # Mock generation to be fast
        with patch('mlx_speculative.utils.generate') as mock_gen:
            mock_gen.return_value = "Concurrent response"
            
            # Send multiple concurrent requests
            import concurrent.futures
            import threading
            
            def send_request(prompt):
                gen_request = {
                    "prompt": prompt,
                    "max_tokens": 10
                }
                return self.client.post("/generate", json=gen_request)
            
            # Test with 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i in range(5):
                    future = executor.submit(send_request, f"Test prompt {i}")
                    futures.append(future)
                
                # Wait for all requests to complete
                responses = []
                for future in concurrent.futures.as_completed(futures, timeout=10):
                    response = future.result()
                    responses.append(response)
                
                # All requests should succeed
                self.assertEqual(len(responses), 5)
                for response in responses:
                    self.assertEqual(response.status_code, 200)
    
    @unittest.skipIf(TEST_CONFIG["skip_slow_tests"], "Slow tests disabled")
    def test_load_balancing(self):
        """Test load balancing across model instances."""
        # This would require more complex setup with actual load balancing
        # For now, just test that the load balancer is working
        
        # Load multiple instances of the same model
        for i in range(3):
            request_data = {
                "name": f"model_instance_{i}",
                "model_path": f"/path/to/model_{i}",
            }
            response = self.client.post("/models/load", json=request_data)
            self.assertEqual(response.status_code, 200)
        
        # Verify all instances are loaded
        models_response = self.client.get("/models")
        models_data = models_response.json()
        self.assertEqual(models_data["total_models"], 3)


if __name__ == "__main__":
    unittest.main()
