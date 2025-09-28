# Copyright © 2025 Manus AI

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os

from mlx_speculative.multi_model import (
    MultiModelManager, ModelMetadata, ModelGroup, ModelLoadBalancer
)
from mlx_speculative.models import ModelConfig
from tests import TEST_CONFIG


class TestModelLoadBalancer(unittest.TestCase):
    """Test the ModelLoadBalancer class."""
    
    def setUp(self):
        self.balancer = ModelLoadBalancer()
    
    def test_register_instance(self):
        """Test instance registration."""
        self.balancer.register_instance("model1", "instance1")
        self.balancer.register_instance("model1", "instance2")
        
        self.assertIn("model1", self.balancer.model_instances)
        self.assertEqual(len(self.balancer.model_instances["model1"]), 2)
        self.assertIn("instance1", self.balancer.model_instances["model1"])
        self.assertIn("instance2", self.balancer.model_instances["model1"])
    
    def test_round_robin_strategy(self):
        """Test round-robin load balancing."""
        self.balancer.register_instance("model1", "instance1")
        self.balancer.register_instance("model1", "instance2")
        self.balancer.register_instance("model1", "instance3")
        
        # Test round-robin selection
        instances = []
        for _ in range(6):  # 2 full rounds
            instance = self.balancer.get_best_instance("model1", "round_robin")
            instances.append(instance)
        
        # Should cycle through instances
        expected = ["instance1", "instance2", "instance3"] * 2
        self.assertEqual(instances, expected)
    
    def test_least_loaded_strategy(self):
        """Test least-loaded load balancing."""
        self.balancer.register_instance("model1", "instance1")
        self.balancer.register_instance("model1", "instance2")
        
        # Set different loads
        self.balancer.update_load("instance1", 5)
        self.balancer.update_load("instance2", 2)
        
        # Should select least loaded
        instance = self.balancer.get_best_instance("model1", "least_loaded")
        self.assertEqual(instance, "instance2")
    
    def test_update_load(self):
        """Test load updating."""
        self.balancer.register_instance("model1", "instance1")
        
        # Initial load should be 0
        self.assertEqual(self.balancer.instance_loads["instance1"], 0)
        
        # Update load
        self.balancer.update_load("instance1", 3)
        self.assertEqual(self.balancer.instance_loads["instance1"], 3)
        
        # Decrease load
        self.balancer.update_load("instance1", -1)
        self.assertEqual(self.balancer.instance_loads["instance1"], 2)
        
        # Should not go below 0
        self.balancer.update_load("instance1", -10)
        self.assertEqual(self.balancer.instance_loads["instance1"], 0)


class TestModelMetadata(unittest.TestCase):
    """Test the ModelMetadata class."""
    
    def test_initialization(self):
        """Test metadata initialization."""
        config = ModelConfig(model_path="/path/to/model")
        metadata = ModelMetadata(
            name="test_model",
            config=config,
            load_time=1.5,
            memory_usage={"total": 1000.0},
            model_info={"num_layers": 12},
            last_used=1234567890.0,
            tags={"llm", "chat"}
        )
        
        self.assertEqual(metadata.name, "test_model")
        self.assertEqual(metadata.config, config)
        self.assertEqual(metadata.load_time, 1.5)
        self.assertEqual(metadata.memory_usage["total"], 1000.0)
        self.assertEqual(metadata.model_info["num_layers"], 12)
        self.assertEqual(metadata.last_used, 1234567890.0)
        self.assertEqual(metadata.tags, {"llm", "chat"})
        self.assertEqual(metadata.usage_count, 0)
        self.assertFalse(metadata.is_default)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelConfig(model_path="/path/to/model")
        metadata = ModelMetadata(
            name="test_model",
            config=config,
            load_time=1.5,
            memory_usage={"total": 1000.0},
            model_info={"num_layers": 12},
            last_used=1234567890.0,
            tags={"llm", "chat"}
        )
        
        data = metadata.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data["name"], "test_model")
        self.assertIsInstance(data["tags"], list)
        self.assertIn("llm", data["tags"])
        self.assertIn("chat", data["tags"])


class TestModelGroup(unittest.TestCase):
    """Test the ModelGroup class."""
    
    def test_initialization(self):
        """Test group initialization."""
        group = ModelGroup(
            name="llama_family",
            models=["llama-7b", "llama-13b", "llama-70b"],
            default_model="llama-13b",
            description="Llama model family",
            tags={"llama", "meta"}
        )
        
        self.assertEqual(group.name, "llama_family")
        self.assertEqual(group.models, ["llama-7b", "llama-13b", "llama-70b"])
        self.assertEqual(group.default_model, "llama-13b")
        self.assertEqual(group.description, "Llama model family")
        self.assertEqual(group.tags, {"llama", "meta"})


class TestMultiModelManager(unittest.TestCase):
    """Test the MultiModelManager class."""
    
    def setUp(self):
        self.manager = MultiModelManager(
            max_models=5,
            memory_limit_gb=8.0,
            auto_unload=True,
            load_balancing=True,
        )
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.max_models, 5)
        self.assertEqual(self.manager.memory_limit_gb, 8.0)
        self.assertTrue(self.manager.auto_unload)
        self.assertTrue(self.manager.load_balancing)
        self.assertIsNotNone(self.manager.load_balancer)
        self.assertIsNone(self.manager.default_model)
    
    @patch('mlx_speculative.multi_model.load_model_pair')
    def test_load_model_success(self, mock_load_model_pair):
        """Test successful model loading."""
        # Mock the model loading
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
        
        # Mock model attributes
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.n_heads = 12
        mock_target.n_kv_heads = 4
        mock_target.vocab_size = 32000
        mock_target.parameters.return_value = [Mock(nbytes=1000) for _ in range(10)]
        
        config = ModelConfig(model_path="/path/to/model")
        
        success = self.manager.load_model("test_model", config, set_default=True)
        
        self.assertTrue(success)
        self.assertIn("test_model", self.manager.engines)
        self.assertIn("test_model", self.manager.tokenizers)
        self.assertIn("test_model", self.manager.metadata)
        self.assertEqual(self.manager.default_model, "test_model")
    
    @patch('mlx_speculative.multi_model.load_model_pair')
    def test_load_model_failure(self, mock_load_model_pair):
        """Test model loading failure."""
        # Mock loading failure
        mock_load_model_pair.side_effect = Exception("Loading failed")
        
        config = ModelConfig(model_path="/path/to/model")
        
        success = self.manager.load_model("test_model", config)
        
        self.assertFalse(success)
        self.assertNotIn("test_model", self.manager.engines)
    
    @patch('mlx_speculative.multi_model.load_model_pair')
    def test_unload_model(self, mock_load_model_pair):
        """Test model unloading."""
        # Setup: load a model first
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
        
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.parameters.return_value = [Mock(nbytes=1000)]
        
        config = ModelConfig(model_path="/path/to/model")
        self.manager.load_model("test_model", config)
        
        # Test unloading
        success = self.manager.unload_model("test_model")
        
        self.assertTrue(success)
        self.assertNotIn("test_model", self.manager.engines)
        self.assertNotIn("test_model", self.manager.tokenizers)
        self.assertNotIn("test_model", self.manager.metadata)
    
    def test_unload_nonexistent_model(self):
        """Test unloading a model that doesn't exist."""
        success = self.manager.unload_model("nonexistent_model")
        self.assertFalse(success)
    
    @patch('mlx_speculative.multi_model.load_model_pair')
    def test_get_model(self, mock_load_model_pair):
        """Test getting a model."""
        # Setup: load a model first
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
        
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.parameters.return_value = [Mock(nbytes=1000)]
        
        config = ModelConfig(model_path="/path/to/model")
        self.manager.load_model("test_model", config)
        
        # Test getting the model
        engine, tokenizer, actual_name = self.manager.get_model("test_model")
        
        self.assertIsNotNone(engine)
        self.assertEqual(tokenizer, mock_tokenizer)
        self.assertEqual(actual_name, "test_model")
        
        # Check that usage stats were updated
        metadata = self.manager.metadata["test_model"]
        self.assertEqual(metadata.usage_count, 1)
        self.assertEqual(self.manager.total_requests, 1)
    
    def test_get_nonexistent_model(self):
        """Test getting a model that doesn't exist."""
        with self.assertRaises(ValueError):
            self.manager.get_model("nonexistent_model")
    
    @patch('mlx_speculative.multi_model.load_model_pair')
    def test_create_model_group(self, mock_load_model_pair):
        """Test creating a model group."""
        # Setup: load some models first
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
        
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.parameters.return_value = [Mock(nbytes=1000)]
        
        config = ModelConfig(model_path="/path/to/model")
        
        # Load multiple models
        for model_name in ["model1", "model2", "model3"]:
            self.manager.load_model(model_name, config)
        
        # Create a group
        success = self.manager.create_model_group(
            group_name="test_group",
            model_names=["model1", "model2", "model3"],
            default_model="model2",
            description="Test group",
            tags={"test"}
        )
        
        self.assertTrue(success)
        self.assertIn("test_group", self.manager.model_groups)
        
        group = self.manager.model_groups["test_group"]
        self.assertEqual(group.name, "test_group")
        self.assertEqual(group.models, ["model1", "model2", "model3"])
        self.assertEqual(group.default_model, "model2")
    
    def test_create_group_with_nonexistent_model(self):
        """Test creating a group with nonexistent models."""
        success = self.manager.create_model_group(
            group_name="test_group",
            model_names=["nonexistent1", "nonexistent2"],
            default_model="nonexistent1"
        )
        
        self.assertFalse(success)
        self.assertNotIn("test_group", self.manager.model_groups)
    
    @patch('mlx_speculative.multi_model.load_model_pair')
    def test_search_models(self, mock_load_model_pair):
        """Test model searching."""
        # Setup: load models with different tags
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
        
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.parameters.return_value = [Mock(nbytes=1000)]
        
        config = ModelConfig(model_path="/path/to/model")
        
        # Load models with different tags
        self.manager.load_model("llama-7b", config, tags={"llama", "7b"})
        self.manager.load_model("llama-13b", config, tags={"llama", "13b"})
        self.manager.load_model("phi-3", config, tags={"phi", "small"})
        
        # Test search by query
        results = self.manager.search_models(query="llama")
        self.assertEqual(set(results), {"llama-7b", "llama-13b"})
        
        # Test search by tags
        results = self.manager.search_models(tags={"llama"})
        self.assertEqual(set(results), {"llama-7b", "llama-13b"})
        
        # Test search by specific tag
        results = self.manager.search_models(tags={"7b"})
        self.assertEqual(results, ["llama-7b"])
    
    def test_list_models(self):
        """Test listing models."""
        result = self.manager.list_models()
        
        self.assertIn("models", result)
        self.assertIn("default_model", result)
        self.assertIn("total_models", result)
        self.assertIn("groups", result)
        self.assertIn("total_groups", result)
        
        self.assertEqual(result["total_models"], 0)
        self.assertEqual(result["total_groups"], 0)
        self.assertIsNone(result["default_model"])
    
    def test_save_and_load_config(self):
        """Test configuration saving and loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save empty config
            self.manager.save_config(config_path)
            
            # Verify file was created and has valid JSON
            self.assertTrue(os.path.exists(config_path))
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.assertIn("models", config_data)
            self.assertIn("groups", config_data)
            self.assertIn("settings", config_data)
            
            # Test loading
            new_manager = MultiModelManager()
            success = new_manager.load_config(config_path)
            self.assertTrue(success)
            
        finally:
            # Clean up
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    @patch('mlx_speculative.multi_model.load_model_pair')
    def test_performance_stats(self, mock_load_model_pair):
        """Test performance statistics."""
        # Setup: load a model and use it
        mock_target = Mock()
        mock_draft = Mock()
        mock_tokenizer = Mock()
        mock_load_model_pair.return_value = (mock_target, mock_draft, mock_tokenizer)
        
        mock_target.layers = [Mock() for _ in range(12)]
        mock_target.head_dim = 64
        mock_target.parameters.return_value = [Mock(nbytes=1000)]
        
        config = ModelConfig(model_path="/path/to/model")
        self.manager.load_model("test_model", config)
        
        # Use the model
        self.manager.get_model("test_model")
        
        # Get stats
        stats = self.manager.get_performance_stats()
        
        self.assertIn("total_requests", stats)
        self.assertIn("models", stats)
        self.assertIn("memory_usage", stats)
        self.assertIn("load_balancing", stats)
        
        self.assertEqual(stats["total_requests"], 1)
        self.assertIn("test_model", stats["models"])
        
        model_stats = stats["models"]["test_model"]
        self.assertEqual(model_stats["usage_count"], 1)
        self.assertEqual(model_stats["request_count"], 1)


if __name__ == "__main__":
    unittest.main()
