# Copyright © 2025 Manus AI

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import threading

import mlx.core as mx
import mlx.nn as nn

from .core import SpeculativeEngine
from .models import ModelConfig, load_model_pair, get_model_info, estimate_memory_usage
from .sample_utils import create_sampler


@dataclass
class ModelMetadata:
    """Metadata for a loaded model."""
    name: str
    config: ModelConfig
    load_time: float
    memory_usage: Dict[str, float]
    model_info: Dict[str, Any]
    last_used: float
    usage_count: int = 0
    is_default: bool = False
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['tags'] = list(self.tags)  # Convert set to list for JSON serialization
        return data


@dataclass
class ModelGroup:
    """A group of related models (e.g., different sizes of the same family)."""
    name: str
    models: List[str]
    default_model: str
    description: str = ""
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()


class ModelLoadBalancer:
    """Load balancer for distributing requests across multiple model instances."""
    
    def __init__(self):
        self.model_instances: Dict[str, List[str]] = {}  # model_name -> [instance_ids]
        self.instance_loads: Dict[str, int] = {}  # instance_id -> current_load
        self.round_robin_counters: Dict[str, int] = {}
    
    def register_instance(self, model_name: str, instance_id: str):
        """Register a model instance."""
        if model_name not in self.model_instances:
            self.model_instances[model_name] = []
            self.round_robin_counters[model_name] = 0
        
        self.model_instances[model_name].append(instance_id)
        self.instance_loads[instance_id] = 0
    
    def get_best_instance(self, model_name: str, strategy: str = "round_robin") -> Optional[str]:
        """Get the best instance for a model based on load balancing strategy."""
        if model_name not in self.model_instances or not self.model_instances[model_name]:
            return None
        
        instances = self.model_instances[model_name]
        
        if strategy == "round_robin":
            counter = self.round_robin_counters[model_name]
            instance = instances[counter % len(instances)]
            self.round_robin_counters[model_name] = (counter + 1) % len(instances)
            return instance
        
        elif strategy == "least_loaded":
            return min(instances, key=lambda x: self.instance_loads.get(x, 0))
        
        else:
            # Default to first instance
            return instances[0]
    
    def update_load(self, instance_id: str, load_delta: int):
        """Update the load for an instance."""
        if instance_id in self.instance_loads:
            self.instance_loads[instance_id] = max(0, self.instance_loads[instance_id] + load_delta)


class MultiModelManager:
    """
    Advanced multi-model manager with support for model groups, load balancing,
    automatic scaling, and intelligent model selection.
    """
    
    def __init__(
        self,
        max_models: int = 10,
        memory_limit_gb: Optional[float] = None,
        auto_unload: bool = True,
        load_balancing: bool = True,
    ):
        self.max_models = max_models
        self.memory_limit_gb = memory_limit_gb
        self.auto_unload = auto_unload
        self.load_balancing = load_balancing
        
        # Model storage
        self.engines: Dict[str, SpeculativeEngine] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.model_groups: Dict[str, ModelGroup] = {}
        
        # Load balancing
        self.load_balancer = ModelLoadBalancer() if load_balancing else None
        
        # Threading and async support
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.load_lock = threading.RLock()
        
        # Default model
        self.default_model: Optional[str] = None
        
        # Performance tracking
        self.total_requests = 0
        self.model_request_counts: Dict[str, int] = {}
        
    def load_model(
        self,
        name: str,
        config: ModelConfig,
        auto_draft: bool = True,
        draft_layers_ratio: float = 0.5,
        set_default: bool = False,
        tags: Optional[Set[str]] = None,
        force: bool = False,
    ) -> bool:
        """
        Load a model with advanced options.
        
        Args:
            name: Model name
            config: Model configuration
            auto_draft: Whether to create automatic draft model
            draft_layers_ratio: Ratio for draft model creation
            set_default: Whether to set as default model
            tags: Tags for model categorization
            force: Force loading even if limits are exceeded
            
        Returns:
            True if model was loaded successfully
        """
        with self.load_lock:
            # Check if model already exists
            if name in self.engines and not force:
                print(f"Model '{name}' already loaded")
                return True
            
            # Check limits
            if not force and not self._can_load_model():
                if self.auto_unload:
                    self._make_space_for_model()
                else:
                    print(f"Cannot load model '{name}': limits exceeded")
                    return False
            
            try:
                start_time = time.time()
                
                # Load model pair
                target_model, draft_model, tokenizer = load_model_pair(
                    config, auto_draft=auto_draft, draft_layers_ratio=draft_layers_ratio
                )
                
                # Create engine
                engine = SpeculativeEngine(
                    target_model=target_model,
                    draft_model=draft_model,
                    num_draft_tokens=4,
                    auto_draft=auto_draft,
                )
                
                load_time = time.time() - start_time
                
                # Get model information
                model_info = get_model_info(target_model)
                memory_usage = estimate_memory_usage(target_model)
                
                # Store model
                self.engines[name] = engine
                self.tokenizers[name] = tokenizer
                
                # Create metadata
                metadata = ModelMetadata(
                    name=name,
                    config=config,
                    load_time=load_time,
                    memory_usage=memory_usage,
                    model_info=model_info,
                    last_used=time.time(),
                    tags=tags or set(),
                )
                self.metadata[name] = metadata
                
                # Set default if requested or if first model
                if set_default or self.default_model is None:
                    self.default_model = name
                    metadata.is_default = True
                
                # Register with load balancer
                if self.load_balancer:
                    self.load_balancer.register_instance(name, name)
                
                # Initialize request count
                self.model_request_counts[name] = 0
                
                print(f"Successfully loaded model '{name}' in {load_time:.2f}s")
                print(f"Memory usage: {memory_usage['total']:.1f} MB")
                
                return True
                
            except Exception as e:
                print(f"Failed to load model '{name}': {e}")
                return False
    
    async def load_model_async(self, *args, **kwargs) -> bool:
        """Asynchronously load a model."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.load_model, *args, **kwargs)
    
    def unload_model(self, name: str) -> bool:
        """Unload a model from memory."""
        with self.load_lock:
            if name not in self.engines:
                print(f"Model '{name}' not found")
                return False
            
            try:
                # Remove from storage
                del self.engines[name]
                del self.tokenizers[name]
                del self.metadata[name]
                
                # Update default if necessary
                if self.default_model == name:
                    remaining_models = list(self.engines.keys())
                    self.default_model = remaining_models[0] if remaining_models else None
                
                # Clean up request count
                if name in self.model_request_counts:
                    del self.model_request_counts[name]
                
                print(f"Unloaded model '{name}'")
                return True
                
            except Exception as e:
                print(f"Failed to unload model '{name}': {e}")
                return False
    
    def get_model(self, name: Optional[str] = None) -> Tuple[SpeculativeEngine, Any, str]:
        """
        Get a model engine and tokenizer with load balancing.
        
        Returns:
            Tuple of (engine, tokenizer, actual_model_name)
        """
        # Determine model name
        if name is None:
            name = self.default_model
        
        if name is None:
            raise ValueError("No models loaded and no default model set")
        
        # Check if it's a model group
        if name in self.model_groups:
            group = self.model_groups[name]
            name = group.default_model
        
        # Use load balancer if available
        if self.load_balancer:
            actual_name = self.load_balancer.get_best_instance(name) or name
        else:
            actual_name = name
        
        if actual_name not in self.engines:
            raise ValueError(f"Model '{actual_name}' not found")
        
        # Update usage statistics
        self.metadata[actual_name].last_used = time.time()
        self.metadata[actual_name].usage_count += 1
        self.model_request_counts[actual_name] = self.model_request_counts.get(actual_name, 0) + 1
        self.total_requests += 1
        
        return self.engines[actual_name], self.tokenizers[actual_name], actual_name
    
    def create_model_group(
        self,
        group_name: str,
        model_names: List[str],
        default_model: str,
        description: str = "",
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """Create a model group for related models."""
        # Validate that all models exist
        for model_name in model_names:
            if model_name not in self.engines:
                print(f"Model '{model_name}' not found, cannot create group")
                return False
        
        if default_model not in model_names:
            print(f"Default model '{default_model}' not in model list")
            return False
        
        group = ModelGroup(
            name=group_name,
            models=model_names,
            default_model=default_model,
            description=description,
            tags=tags or set(),
        )
        
        self.model_groups[group_name] = group
        print(f"Created model group '{group_name}' with {len(model_names)} models")
        return True
    
    def list_models(self, include_groups: bool = True) -> Dict[str, Any]:
        """List all loaded models and groups."""
        result = {
            "models": list(self.engines.keys()),
            "default_model": self.default_model,
            "total_models": len(self.engines),
        }
        
        if include_groups:
            result["groups"] = list(self.model_groups.keys())
            result["total_groups"] = len(self.model_groups)
        
        return result
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        if name in self.metadata:
            return self.metadata[name].to_dict()
        elif name in self.model_groups:
            group = self.model_groups[name]
            return {
                "type": "group",
                "name": group.name,
                "models": group.models,
                "default_model": group.default_model,
                "description": group.description,
                "tags": list(group.tags),
            }
        else:
            raise ValueError(f"Model or group '{name}' not found")
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        model_type: Optional[str] = None,
    ) -> List[str]:
        """Search for models based on criteria."""
        results = []
        
        for name, metadata in self.metadata.items():
            # Check query match
            if query and query.lower() not in name.lower():
                continue
            
            # Check tags
            if tags and not tags.intersection(metadata.tags):
                continue
            
            # Check model type (could be based on model_info)
            if model_type:
                # This could be extended to check model architecture, size, etc.
                pass
            
            results.append(name)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models."""
        stats = {
            "total_requests": self.total_requests,
            "models": {},
            "memory_usage": {},
            "load_balancing": {},
        }
        
        total_memory = 0
        for name, metadata in self.metadata.items():
            model_stats = {
                "usage_count": metadata.usage_count,
                "last_used": metadata.last_used,
                "load_time": metadata.load_time,
                "request_count": self.model_request_counts.get(name, 0),
                "memory_mb": metadata.memory_usage.get("total", 0),
            }
            stats["models"][name] = model_stats
            total_memory += model_stats["memory_mb"]
        
        stats["memory_usage"]["total_mb"] = total_memory
        stats["memory_usage"]["limit_mb"] = (self.memory_limit_gb * 1024) if self.memory_limit_gb else None
        
        if self.load_balancer:
            stats["load_balancing"]["instance_loads"] = dict(self.load_balancer.instance_loads)
            stats["load_balancing"]["model_instances"] = dict(self.load_balancer.model_instances)
        
        return stats
    
    def optimize_models(self) -> Dict[str, Any]:
        """Optimize model loading based on usage patterns."""
        optimization_report = {
            "actions_taken": [],
            "recommendations": [],
        }
        
        # Find least used models
        if len(self.engines) > self.max_models // 2:
            sorted_models = sorted(
                self.metadata.items(),
                key=lambda x: (x[1].last_used, x[1].usage_count)
            )
            
            for name, metadata in sorted_models[:2]:  # Consider unloading 2 least used
                if not metadata.is_default and metadata.usage_count < 5:
                    optimization_report["recommendations"].append(
                        f"Consider unloading '{name}' (used {metadata.usage_count} times)"
                    )
        
        # Memory optimization
        if self.memory_limit_gb:
            total_memory = sum(m.memory_usage.get("total", 0) for m in self.metadata.values())
            memory_limit_mb = self.memory_limit_gb * 1024
            
            if total_memory > memory_limit_mb * 0.8:  # 80% threshold
                optimization_report["recommendations"].append(
                    f"Memory usage high: {total_memory:.1f}MB / {memory_limit_mb:.1f}MB"
                )
        
        return optimization_report
    
    def _can_load_model(self) -> bool:
        """Check if a new model can be loaded based on limits."""
        # Check model count limit
        if len(self.engines) >= self.max_models:
            return False
        
        # Check memory limit (simplified check)
        if self.memory_limit_gb:
            total_memory = sum(m.memory_usage.get("total", 0) for m in self.metadata.values())
            memory_limit_mb = self.memory_limit_gb * 1024
            
            if total_memory > memory_limit_mb * 0.9:  # 90% threshold
                return False
        
        return True
    
    def _make_space_for_model(self):
        """Make space for a new model by unloading least used models."""
        if not self.metadata:
            return
        
        # Sort by usage (least used first)
        sorted_models = sorted(
            self.metadata.items(),
            key=lambda x: (x[1].is_default, x[1].last_used, x[1].usage_count)
        )
        
        # Unload the least used non-default model
        for name, metadata in sorted_models:
            if not metadata.is_default:
                print(f"Auto-unloading '{name}' to make space")
                self.unload_model(name)
                break
    
    def save_config(self, config_path: str):
        """Save current model configuration to file."""
        config = {
            "models": {},
            "groups": {},
            "settings": {
                "max_models": self.max_models,
                "memory_limit_gb": self.memory_limit_gb,
                "auto_unload": self.auto_unload,
                "load_balancing": self.load_balancing,
                "default_model": self.default_model,
            }
        }
        
        # Save model configurations
        for name, metadata in self.metadata.items():
            config["models"][name] = {
                "config": asdict(metadata.config),
                "tags": list(metadata.tags),
                "is_default": metadata.is_default,
            }
        
        # Save model groups
        for name, group in self.model_groups.items():
            config["groups"][name] = {
                "models": group.models,
                "default_model": group.default_model,
                "description": group.description,
                "tags": list(group.tags),
            }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
    
    def load_config(self, config_path: str) -> bool:
        """Load model configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load settings
            settings = config.get("settings", {})
            self.max_models = settings.get("max_models", self.max_models)
            self.memory_limit_gb = settings.get("memory_limit_gb", self.memory_limit_gb)
            self.auto_unload = settings.get("auto_unload", self.auto_unload)
            self.load_balancing = settings.get("load_balancing", self.load_balancing)
            
            # Load models
            models_config = config.get("models", {})
            for name, model_data in models_config.items():
                model_config = ModelConfig(**model_data["config"])
                tags = set(model_data.get("tags", []))
                is_default = model_data.get("is_default", False)
                
                self.load_model(
                    name=name,
                    config=model_config,
                    tags=tags,
                    set_default=is_default,
                )
            
            # Load groups
            groups_config = config.get("groups", {})
            for name, group_data in groups_config.items():
                self.create_model_group(
                    group_name=name,
                    model_names=group_data["models"],
                    default_model=group_data["default_model"],
                    description=group_data.get("description", ""),
                    tags=set(group_data.get("tags", [])),
                )
            
            print(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return False
