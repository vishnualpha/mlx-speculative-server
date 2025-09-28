# Copyright © 2025 Manus AI

"""
Hugging Face utilities for MLX Speculative Server

This module provides utilities for downloading and loading models from Hugging Face,
including automatic conversion to MLX format and caching.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging

try:
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    from transformers import AutoTokenizer, AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import mlx.core as mx
    from mlx_lm import load, convert
    from mlx_lm.utils import get_model_path
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False

logger = logging.getLogger(__name__)


class HuggingFaceModelManager:
    """Manager for Hugging Face model operations."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the HF model manager.
        
        Args:
            cache_dir: Directory to cache downloaded models. If None, uses default HF cache.
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub and transformers are required for HF model support")
        
        if not MLX_LM_AVAILABLE:
            raise ImportError("mlx and mlx_lm are required for model loading")
        
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        self.mlx_cache_dir = os.path.expanduser("~/.cache/mlx_models")
        os.makedirs(self.mlx_cache_dir, exist_ok=True)
        
        self.api = HfApi()
    
    def is_model_available_locally(self, model_id: str) -> bool:
        """Check if model is available locally in MLX format.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            True if model is available locally in MLX format
        """
        mlx_path = os.path.join(self.mlx_cache_dir, model_id.replace("/", "--"))
        return os.path.exists(mlx_path) and os.path.exists(os.path.join(mlx_path, "config.json"))
    
    def get_local_model_path(self, model_id: str) -> str:
        """Get the local path for a model.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            Local path to the MLX model
        """
        return os.path.join(self.mlx_cache_dir, model_id.replace("/", "--"))
    
    def download_model(
        self,
        model_id: str,
        revision: Optional[str] = None,
        force_download: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """Download a model from Hugging Face.
        
        Args:
            model_id: Hugging Face model identifier
            revision: Model revision/branch to download
            force_download: Force re-download even if cached
            token: HF authentication token
            
        Returns:
            Path to the downloaded model
        """
        logger.info(f"Downloading model {model_id} from Hugging Face...")
        
        try:
            # Download the model
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                force_download=force_download,
                token=token,
            )
            
            logger.info(f"Successfully downloaded {model_id} to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise
    
    def convert_to_mlx(
        self,
        model_id: str,
        hf_path: str,
        quantize: Optional[str] = None,
        force_convert: bool = False,
    ) -> str:
        """Convert a Hugging Face model to MLX format.
        
        Args:
            model_id: Hugging Face model identifier
            hf_path: Path to the Hugging Face model
            quantize: Quantization method (e.g., "q4", "q8")
            force_convert: Force re-conversion even if MLX model exists
            
        Returns:
            Path to the MLX model
        """
        mlx_path = self.get_local_model_path(model_id)
        
        # Check if already converted
        if not force_convert and self.is_model_available_locally(model_id):
            logger.info(f"MLX model {model_id} already exists at {mlx_path}")
            return mlx_path
        
        logger.info(f"Converting {model_id} to MLX format...")
        
        try:
            # Create MLX model directory
            os.makedirs(mlx_path, exist_ok=True)
            
            # Convert using mlx_lm
            convert(
                hf_path=hf_path,
                mlx_path=mlx_path,
                quantize=quantize,
            )
            
            logger.info(f"Successfully converted {model_id} to MLX format at {mlx_path}")
            return mlx_path
            
        except Exception as e:
            logger.error(f"Failed to convert model {model_id} to MLX: {e}")
            # Clean up partial conversion
            if os.path.exists(mlx_path):
                shutil.rmtree(mlx_path)
            raise
    
    def load_model_and_tokenizer(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        force_download: bool = False,
        force_convert: bool = False,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> Tuple[Any, Any]:
        """Load a model and tokenizer from Hugging Face.
        
        Args:
            model_id: Hugging Face model identifier
            revision: Model revision/branch
            quantize: Quantization method
            force_download: Force re-download
            force_convert: Force re-conversion
            token: HF authentication token
            trust_remote_code: Trust remote code execution
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Check if model is already available locally in MLX format
            if not force_convert and self.is_model_available_locally(model_id):
                logger.info(f"Loading cached MLX model {model_id}")
                mlx_path = self.get_local_model_path(model_id)
                model, tokenizer = load(mlx_path)
                return model, tokenizer
            
            # Download from HF if needed
            if force_download or not self._is_hf_model_cached(model_id):
                hf_path = self.download_model(
                    model_id=model_id,
                    revision=revision,
                    force_download=force_download,
                    token=token,
                )
            else:
                # Use cached HF model
                hf_path = self._get_hf_model_path(model_id)
                logger.info(f"Using cached HF model at {hf_path}")
            
            # Convert to MLX format
            mlx_path = self.convert_to_mlx(
                model_id=model_id,
                hf_path=hf_path,
                quantize=quantize,
                force_convert=force_convert,
            )
            
            # Load the MLX model
            logger.info(f"Loading MLX model from {mlx_path}")
            model, tokenizer = load(mlx_path)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def _is_hf_model_cached(self, model_id: str) -> bool:
        """Check if HF model is cached locally."""
        try:
            # Try to get the model path without downloading
            model_path = get_model_path(model_id, check_local=True)
            return model_path is not None
        except:
            return False
    
    def _get_hf_model_path(self, model_id: str) -> str:
        """Get the local HF model path."""
        return get_model_path(model_id, check_local=True)
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a model from Hugging Face.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            Model information dictionary
        """
        try:
            # Get model info from HF API
            model_info = self.api.model_info(model_id)
            
            # Try to get config
            config = None
            try:
                config = AutoConfig.from_pretrained(model_id)
                config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
            except:
                config_dict = {}
            
            return {
                "model_id": model_id,
                "downloads": getattr(model_info, 'downloads', 0),
                "likes": getattr(model_info, 'likes', 0),
                "tags": getattr(model_info, 'tags', []),
                "pipeline_tag": getattr(model_info, 'pipeline_tag', None),
                "library_name": getattr(model_info, 'library_name', None),
                "config": config_dict,
                "is_local": self.is_model_available_locally(model_id),
                "local_path": self.get_local_model_path(model_id) if self.is_model_available_locally(model_id) else None,
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            return {
                "model_id": model_id,
                "error": str(e),
                "is_local": self.is_model_available_locally(model_id),
                "local_path": self.get_local_model_path(model_id) if self.is_model_available_locally(model_id) else None,
            }
    
    def list_local_models(self) -> List[Dict[str, Any]]:
        """List all locally cached MLX models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        if not os.path.exists(self.mlx_cache_dir):
            return models
        
        for model_dir in os.listdir(self.mlx_cache_dir):
            model_path = os.path.join(self.mlx_cache_dir, model_dir)
            
            if not os.path.isdir(model_path):
                continue
            
            # Check if it's a valid MLX model
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                continue
            
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Convert directory name back to model ID
                model_id = model_dir.replace("--", "/")
                
                models.append({
                    "model_id": model_id,
                    "local_path": model_path,
                    "config": config,
                    "size_mb": self._get_directory_size(model_path) / (1024 * 1024),
                })
                
            except Exception as e:
                logger.warning(f"Failed to read config for {model_dir}: {e}")
                continue
        
        return models
    
    def _get_directory_size(self, path: str) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def delete_local_model(self, model_id: str) -> bool:
        """Delete a locally cached MLX model.
        
        Args:
            model_id: Hugging Face model identifier
            
        Returns:
            True if successfully deleted
        """
        mlx_path = self.get_local_model_path(model_id)
        
        if not os.path.exists(mlx_path):
            logger.warning(f"Model {model_id} not found locally")
            return False
        
        try:
            shutil.rmtree(mlx_path)
            logger.info(f"Successfully deleted local model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def search_models(
        self,
        query: Optional[str] = None,
        task: Optional[str] = None,
        library: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face.
        
        Args:
            query: Search query
            task: Task type (e.g., "text-generation")
            library: Library name (e.g., "transformers")
            tags: List of tags to filter by
            limit: Maximum number of results
            
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.api.list_models(
                search=query,
                task=task,
                library=library,
                tags=tags,
                limit=limit,
            )
            
            results = []
            for model in models:
                results.append({
                    "model_id": model.modelId,
                    "downloads": getattr(model, 'downloads', 0),
                    "likes": getattr(model, 'likes', 0),
                    "tags": getattr(model, 'tags', []),
                    "pipeline_tag": getattr(model, 'pipeline_tag', None),
                    "library_name": getattr(model, 'library_name', None),
                    "is_local": self.is_model_available_locally(model.modelId),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []


# Global HF manager instance
_hf_manager = None


def get_hf_manager(cache_dir: Optional[str] = None) -> HuggingFaceModelManager:
    """Get the global HF manager instance."""
    global _hf_manager
    if _hf_manager is None:
        _hf_manager = HuggingFaceModelManager(cache_dir=cache_dir)
    return _hf_manager


def load_hf_model(
    model_id: str,
    revision: Optional[str] = None,
    quantize: Optional[str] = None,
    force_download: bool = False,
    force_convert: bool = False,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[Any, Any]:
    """Convenience function to load a model from Hugging Face.
    
    Args:
        model_id: Hugging Face model identifier
        revision: Model revision/branch
        quantize: Quantization method
        force_download: Force re-download
        force_convert: Force re-conversion
        token: HF authentication token
        trust_remote_code: Trust remote code execution
        
    Returns:
        Tuple of (model, tokenizer)
    """
    hf_manager = get_hf_manager()
    return hf_manager.load_model_and_tokenizer(
        model_id=model_id,
        revision=revision,
        quantize=quantize,
        force_download=force_download,
        force_convert=force_convert,
        token=token,
        trust_remote_code=trust_remote_code,
    )


def is_hf_model_id(model_path: str) -> bool:
    """Check if a model path is a Hugging Face model ID.
    
    Args:
        model_path: Model path or ID
        
    Returns:
        True if it looks like a HF model ID
    """
    # Simple heuristic: if it contains "/" and doesn't start with "/", ".", or "~"
    # it's likely a HF model ID
    if "/" in model_path and not model_path.startswith(("/", ".", "~")):
        return True
    
    # Also check if it's not a local path
    return not os.path.exists(model_path)


def resolve_model_path(model_path: str, **kwargs) -> Tuple[str, bool]:
    """Resolve a model path, downloading from HF if necessary.
    
    Args:
        model_path: Model path or Hugging Face model ID
        **kwargs: Additional arguments for HF loading
        
    Returns:
        Tuple of (resolved_path, is_hf_model)
    """
    if is_hf_model_id(model_path):
        # It's a HF model ID, get the local MLX path
        hf_manager = get_hf_manager()
        
        # Load the model (this will download and convert if needed)
        model, tokenizer = hf_manager.load_model_and_tokenizer(model_path, **kwargs)
        
        # Return the local MLX path
        local_path = hf_manager.get_local_model_path(model_path)
        return local_path, True
    else:
        # It's a local path
        return model_path, False
