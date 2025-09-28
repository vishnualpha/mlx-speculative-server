# Copyright © 2025 Manus AI

"""
Hugging Face model management endpoints for MLX Speculative Server
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .hf_utils import get_hf_manager, HF_AVAILABLE

# Create router for HF endpoints
hf_router = APIRouter(prefix="/hf", tags=["huggingface"])


class HFModelSearchRequest(BaseModel):
    """Request model for HF model search."""
    query: Optional[str] = Field(None, description="Search query")
    task: Optional[str] = Field(None, description="Task type (e.g., 'text-generation')")
    library: Optional[str] = Field(None, description="Library name (e.g., 'transformers')")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")


class HFModelDownloadRequest(BaseModel):
    """Request model for HF model download."""
    model_id: str = Field(..., description="Hugging Face model identifier")
    revision: Optional[str] = Field(None, description="Model revision/branch")
    quantize: Optional[str] = Field(None, description="Quantization method (e.g., 'q4', 'q8')")
    force_download: bool = Field(False, description="Force re-download")
    force_convert: bool = Field(False, description="Force re-conversion to MLX")
    trust_remote_code: bool = Field(False, description="Trust remote code execution")


@hf_router.get("/available")
async def check_hf_availability():
    """Check if Hugging Face integration is available."""
    return {
        "available": HF_AVAILABLE,
        "message": "Hugging Face integration is available" if HF_AVAILABLE else 
                  "Hugging Face integration requires: pip install huggingface_hub transformers"
    }


@hf_router.get("/models/local")
async def list_local_hf_models():
    """List all locally cached MLX models from Hugging Face."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        models = hf_manager.list_local_models()
        
        return {
            "models": models,
            "total": len(models),
            "cache_dir": hf_manager.mlx_cache_dir,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list local models: {e}")


@hf_router.post("/models/search")
async def search_hf_models(request: HFModelSearchRequest):
    """Search for models on Hugging Face."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        results = hf_manager.search_models(
            query=request.query,
            task=request.task,
            library=request.library,
            tags=request.tags,
            limit=request.limit,
        )
        
        return {
            "results": results,
            "total": len(results),
            "query": request.query,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search models: {e}")


@hf_router.get("/models/{model_id:path}/info")
async def get_hf_model_info(model_id: str):
    """Get information about a Hugging Face model."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        info = hf_manager.get_model_info(model_id)
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")


@hf_router.post("/models/download")
async def download_hf_model(request: HFModelDownloadRequest):
    """Download and convert a model from Hugging Face."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        
        # Check if already available locally
        if not request.force_download and not request.force_convert:
            if hf_manager.is_model_available_locally(request.model_id):
                return {
                    "status": "already_available",
                    "message": f"Model {request.model_id} is already available locally",
                    "local_path": hf_manager.get_local_model_path(request.model_id),
                }
        
        # Download and convert the model
        model, tokenizer = hf_manager.load_model_and_tokenizer(
            model_id=request.model_id,
            revision=request.revision,
            quantize=request.quantize,
            force_download=request.force_download,
            force_convert=request.force_convert,
            trust_remote_code=request.trust_remote_code,
        )
        
        local_path = hf_manager.get_local_model_path(request.model_id)
        
        return {
            "status": "success",
            "message": f"Successfully downloaded and converted {request.model_id}",
            "model_id": request.model_id,
            "local_path": local_path,
            "quantized": request.quantize is not None,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {e}")


@hf_router.delete("/models/{model_id:path}")
async def delete_local_hf_model(model_id: str):
    """Delete a locally cached MLX model."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        
        if not hf_manager.is_model_available_locally(model_id):
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found locally")
        
        success = hf_manager.delete_local_model(model_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Successfully deleted local model {model_id}",
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete model {model_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")


@hf_router.get("/models/{model_id:path}/status")
async def check_hf_model_status(model_id: str):
    """Check the status of a Hugging Face model (local availability, etc.)."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        
        is_local = hf_manager.is_model_available_locally(model_id)
        local_path = hf_manager.get_local_model_path(model_id) if is_local else None
        
        # Get basic model info
        try:
            model_info = hf_manager.get_model_info(model_id)
        except:
            model_info = {"error": "Failed to fetch model info from HF"}
        
        return {
            "model_id": model_id,
            "is_local": is_local,
            "local_path": local_path,
            "model_info": model_info,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check model status: {e}")


@hf_router.get("/cache/stats")
async def get_cache_stats():
    """Get statistics about the local model cache."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        models = hf_manager.list_local_models()
        
        total_size_mb = sum(model.get("size_mb", 0) for model in models)
        
        return {
            "cache_dir": hf_manager.mlx_cache_dir,
            "total_models": len(models),
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_mb / 1024,
            "models": [
                {
                    "model_id": model["model_id"],
                    "size_mb": model.get("size_mb", 0),
                }
                for model in models
            ],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {e}")


@hf_router.post("/cache/clear")
async def clear_cache(
    confirm: bool = Query(False, description="Confirmation required to clear cache"),
    model_ids: Optional[List[str]] = Query(None, description="Specific models to delete")
):
    """Clear the local model cache (with confirmation)."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Cache clearing requires confirmation. Add ?confirm=true to the request."
        )
    
    try:
        hf_manager = get_hf_manager()
        
        if model_ids:
            # Delete specific models
            deleted = []
            failed = []
            
            for model_id in model_ids:
                if hf_manager.delete_local_model(model_id):
                    deleted.append(model_id)
                else:
                    failed.append(model_id)
            
            return {
                "status": "partial" if failed else "success",
                "deleted": deleted,
                "failed": failed,
                "message": f"Deleted {len(deleted)} models, failed to delete {len(failed)} models",
            }
        else:
            # Delete all models
            models = hf_manager.list_local_models()
            deleted = []
            failed = []
            
            for model in models:
                model_id = model["model_id"]
                if hf_manager.delete_local_model(model_id):
                    deleted.append(model_id)
                else:
                    failed.append(model_id)
            
            return {
                "status": "partial" if failed else "success",
                "deleted": deleted,
                "failed": failed,
                "message": f"Cleared cache: deleted {len(deleted)} models, failed to delete {len(failed)} models",
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


# Popular model suggestions
POPULAR_MODELS = [
    {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Small, efficient model good for testing",
        "size": "~2GB",
        "tags": ["small", "instruct", "microsoft"],
    },
    {
        "model_id": "microsoft/Phi-3-medium-4k-instruct", 
        "description": "Medium-sized model with good performance",
        "size": "~8GB",
        "tags": ["medium", "instruct", "microsoft"],
    },
    {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "description": "Popular chat model from Meta",
        "size": "~13GB",
        "tags": ["chat", "llama", "meta"],
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "High-quality instruction-following model",
        "size": "~13GB", 
        "tags": ["instruct", "mistral"],
    },
]


@hf_router.get("/models/popular")
async def get_popular_models():
    """Get a list of popular models recommended for MLX."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=501, detail="Hugging Face integration not available")
    
    try:
        hf_manager = get_hf_manager()
        
        # Add local availability status to each model
        for model in POPULAR_MODELS:
            model["is_local"] = hf_manager.is_model_available_locally(model["model_id"])
            if model["is_local"]:
                model["local_path"] = hf_manager.get_local_model_path(model["model_id"])
        
        return {
            "models": POPULAR_MODELS,
            "total": len(POPULAR_MODELS),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get popular models: {e}")
