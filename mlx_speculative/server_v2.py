# Copyright © 2025 Manus AI

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import mlx.core as mx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .core import SpeculativeEngine
from .models import ModelConfig
from .multi_model import MultiModelManager
from .utils import batch_generate, stream_generate
from .sample_utils import create_sampler


# Enhanced Request/Response Models
class GenerationRequest(BaseModel):
    """Enhanced request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    model: Optional[str] = Field(None, description="Model name or group to use")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(False, description="Whether to stream the response")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Advanced options
    use_cache: bool = Field(True, description="Whether to use KV cache")
    priority: int = Field(0, description="Request priority (higher = more priority)")
    timeout: Optional[float] = Field(None, description="Request timeout in seconds")


class BatchGenerationRequest(BaseModel):
    """Enhanced request model for batch text generation."""
    prompts: List[str] = Field(..., description="List of input prompts")
    model: Optional[str] = Field(None, description="Model name or group to use")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Repetition penalty")
    format_prompts: bool = Field(True, description="Whether to apply chat template")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Advanced options
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    priority: int = Field(0, description="Request priority")
    timeout: Optional[float] = Field(None, description="Request timeout in seconds")


class ModelLoadRequest(BaseModel):
    """Request model for loading a new model."""
    name: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Path to model")
    draft_model_path: Optional[str] = Field(None, description="Path to draft model")
    trust_remote_code: bool = Field(False, description="Trust remote code")
    auto_draft: bool = Field(True, description="Enable automatic draft model")
    draft_layers_ratio: float = Field(0.5, description="Draft model layers ratio")
    set_default: bool = Field(False, description="Set as default model")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    force: bool = Field(False, description="Force loading even if limits exceeded")


class ModelGroupRequest(BaseModel):
    """Request model for creating a model group."""
    name: str = Field(..., description="Group name")
    models: List[str] = Field(..., description="List of model names")
    default_model: str = Field(..., description="Default model in group")
    description: str = Field("", description="Group description")
    tags: List[str] = Field(default_factory=list, description="Group tags")


class ModelSearchRequest(BaseModel):
    """Request model for searching models."""
    query: Optional[str] = Field(None, description="Search query")
    tags: List[str] = Field(default_factory=list, description="Required tags")
    model_type: Optional[str] = Field(None, description="Model type filter")


class EnhancedGenerationResponse(BaseModel):
    """Enhanced response model for text generation."""
    id: str = Field(..., description="Unique request ID")
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, Any] = Field(..., description="Usage statistics")
    finish_reason: str = Field(..., description="Reason for completion")
    
    # Enhanced metrics
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    speculative_stats: Dict[str, Any] = Field(..., description="Speculative decoding stats")


class ServerStats(BaseModel):
    """Enhanced server statistics."""
    uptime: float
    total_requests: int
    total_tokens: int
    average_throughput: float
    active_requests: int
    
    # Model information
    loaded_models: List[str]
    model_groups: List[str]
    default_model: Optional[str]
    
    # Performance metrics
    model_performance: Dict[str, Any]
    memory_usage: Dict[str, Any]
    load_balancing: Dict[str, Any]


@dataclass
class EnhancedRequestMetrics:
    """Enhanced metrics for tracking request performance."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    tokens_generated: int = 0
    prompt_tokens: int = 0
    model_name: str = ""
    batch_size: int = 1
    priority: int = 0
    
    # Speculative decoding metrics
    draft_tokens: int = 0
    accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    
    @property
    def elapsed_time(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def throughput(self) -> float:
        elapsed = self.elapsed_time
        return self.tokens_generated / elapsed if elapsed > 0 else 0.0


class EnhancedSpeculativeServer:
    """Enhanced high-performance speculative decoding server with multi-model support."""
    
    def __init__(
        self,
        max_batch_size: int = 8,
        batch_timeout: float = 0.1,
        max_concurrent_requests: int = 100,
        max_models: int = 10,
        memory_limit_gb: Optional[float] = None,
        auto_unload: bool = True,
        load_balancing: bool = True,
    ):
        self.model_manager = MultiModelManager(
            max_models=max_models,
            memory_limit_gb=memory_limit_gb,
            auto_unload=auto_unload,
            load_balancing=load_balancing,
        )
        
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_requests = max_concurrent_requests
        
        self.active_requests: Dict[str, EnhancedRequestMetrics] = {}
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Performance metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
        # Request queue with priority support
        self.request_queue = asyncio.PriorityQueue()
        self.processing_task: Optional[asyncio.Task] = None
    
    async def start_processing(self):
        """Start the background request processing task."""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_requests())
    
    async def stop_processing(self):
        """Stop the background request processing task."""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def _process_requests(self):
        """Background task to process requests with priority support."""
        while True:
            try:
                # Get high-priority requests first
                batch = []
                start_time = time.time()
                
                # Wait for at least one request
                if self.request_queue.empty():
                    priority, request = await self.request_queue.get()
                    batch.append(request)
                
                # Collect more requests up to batch size or timeout
                while (len(batch) < self.max_batch_size and 
                       time.time() - start_time < self.batch_timeout):
                    try:
                        priority, request = await asyncio.wait_for(
                            self.request_queue.get(), timeout=0.01
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_batch(batch)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing requests: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests with model grouping."""
        if not batch:
            return
        
        # Group requests by model
        model_batches: Dict[str, List[Dict[str, Any]]] = {}
        for request in batch:
            model_name = request.get("model")
            if model_name not in model_batches:
                model_batches[model_name] = []
            model_batches[model_name].append(request)
        
        # Process each model batch
        tasks = []
        for model_name, requests in model_batches.items():
            task = asyncio.create_task(self._process_model_batch(model_name, requests))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_model_batch(self, model_name: str, requests: List[Dict[str, Any]]):
        """Process a batch of requests for a specific model with enhanced metrics."""
        try:
            engine, tokenizer, actual_model_name = self.model_manager.get_model(model_name)
            
            # Extract prompts and parameters
            prompts = [req["prompt"] for req in requests]
            params = requests[0]  # Use first request's parameters
            
            # Set random seed if provided
            if params.get("seed") is not None:
                mx.random.seed(params["seed"])
            
            # Track start time for batch
            batch_start_time = time.time()
            
            # Generate responses
            responses = batch_generate(
                engine=engine,
                tokenizer=tokenizer,
                prompts=prompts,
                max_tokens=params.get("max_tokens", 100),
                temperature=params.get("temperature", 0.0),
                top_p=params.get("top_p", 1.0),
                top_k=params.get("top_k"),
                repetition_penalty=params.get("repetition_penalty", 1.0),
                format_prompts=params.get("format_prompts", True),
                verbose=False,
            )
            
            batch_end_time = time.time()
            batch_elapsed = batch_end_time - batch_start_time
            
            # Update metrics for each request
            for request, response in zip(requests, responses):
                request_id = request["id"]
                if request_id in self.active_requests:
                    metrics = self.active_requests[request_id]
                    metrics.end_time = batch_end_time
                    metrics.tokens_generated = len(tokenizer.encode(response))
                    metrics.model_name = actual_model_name
                    
                    # Add speculative decoding stats
                    engine_stats = engine.stats
                    metrics.draft_tokens = engine_stats.get("draft_tokens", 0)
                    metrics.accepted_tokens = engine_stats.get("accepted_tokens", 0)
                    metrics.acceptance_rate = engine_stats.get("acceptance_rate", 0.0)
                    
                    # Store response and metrics
                    request["response"] = response
                    request["metrics"] = metrics
                    request["batch_elapsed"] = batch_elapsed
                    
                    # Update global stats
                    self.total_requests += 1
                    self.total_tokens += metrics.tokens_generated
        
        except Exception as e:
            # Mark all requests in batch as failed
            for request in requests:
                request["error"] = str(e)
    
    async def generate(self, request: GenerationRequest) -> EnhancedGenerationResponse:
        """Generate text with enhanced metrics and model selection."""
        async with self.request_semaphore:
            request_id = str(uuid.uuid4())
            
            # Create enhanced metrics
            metrics = EnhancedRequestMetrics(
                request_id=request_id,
                start_time=time.time(),
                prompt_tokens=len(request.prompt.split()),
                priority=request.priority,
            )
            self.active_requests[request_id] = metrics
            
            try:
                engine, tokenizer, actual_model_name = self.model_manager.get_model(request.model)
                
                # Set random seed if provided
                if request.seed is not None:
                    mx.random.seed(request.seed)
                
                # Generate response
                from .utils import generate
                response_text = generate(
                    engine=engine,
                    tokenizer=tokenizer,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    verbose=False,
                )
                
                # Update metrics
                metrics.end_time = time.time()
                metrics.tokens_generated = len(tokenizer.encode(response_text))
                metrics.model_name = actual_model_name
                
                # Get speculative decoding stats
                engine_stats = engine.stats
                metrics.draft_tokens = engine_stats.get("draft_tokens", 0)
                metrics.accepted_tokens = engine_stats.get("accepted_tokens", 0)
                metrics.acceptance_rate = engine_stats.get("acceptance_rate", 0.0)
                
                # Update global stats
                self.total_requests += 1
                self.total_tokens += metrics.tokens_generated
                
                return EnhancedGenerationResponse(
                    id=request_id,
                    text=response_text,
                    model=actual_model_name,
                    usage={
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.tokens_generated,
                        "total_tokens": metrics.prompt_tokens + metrics.tokens_generated,
                    },
                    finish_reason="length",
                    performance={
                        "elapsed_time": metrics.elapsed_time,
                        "throughput": metrics.throughput,
                        "priority": metrics.priority,
                    },
                    speculative_stats={
                        "draft_tokens": metrics.draft_tokens,
                        "accepted_tokens": metrics.accepted_tokens,
                        "acceptance_rate": metrics.acceptance_rate,
                        "speedup": metrics.acceptance_rate + 1.0,  # Approximate speedup
                    },
                )
            
            finally:
                # Clean up metrics
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
    
    async def load_model(self, request: ModelLoadRequest) -> Dict[str, Any]:
        """Load a new model with enhanced options."""
        config = ModelConfig(
            model_path=request.model_path,
            draft_model_path=request.draft_model_path,
            trust_remote_code=request.trust_remote_code,
        )
        
        success = await self.model_manager.load_model_async(
            name=request.name,
            config=config,
            auto_draft=request.auto_draft,
            draft_layers_ratio=request.draft_layers_ratio,
            set_default=request.set_default,
            tags=set(request.tags),
            force=request.force,
        )
        
        if success:
            model_info = self.model_manager.get_model_info(request.name)
            return {
                "status": "success",
                "message": f"Model '{request.name}' loaded successfully",
                "model_info": model_info,
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to load model '{request.name}'",
            }
    
    async def create_model_group(self, request: ModelGroupRequest) -> Dict[str, Any]:
        """Create a model group."""
        success = self.model_manager.create_model_group(
            group_name=request.name,
            model_names=request.models,
            default_model=request.default_model,
            description=request.description,
            tags=set(request.tags),
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Model group '{request.name}' created successfully",
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to create model group '{request.name}'",
            }
    
    def search_models(self, request: ModelSearchRequest) -> List[str]:
        """Search for models based on criteria."""
        return self.model_manager.search_models(
            query=request.query,
            tags=set(request.tags) if request.tags else None,
            model_type=request.model_type,
        )
    
    def get_enhanced_stats(self) -> ServerStats:
        """Get enhanced server statistics."""
        uptime = time.time() - self.start_time
        model_list = self.model_manager.list_models()
        model_performance = self.model_manager.get_performance_stats()
        
        return ServerStats(
            uptime=uptime,
            total_requests=self.total_requests,
            total_tokens=self.total_tokens,
            average_throughput=self.total_tokens / uptime if uptime > 0 else 0,
            active_requests=len(self.active_requests),
            loaded_models=model_list["models"],
            model_groups=model_list.get("groups", []),
            default_model=model_list["default_model"],
            model_performance=model_performance["models"],
            memory_usage=model_performance["memory_usage"],
            load_balancing=model_performance["load_balancing"],
        )


# Global enhanced server instance
enhanced_server = EnhancedSpeculativeServer()


@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Manage enhanced server lifecycle."""
    # Startup
    await enhanced_server.start_processing()
    yield
    # Shutdown
    await enhanced_server.stop_processing()


# Enhanced FastAPI app
enhanced_app = FastAPI(
    title="MLX Speculative Decoding Server - Enhanced",
    description="High-performance speculative decoding server with multi-model support for Apple Silicon",
    version="0.2.0",
    lifespan=enhanced_lifespan,
)


@enhanced_app.post("/generate", response_model=EnhancedGenerationResponse)
async def enhanced_generate_text(request: GenerationRequest):
    """Generate text with enhanced features."""
    if request.stream:
        return await enhanced_server.stream_generate_async(request)
    else:
        return await enhanced_server.generate(request)


@enhanced_app.post("/models/load")
async def load_model_endpoint(request: ModelLoadRequest):
    """Load a new model."""
    return await enhanced_server.load_model(request)


@enhanced_app.delete("/models/{model_name}")
async def unload_model_endpoint(model_name: str):
    """Unload a model."""
    success = enhanced_server.model_manager.unload_model(model_name)
    if success:
        return {"status": "success", "message": f"Model '{model_name}' unloaded"}
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


@enhanced_app.post("/groups")
async def create_model_group_endpoint(request: ModelGroupRequest):
    """Create a model group."""
    return await enhanced_server.create_model_group(request)


@enhanced_app.get("/models")
async def list_models_enhanced():
    """List all models and groups."""
    return enhanced_server.model_manager.list_models(include_groups=True)


@enhanced_app.get("/models/{model_name}")
async def get_model_info_enhanced(model_name: str):
    """Get detailed information about a model or group."""
    try:
        return enhanced_server.model_manager.get_model_info(model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@enhanced_app.post("/models/search")
async def search_models_endpoint(request: ModelSearchRequest):
    """Search for models based on criteria."""
    results = enhanced_server.search_models(request)
    return {"results": results}


@enhanced_app.get("/stats", response_model=ServerStats)
async def get_enhanced_stats():
    """Get enhanced server statistics."""
    return enhanced_server.get_enhanced_stats()


@enhanced_app.get("/optimize")
async def optimize_models():
    """Get model optimization recommendations."""
    return enhanced_server.model_manager.optimize_models()


@enhanced_app.post("/config/save")
async def save_config(config_path: str = Query(..., description="Path to save configuration")):
    """Save current configuration to file."""
    try:
        enhanced_server.model_manager.save_config(config_path)
        return {"status": "success", "message": f"Configuration saved to {config_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enhanced_app.post("/config/load")
async def load_config(config_path: str = Query(..., description="Path to load configuration from")):
    """Load configuration from file."""
    try:
        success = enhanced_server.model_manager.load_config(config_path)
        if success:
            return {"status": "success", "message": f"Configuration loaded from {config_path}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to load configuration")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enhanced_app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with model status."""
    model_list = enhanced_server.model_manager.list_models()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": model_list["total_models"],
        "groups_created": model_list.get("total_groups", 0),
        "default_model": model_list["default_model"],
    }


def run_enhanced_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    models: Optional[Dict[str, str]] = None,
    config_file: Optional[str] = None,
    **kwargs
):
    """
    Run the enhanced speculative decoding server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        models: Dictionary of model_name -> model_path to load
        config_file: Path to configuration file
        **kwargs: Additional arguments for uvicorn
    """
    # Load configuration if provided
    if config_file:
        enhanced_server.model_manager.load_config(config_file)
    
    # Load models if provided
    elif models:
        for name, path in models.items():
            config = ModelConfig(model_path=path)
            enhanced_server.model_manager.load_model(
                name=name,
                config=config,
                set_default=(name == list(models.keys())[0])
            )
    
    # Run server
    uvicorn.run(
        enhanced_app,
        host=host,
        port=port,
        **kwargs
    )
