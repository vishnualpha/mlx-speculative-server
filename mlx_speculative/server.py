# Copyright © 2025 Manus AI

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import mlx.core as mx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from .core import SpeculativeEngine
from .models import ModelConfig, load_model_pair
from .utils import batch_generate, stream_generate
from .sample_utils import create_sampler


# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    model: Optional[str] = Field(None, description="Model name to use")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(False, description="Whether to stream the response")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation."""
    prompts: List[str] = Field(..., description="List of input prompts")
    model: Optional[str] = Field(None, description="Model name to use")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Repetition penalty")
    format_prompts: bool = Field(True, description="Whether to apply chat template")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    id: str = Field(..., description="Unique request ID")
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, Any] = Field(..., description="Usage statistics")
    finish_reason: str = Field(..., description="Reason for completion")


class BatchGenerationResponse(BaseModel):
    """Response model for batch text generation."""
    id: str = Field(..., description="Unique request ID")
    responses: List[str] = Field(..., description="Generated texts")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, Any] = Field(..., description="Usage statistics")


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    id: str = Field(..., description="Request ID")
    text: str = Field(..., description="Text chunk")
    finish_reason: Optional[str] = Field(None, description="Completion reason if finished")


@dataclass
class RequestMetrics:
    """Metrics for tracking request performance."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    tokens_generated: int = 0
    prompt_tokens: int = 0
    model_name: str = ""
    batch_size: int = 1
    
    @property
    def elapsed_time(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def throughput(self) -> float:
        elapsed = self.elapsed_time
        return self.tokens_generated / elapsed if elapsed > 0 else 0.0


class RequestQueue:
    """Asynchronous request queue with batching capabilities."""
    
    def __init__(self, max_batch_size: int = 8, batch_timeout: float = 0.1):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue = asyncio.Queue()
        self.processing = False
        
    async def add_request(self, request: Dict[str, Any]) -> str:
        """Add a request to the queue and return request ID."""
        request_id = str(uuid.uuid4())
        request["id"] = request_id
        request["timestamp"] = time.time()
        await self.queue.put(request)
        return request_id
    
    async def get_batch(self) -> List[Dict[str, Any]]:
        """Get a batch of requests from the queue."""
        batch = []
        
        # Wait for at least one request
        if self.queue.empty():
            request = await self.queue.get()
            batch.append(request)
        
        # Collect additional requests up to max_batch_size or timeout
        start_time = time.time()
        while (len(batch) < self.max_batch_size and 
               time.time() - start_time < self.batch_timeout):
            try:
                request = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                batch.append(request)
            except asyncio.TimeoutError:
                break
        
        return batch


class ModelManager:
    """Manages multiple models and their engines."""
    
    def __init__(self):
        self.models: Dict[str, SpeculativeEngine] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.default_model: Optional[str] = None
        
    def load_model(
        self,
        name: str,
        config: ModelConfig,
        auto_draft: bool = True,
        draft_layers_ratio: float = 0.5,
        set_default: bool = False,
    ) -> None:
        """Load a model and its speculative engine."""
        target_model, draft_model, tokenizer = load_model_pair(
            config, auto_draft=auto_draft, draft_layers_ratio=draft_layers_ratio
        )
        
        engine = SpeculativeEngine(
            target_model=target_model,
            draft_model=draft_model,
            num_draft_tokens=4,
            auto_draft=auto_draft,
        )
        
        self.models[name] = engine
        self.tokenizers[name] = tokenizer
        self.model_configs[name] = config
        
        if set_default or self.default_model is None:
            self.default_model = name
    
    def get_model(self, name: Optional[str] = None) -> tuple[SpeculativeEngine, Any]:
        """Get a model engine and tokenizer."""
        model_name = name or self.default_model
        if model_name is None or model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.models[model_name], self.tokenizers[model_name]
    
    def list_models(self) -> List[str]:
        """List available model names."""
        return list(self.models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        
        engine = self.models[name]
        config = self.model_configs[name]
        
        return {
            "name": name,
            "config": asdict(config),
            "stats": engine.stats,
            "is_default": name == self.default_model,
        }


class SpeculativeServer:
    """High-performance speculative decoding server."""
    
    def __init__(
        self,
        max_batch_size: int = 8,
        batch_timeout: float = 0.1,
        max_concurrent_requests: int = 100,
    ):
        self.model_manager = ModelManager()
        self.request_queue = RequestQueue(max_batch_size, batch_timeout)
        self.max_concurrent_requests = max_concurrent_requests
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.processing_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
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
        """Background task to process requests in batches."""
        while True:
            try:
                batch = await self.request_queue.get_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing batch: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests."""
        if not batch:
            return
        
        # Group requests by model
        model_batches: Dict[str, List[Dict[str, Any]]] = {}
        for request in batch:
            model_name = request.get("model") or self.model_manager.default_model
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
        """Process a batch of requests for a specific model."""
        try:
            engine, tokenizer = self.model_manager.get_model(model_name)
            
            # Extract prompts and parameters
            prompts = [req["prompt"] for req in requests]
            
            # Use parameters from first request (assuming similar parameters in batch)
            params = requests[0]
            
            # Set random seed if provided
            if params.get("seed") is not None:
                mx.random.seed(params["seed"])
            
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
            
            # Update metrics and complete requests
            for request, response in zip(requests, responses):
                request_id = request["id"]
                if request_id in self.active_requests:
                    metrics = self.active_requests[request_id]
                    metrics.end_time = time.time()
                    metrics.tokens_generated = len(tokenizer.encode(response))
                    metrics.model_name = model_name
                    
                    # Store response for retrieval
                    request["response"] = response
                    request["metrics"] = metrics
                    
                    # Update global stats
                    self.total_requests += 1
                    self.total_tokens += metrics.tokens_generated
        
        except Exception as e:
            # Mark all requests in batch as failed
            for request in requests:
                request["error"] = str(e)
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for a single request."""
        async with self.request_semaphore:
            request_id = str(uuid.uuid4())
            
            # Create metrics
            metrics = RequestMetrics(
                request_id=request_id,
                start_time=time.time(),
                prompt_tokens=len(request.prompt.split()),  # Rough estimate
            )
            self.active_requests[request_id] = metrics
            
            try:
                engine, tokenizer = self.model_manager.get_model(request.model)
                
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
                metrics.model_name = request.model or self.model_manager.default_model
                
                # Update global stats
                self.total_requests += 1
                self.total_tokens += metrics.tokens_generated
                
                return GenerationResponse(
                    id=request_id,
                    text=response_text,
                    model=metrics.model_name,
                    usage={
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.tokens_generated,
                        "total_tokens": metrics.prompt_tokens + metrics.tokens_generated,
                        "elapsed_time": metrics.elapsed_time,
                        "throughput": metrics.throughput,
                    },
                    finish_reason="length",
                )
            
            finally:
                # Clean up metrics
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
    
    async def batch_generate_async(self, request: BatchGenerationRequest) -> BatchGenerationResponse:
        """Generate text for multiple prompts."""
        async with self.request_semaphore:
            request_id = str(uuid.uuid4())
            
            # Create metrics
            metrics = RequestMetrics(
                request_id=request_id,
                start_time=time.time(),
                batch_size=len(request.prompts),
                prompt_tokens=sum(len(p.split()) for p in request.prompts),  # Rough estimate
            )
            self.active_requests[request_id] = metrics
            
            try:
                engine, tokenizer = self.model_manager.get_model(request.model)
                
                # Set random seed if provided
                if request.seed is not None:
                    mx.random.seed(request.seed)
                
                # Generate responses
                responses = batch_generate(
                    engine=engine,
                    tokenizer=tokenizer,
                    prompts=request.prompts,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    format_prompts=request.format_prompts,
                    verbose=False,
                )
                
                # Update metrics
                metrics.end_time = time.time()
                metrics.tokens_generated = sum(len(tokenizer.encode(r)) for r in responses)
                metrics.model_name = request.model or self.model_manager.default_model
                
                # Update global stats
                self.total_requests += 1
                self.total_tokens += metrics.tokens_generated
                
                return BatchGenerationResponse(
                    id=request_id,
                    responses=responses,
                    model=metrics.model_name,
                    usage={
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.tokens_generated,
                        "total_tokens": metrics.prompt_tokens + metrics.tokens_generated,
                        "elapsed_time": metrics.elapsed_time,
                        "throughput": metrics.throughput,
                        "batch_size": metrics.batch_size,
                    },
                )
            
            finally:
                # Clean up metrics
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
    
    async def stream_generate_async(self, request: GenerationRequest):
        """Generate streaming text response."""
        async with self.request_semaphore:
            request_id = str(uuid.uuid4())
            
            try:
                engine, tokenizer = self.model_manager.get_model(request.model)
                
                # Set random seed if provided
                if request.seed is not None:
                    mx.random.seed(request.seed)
                
                # Stream generation
                async def generate_stream():
                    for chunk in stream_generate(
                        engine=engine,
                        tokenizer=tokenizer,
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=request.repetition_penalty,
                    ):
                        yield f"data: {json.dumps(StreamChunk(id=request_id, text=chunk).dict())}\n\n"
                    
                    # Send final chunk
                    yield f"data: {json.dumps(StreamChunk(id=request_id, text='', finish_reason='length').dict())}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime": uptime,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "average_throughput": self.total_tokens / uptime if uptime > 0 else 0,
            "active_requests": len(self.active_requests),
            "loaded_models": self.model_manager.list_models(),
            "default_model": self.model_manager.default_model,
        }


# Global server instance
server = SpeculativeServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle."""
    # Startup
    await server.start_processing()
    yield
    # Shutdown
    await server.stop_processing()


# FastAPI app
app = FastAPI(
    title="MLX Speculative Decoding Server",
    description="High-performance speculative decoding server for Apple Silicon",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from a prompt."""
    if request.stream:
        return await server.stream_generate_async(request)
    else:
        return await server.generate(request)


@app.post("/batch_generate", response_model=BatchGenerationResponse)
async def batch_generate_text(request: BatchGenerationRequest):
    """Generate text for multiple prompts."""
    return await server.batch_generate_async(request)


@app.get("/models")
async def list_models():
    """List available models."""
    return {"models": server.model_manager.list_models()}


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    try:
        return server.model_manager.get_model_info(model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    return server.get_stats()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    models: Optional[Dict[str, str]] = None,
    **kwargs
):
    """
    Run the speculative decoding server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        models: Dictionary of model_name -> model_path to load
        **kwargs: Additional arguments for uvicorn
    """
    # Load models if provided
    if models:
        for name, path in models.items():
            config = ModelConfig(model_path=path)
            server.model_manager.load_model(
                name=name,
                config=config,
                set_default=(name == list(models.keys())[0])
            )
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        **kwargs
    )
