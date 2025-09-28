#!/usr/bin/env python3
"""
Example client for MLX Speculative Server

This script demonstrates how to interact with the MLX Speculative Server
using Python requests.
"""

import requests
import json
import time
from typing import List, Dict, Any


class MLXSpeculativeClient:
    """Client for MLX Speculative Server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        response = self.session.get(f"{self.base_url}/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def load_model(
        self,
        name: str,
        model_path: str,
        draft_model_path: str = None,
        set_default: bool = False,
        tags: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Load a new model."""
        data = {
            "name": name,
            "model_path": model_path,
            "set_default": set_default,
            "tags": tags or [],
            **kwargs
        }
        
        if draft_model_path:
            data["draft_model_path"] = draft_model_path
        
        response = self.session.post(f"{self.base_url}/models/load", json=data)
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model."""
        response = self.session.delete(f"{self.base_url}/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from a prompt."""
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        if model:
            data["model"] = model
        
        response = self.session.post(f"{self.base_url}/generate", json=data)
        response.raise_for_status()
        return response.json()
    
    def generate_stream(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """Generate text with streaming."""
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **kwargs
        }
        
        if model:
            data["model"] = model
        
        response = self.session.post(
            f"{self.base_url}/generate", 
            json=data, 
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        yield chunk
                    except json.JSONDecodeError:
                        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        model: str = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text for multiple prompts."""
        data = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        if model:
            data["model"] = model
        
        response = self.session.post(f"{self.base_url}/batch_generate", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def search_models(
        self,
        query: str = None,
        tags: List[str] = None,
        model_type: str = None
    ) -> Dict[str, Any]:
        """Search for models."""
        data = {}
        if query:
            data["query"] = query
        if tags:
            data["tags"] = tags
        if model_type:
            data["model_type"] = model_type
        
        response = self.session.post(f"{self.base_url}/models/search", json=data)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the MLX Speculative Client."""
    # Initialize client
    client = MLXSpeculativeClient("http://localhost:8000")
    
    try:
        # Check server health
        print("Checking server health...")
        health = client.health_check()
        print(f"Server status: {health['status']}")
        print(f"Models loaded: {health['models_loaded']}")
        print()
        
        # List available models
        print("Available models:")
        models = client.list_models()
        for model in models["models"]:
            print(f"  - {model}")
        print(f"Default model: {models['default_model']}")
        print()
        
        # Example: Load a model (uncomment if you have a model)
        # print("Loading model...")
        # result = client.load_model(
        #     name="phi3-mini",
        #     model_path="/path/to/phi3-mini",
        #     set_default=True,
        #     tags=["phi", "small"]
        # )
        # print(f"Load result: {result['status']}")
        # print()
        
        # Example: Generate text (requires a loaded model)
        if models["models"]:
            print("Generating text...")
            result = client.generate(
                prompt="What is the future of artificial intelligence?",
                max_tokens=50,
                temperature=0.7
            )
            
            print(f"Generated text: {result['text']}")
            print(f"Model used: {result['model']}")
            print(f"Tokens generated: {result['usage']['completion_tokens']}")
            print(f"Throughput: {result['performance']['throughput']:.1f} tok/s")
            print(f"Acceptance rate: {result['speculative_stats']['acceptance_rate']:.2f}")
            print()
        
        # Example: Streaming generation
        if models["models"]:
            print("Streaming generation:")
            print("Generated text: ", end="", flush=True)
            
            for chunk in client.generate_stream(
                prompt="Tell me a short story about a robot.",
                max_tokens=100,
                temperature=0.8
            ):
                if chunk.get("text"):
                    print(chunk["text"], end="", flush=True)
            
            print("\n")
        
        # Example: Batch generation
        if models["models"]:
            print("Batch generation...")
            batch_prompts = [
                "What is machine learning?",
                "How does quantum computing work?",
                "Explain blockchain technology."
            ]
            
            result = client.batch_generate(
                prompts=batch_prompts,
                max_tokens=30,
                temperature=0.5
            )
            
            print("Batch results:")
            for i, response in enumerate(result["responses"]):
                print(f"  {i+1}. {response}")
            print()
        
        # Get server statistics
        print("Server statistics:")
        stats = client.get_stats()
        print(f"  Uptime: {stats['uptime']:.1f}s")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Average throughput: {stats['average_throughput']:.1f} tok/s")
        print(f"  Active requests: {stats['active_requests']}")
        print()
        
        # Search models
        if models["models"]:
            print("Searching models...")
            search_result = client.search_models(query="phi")
            print(f"Models matching 'phi': {search_result['results']}")
    
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the MLX Speculative Server is running on http://localhost:8000")
    
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
