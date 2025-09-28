#!/usr/bin/env python3
"""
Example usage of MLX Speculative Server with Hugging Face integration

This script demonstrates how to:
1. Download models from Hugging Face
2. Start the server with HF models
3. Use the enhanced API features
"""

import requests
import time
import json
from pathlib import Path

# Server configuration
SERVER_URL = "http://localhost:8000"

def check_server_health():
    """Check if the server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        return response.status_code == 200
    except:
        return False

def download_model_via_api(model_id: str, quantize: str = None):
    """Download a model via the API."""
    print(f"Downloading model: {model_id}")
    
    data = {
        "model_id": model_id,
        "trust_remote_code": False
    }
    
    if quantize:
        data["quantize"] = quantize
        print(f"With quantization: {quantize}")
    
    response = requests.post(f"{SERVER_URL}/hf/models/download", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
        return result["local_path"]
    else:
        print(f"✗ Failed to download: {response.text}")
        return None

def search_models(query: str):
    """Search for models on Hugging Face."""
    print(f"Searching for models: {query}")
    
    data = {"query": query, "limit": 5}
    response = requests.post(f"{SERVER_URL}/hf/models/search", json=data)
    
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Found {len(results)} models:")
        
        for model in results:
            status = "Local ✓" if model["is_local"] else "Remote"
            print(f"  [{status}] {model['model_id']}")
            print(f"    Downloads: {model.get('downloads', 0):,}")
            if model.get('tags'):
                print(f"    Tags: {', '.join(model['tags'][:3])}")
        
        return results
    else:
        print(f"✗ Search failed: {response.text}")
        return []

def load_model_to_server(model_id: str, name: str = None):
    """Load a model into the server."""
    if not name:
        name = model_id.split("/")[-1]
    
    print(f"Loading model {model_id} as '{name}'")
    
    data = {
        "name": name,
        "model_path": model_id,  # Can be HF model ID
        "auto_draft": True,
        "set_default": True
    }
    
    response = requests.post(f"{SERVER_URL}/models/load", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Model loaded: {result['status']}")
        return True
    else:
        print(f"✗ Failed to load model: {response.text}")
        return False

def generate_text(prompt: str, model: str = None):
    """Generate text using the server."""
    print(f"Generating text for: {prompt[:50]}...")
    
    data = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    if model:
        data["model"] = model
    
    response = requests.post(f"{SERVER_URL}/generate", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Generated text:")
        print(f"  Model: {result['model']}")
        print(f"  Text: {result['text']}")
        print(f"  Throughput: {result['performance']['throughput']:.1f} tok/s")
        print(f"  Acceptance rate: {result['speculative_stats']['acceptance_rate']:.2f}")
        return result["text"]
    else:
        print(f"✗ Generation failed: {response.text}")
        return None

def list_local_models():
    """List locally cached models."""
    print("Local models:")
    
    response = requests.get(f"{SERVER_URL}/hf/models/local")
    
    if response.status_code == 200:
        result = response.json()
        models = result["models"]
        
        if not models:
            print("  No local models found")
            return []
        
        for model in models:
            size_mb = model.get("size_mb", 0)
            print(f"  {model['model_id']} ({size_mb:.1f} MB)")
        
        print(f"Total: {len(models)} models, {result.get('total_size_gb', 0):.1f} GB")
        return models
    else:
        print(f"✗ Failed to list models: {response.text}")
        return []

def get_popular_models():
    """Get popular model recommendations."""
    print("Popular models:")
    
    response = requests.get(f"{SERVER_URL}/hf/models/popular")
    
    if response.status_code == 200:
        result = response.json()
        models = result["models"]
        
        for model in models:
            status = "Local ✓" if model["is_local"] else "Remote"
            print(f"  [{status}] {model['model_id']}")
            print(f"    {model['description']}")
            print(f"    Size: {model['size']}")
        
        return models
    else:
        print(f"✗ Failed to get popular models: {response.text}")
        return []

def main():
    """Main example function."""
    print("MLX Speculative Server - Hugging Face Integration Example")
    print("=" * 60)
    
    # Check if server is running
    if not check_server_health():
        print("✗ Server is not running!")
        print("Start the server with: mlx-spec serve")
        return
    
    print("✓ Server is running")
    print()
    
    # Show popular models
    print("1. Popular Models")
    print("-" * 20)
    popular = get_popular_models()
    print()
    
    # List local models
    print("2. Local Models")
    print("-" * 20)
    local = list_local_models()
    print()
    
    # Search for models
    print("3. Search Models")
    print("-" * 20)
    search_results = search_models("phi-3")
    print()
    
    # Download a small model for testing
    print("4. Download Model")
    print("-" * 20)
    
    # Use Phi-3-mini as it's relatively small
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    # Check if already local
    local_models = [m["model_id"] for m in local]
    if model_id not in local_models:
        local_path = download_model_via_api(model_id)
        if not local_path:
            print("Failed to download model, using existing local model if available")
    else:
        print(f"✓ Model {model_id} already available locally")
    
    print()
    
    # Load model into server
    print("5. Load Model")
    print("-" * 20)
    success = load_model_to_server(model_id, "phi3-mini")
    
    if not success:
        print("Failed to load model, cannot continue with generation")
        return
    
    print()
    
    # Generate text
    print("6. Generate Text")
    print("-" * 20)
    
    test_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about Apple Silicon.",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_text(prompt, "phi3-mini")
        print()
    
    # Show server stats
    print("7. Server Statistics")
    print("-" * 20)
    
    response = requests.get(f"{SERVER_URL}/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"Uptime: {stats['uptime']:.1f}s")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average throughput: {stats['average_throughput']:.1f} tok/s")
        print(f"Active requests: {stats['active_requests']}")
        print(f"Loaded models: {len(stats['loaded_models'])}")
    
    print()
    print("Example completed! 🎉")

if __name__ == "__main__":
    main()
