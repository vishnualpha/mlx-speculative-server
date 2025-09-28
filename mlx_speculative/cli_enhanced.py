#!/usr/bin/env python3
# Copyright © 2025 Manus AI

"""
Enhanced CLI for MLX Speculative Server with Hugging Face integration
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .server_v2 import run_enhanced_server
from .models import ModelConfig
from .hf_utils import HF_AVAILABLE, get_hf_manager


def cmd_serve(args):
    """Start the enhanced server."""
    models = {}
    
    # Handle model loading
    if args.model and args.model_path:
        models[args.model] = args.model_path
    elif args.models:
        # Parse models from JSON string or file
        if os.path.exists(args.models):
            with open(args.models, 'r') as f:
                models = json.load(f)
        else:
            models = json.loads(args.models)
    
    # Run server
    run_enhanced_server(
        host=args.host,
        port=args.port,
        models=models,
        config_file=args.config,
        reload=args.reload,
        workers=args.workers,
    )


def cmd_download(args):
    """Download a model from Hugging Face."""
    if not HF_AVAILABLE:
        print("Error: Hugging Face integration not available")
        print("Install with: pip install huggingface_hub transformers")
        return 1
    
    try:
        hf_manager = get_hf_manager()
        
        print(f"Downloading model: {args.model_id}")
        if args.quantize:
            print(f"Quantization: {args.quantize}")
        
        model, tokenizer = hf_manager.load_model_and_tokenizer(
            model_id=args.model_id,
            revision=args.revision,
            quantize=args.quantize,
            force_download=args.force_download,
            force_convert=args.force_convert,
            trust_remote_code=args.trust_remote_code,
        )
        
        local_path = hf_manager.get_local_model_path(args.model_id)
        print(f"Successfully downloaded and converted to: {local_path}")
        
        if args.test:
            print("\nTesting model...")
            from .utils import generate
            from .core import SpeculativeEngine
            
            # Create a simple engine for testing
            engine = SpeculativeEngine(target_model=model, draft_model=model)
            
            test_prompt = "Hello, how are you?"
            response = generate(
                engine=engine,
                tokenizer=tokenizer,
                prompt=test_prompt,
                max_tokens=20,
                temperature=0.7,
            )
            
            print(f"Test prompt: {test_prompt}")
            print(f"Response: {response}")
        
        return 0
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1


def cmd_list_local(args):
    """List locally cached models."""
    if not HF_AVAILABLE:
        print("Error: Hugging Face integration not available")
        return 1
    
    try:
        hf_manager = get_hf_manager()
        models = hf_manager.list_local_models()
        
        if not models:
            print("No local models found.")
            return 0
        
        print(f"Found {len(models)} local models:")
        print()
        
        total_size = 0
        for model in models:
            size_mb = model.get("size_mb", 0)
            total_size += size_mb
            
            print(f"Model: {model['model_id']}")
            print(f"  Path: {model['local_path']}")
            print(f"  Size: {size_mb:.1f} MB")
            
            config = model.get("config", {})
            if config:
                print(f"  Architecture: {config.get('architectures', ['Unknown'])[0]}")
                if 'num_hidden_layers' in config:
                    print(f"  Layers: {config['num_hidden_layers']}")
                if 'vocab_size' in config:
                    print(f"  Vocab size: {config['vocab_size']}")
            print()
        
        print(f"Total size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
        
        return 0
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return 1


def cmd_search(args):
    """Search for models on Hugging Face."""
    if not HF_AVAILABLE:
        print("Error: Hugging Face integration not available")
        return 1
    
    try:
        hf_manager = get_hf_manager()
        
        results = hf_manager.search_models(
            query=args.query,
            task=args.task,
            library=args.library,
            tags=args.tags,
            limit=args.limit,
        )
        
        if not results:
            print("No models found.")
            return 0
        
        print(f"Found {len(results)} models:")
        print()
        
        for model in results:
            print(f"Model: {model['model_id']}")
            print(f"  Downloads: {model.get('downloads', 0):,}")
            print(f"  Likes: {model.get('likes', 0)}")
            
            tags = model.get('tags', [])
            if tags:
                print(f"  Tags: {', '.join(tags[:5])}")
            
            if model.get('is_local'):
                print("  Status: Available locally ✓")
            else:
                print("  Status: Not downloaded")
            
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error searching models: {e}")
        return 1


def cmd_delete(args):
    """Delete a local model."""
    if not HF_AVAILABLE:
        print("Error: Hugging Face integration not available")
        return 1
    
    try:
        hf_manager = get_hf_manager()
        
        if not hf_manager.is_model_available_locally(args.model_id):
            print(f"Model {args.model_id} not found locally.")
            return 1
        
        if not args.force:
            response = input(f"Delete model {args.model_id}? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0
        
        success = hf_manager.delete_local_model(args.model_id)
        
        if success:
            print(f"Successfully deleted {args.model_id}")
            return 0
        else:
            print(f"Failed to delete {args.model_id}")
            return 1
            
    except Exception as e:
        print(f"Error deleting model: {e}")
        return 1


def cmd_info(args):
    """Get information about a model."""
    if not HF_AVAILABLE:
        print("Error: Hugging Face integration not available")
        return 1
    
    try:
        hf_manager = get_hf_manager()
        info = hf_manager.get_model_info(args.model_id)
        
        print(f"Model: {info['model_id']}")
        print(f"Local: {'Yes' if info.get('is_local') else 'No'}")
        
        if info.get('local_path'):
            print(f"Path: {info['local_path']}")
        
        if 'downloads' in info:
            print(f"Downloads: {info['downloads']:,}")
        
        if 'likes' in info:
            print(f"Likes: {info['likes']}")
        
        if 'tags' in info:
            tags = info['tags']
            if tags:
                print(f"Tags: {', '.join(tags)}")
        
        if 'pipeline_tag' in info:
            print(f"Task: {info['pipeline_tag']}")
        
        config = info.get('config', {})
        if config:
            print("\nConfiguration:")
            if 'architectures' in config:
                print(f"  Architecture: {config['architectures'][0]}")
            if 'num_hidden_layers' in config:
                print(f"  Layers: {config['num_hidden_layers']}")
            if 'vocab_size' in config:
                print(f"  Vocab size: {config['vocab_size']}")
            if 'hidden_size' in config:
                print(f"  Hidden size: {config['hidden_size']}")
        
        return 0
        
    except Exception as e:
        print(f"Error getting model info: {e}")
        return 1


def cmd_popular(args):
    """Show popular models."""
    popular_models = [
        {
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "description": "Small, efficient model good for testing",
            "size": "~2GB",
        },
        {
            "model_id": "microsoft/Phi-3-medium-4k-instruct", 
            "description": "Medium-sized model with good performance",
            "size": "~8GB",
        },
        {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "description": "Popular chat model from Meta",
            "size": "~13GB",
        },
        {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "High-quality instruction-following model",
            "size": "~13GB", 
        },
    ]
    
    print("Popular models for MLX:")
    print()
    
    if HF_AVAILABLE:
        hf_manager = get_hf_manager()
        
        for model in popular_models:
            is_local = hf_manager.is_model_available_locally(model["model_id"])
            status = "✓ Local" if is_local else "  Remote"
            
            print(f"{status} {model['model_id']}")
            print(f"       {model['description']}")
            print(f"       Size: {model['size']}")
            print()
    else:
        for model in popular_models:
            print(f"  {model['model_id']}")
            print(f"    {model['description']}")
            print(f"    Size: {model['size']}")
            print()
        
        print("Note: Install huggingface_hub and transformers to check local availability")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLX Speculative Server - Enhanced CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--model", help="Model name")
    serve_parser.add_argument("--model-path", help="Model path or HF model ID")
    serve_parser.add_argument("--models", help="JSON string or file with model definitions")
    serve_parser.add_argument("--config", help="Configuration file path")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    serve_parser.set_defaults(func=cmd_serve)
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download model from Hugging Face")
    download_parser.add_argument("model_id", help="Hugging Face model ID")
    download_parser.add_argument("--revision", help="Model revision/branch")
    download_parser.add_argument("--quantize", choices=["q4", "q8"], help="Quantization method")
    download_parser.add_argument("--force-download", action="store_true", help="Force re-download")
    download_parser.add_argument("--force-convert", action="store_true", help="Force re-conversion")
    download_parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    download_parser.add_argument("--test", action="store_true", help="Test model after download")
    download_parser.set_defaults(func=cmd_download)
    
    # List local models
    list_parser = subparsers.add_parser("list", help="List local models")
    list_parser.set_defaults(func=cmd_list_local)
    
    # Search models
    search_parser = subparsers.add_parser("search", help="Search Hugging Face models")
    search_parser.add_argument("query", nargs="?", help="Search query")
    search_parser.add_argument("--task", help="Task type (e.g., text-generation)")
    search_parser.add_argument("--library", help="Library name (e.g., transformers)")
    search_parser.add_argument("--tags", nargs="+", help="Tags to filter by")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    search_parser.set_defaults(func=cmd_search)
    
    # Delete model
    delete_parser = subparsers.add_parser("delete", help="Delete local model")
    delete_parser.add_argument("model_id", help="Model ID to delete")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)
    
    # Model info
    info_parser = subparsers.add_parser("info", help="Get model information")
    info_parser.add_argument("model_id", help="Model ID")
    info_parser.set_defaults(func=cmd_info)
    
    # Popular models
    popular_parser = subparsers.add_parser("popular", help="Show popular models")
    popular_parser.set_defaults(func=cmd_popular)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
