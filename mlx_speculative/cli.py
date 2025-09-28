# Copyright © 2025 Manus AI

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from .models import ModelConfig
from .server import run_server, server


def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="MLX Speculative Decoding Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with a single model
  mlx-speculative serve --model llama3 --model-path /path/to/llama3

  # Start server with multiple models
  mlx-speculative serve --config models.json

  # Start server with custom settings
  mlx-speculative serve --model llama3 --model-path /path/to/llama3 \\
                       --host 0.0.0.0 --port 8080 --max-batch-size 16

  # Load a model interactively
  mlx-speculative load-model --name llama3 --path /path/to/llama3

  # Get server statistics
  mlx-speculative stats
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the server")
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    serve_parser.add_argument(
        "--model", help="Model name (for single model setup)"
    )
    serve_parser.add_argument(
        "--model-path", help="Path to model (for single model setup)"
    )
    serve_parser.add_argument(
        "--draft-model-path", help="Path to draft model (optional)"
    )
    serve_parser.add_argument(
        "--config", help="Path to JSON configuration file"
    )
    serve_parser.add_argument(
        "--max-batch-size", type=int, default=8, help="Maximum batch size (default: 8)"
    )
    serve_parser.add_argument(
        "--batch-timeout", type=float, default=0.1, help="Batch timeout in seconds (default: 0.1)"
    )
    serve_parser.add_argument(
        "--max-concurrent", type=int, default=100, help="Maximum concurrent requests (default: 100)"
    )
    serve_parser.add_argument(
        "--auto-draft", action="store_true", default=True, help="Enable automatic draft model creation"
    )
    serve_parser.add_argument(
        "--draft-layers-ratio", type=float, default=0.5, help="Draft model layers ratio (default: 0.5)"
    )
    serve_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code for tokenizer"
    )
    serve_parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    
    # Load model command
    load_parser = subparsers.add_parser("load-model", help="Load a model into running server")
    load_parser.add_argument("--name", required=True, help="Model name")
    load_parser.add_argument("--path", required=True, help="Path to model")
    load_parser.add_argument("--draft-path", help="Path to draft model")
    load_parser.add_argument("--set-default", action="store_true", help="Set as default model")
    load_parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get server statistics")
    stats_parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List loaded models")
    models_parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    
    # Generate command (for testing)
    gen_parser = subparsers.add_parser("generate", help="Generate text (for testing)")
    gen_parser.add_argument("--prompt", required=True, help="Input prompt")
    gen_parser.add_argument("--model", help="Model name")
    gen_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens")
    gen_parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    gen_parser.add_argument("--top-p", type=float, default=1.0, help="Top-p")
    gen_parser.add_argument("--stream", action="store_true", help="Stream output")
    gen_parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    
    return parser


def serve_command(args):
    """Handle the serve command."""
    models = {}
    
    if args.config:
        # Load from config file
        config = load_config_file(args.config)
        models = config.get("models", {})
        
        # Override server settings from config
        if "server" in config:
            server_config = config["server"]
            args.host = server_config.get("host", args.host)
            args.port = server_config.get("port", args.port)
            args.max_batch_size = server_config.get("max_batch_size", args.max_batch_size)
            args.batch_timeout = server_config.get("batch_timeout", args.batch_timeout)
            args.max_concurrent = server_config.get("max_concurrent", args.max_concurrent)
    
    elif args.model and args.model_path:
        # Single model setup
        models[args.model] = args.model_path
    
    else:
        print("Error: Either --config or both --model and --model-path must be provided")
        sys.exit(1)
    
    # Configure server
    from .server import SpeculativeServer
    global server
    server = SpeculativeServer(
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
        max_concurrent_requests=args.max_concurrent,
    )
    
    # Load models
    for name, path in models.items():
        config = ModelConfig(
            model_path=path,
            draft_model_path=getattr(args, 'draft_model_path', None),
            trust_remote_code=args.trust_remote_code,
        )
        
        server.model_manager.load_model(
            name=name,
            config=config,
            auto_draft=args.auto_draft,
            draft_layers_ratio=args.draft_layers_ratio,
            set_default=(name == list(models.keys())[0])
        )
        
        print(f"Loaded model: {name} from {path}")
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Loaded models: {list(models.keys())}")
    print(f"Default model: {server.model_manager.default_model}")
    
    # Run server
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


def load_model_command(args):
    """Handle the load-model command."""
    # This would typically make an API call to a running server
    # For now, we'll just show what would be done
    print(f"Would load model '{args.name}' from '{args.path}'")
    if args.draft_path:
        print(f"With draft model from '{args.draft_path}'")
    if args.set_default:
        print("Would set as default model")


def stats_command(args):
    """Handle the stats command."""
    import requests
    
    try:
        response = requests.get(f"{args.server_url}/stats")
        response.raise_for_status()
        stats = response.json()
        
        print("Server Statistics:")
        print(f"  Uptime: {stats['uptime']:.2f} seconds")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Average throughput: {stats['average_throughput']:.2f} tok/s")
        print(f"  Active requests: {stats['active_requests']}")
        print(f"  Loaded models: {', '.join(stats['loaded_models'])}")
        print(f"  Default model: {stats['default_model']}")
        
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)


def models_command(args):
    """Handle the models command."""
    import requests
    
    try:
        response = requests.get(f"{args.server_url}/models")
        response.raise_for_status()
        data = response.json()
        
        print("Loaded Models:")
        for model in data["models"]:
            print(f"  - {model}")
            
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)


def generate_command(args):
    """Handle the generate command."""
    import requests
    
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": args.stream,
    }
    
    if args.model:
        payload["model"] = args.model
    
    try:
        if args.stream:
            # Streaming request
            response = requests.post(
                f"{args.server_url}/generate",
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            print("Generated text:")
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            print(chunk['text'], end='', flush=True)
                        except json.JSONDecodeError:
                            pass
            print()  # New line at end
        else:
            # Regular request
            response = requests.post(f"{args.server_url}/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            
            print("Generated text:")
            print(result["text"])
            print(f"\nUsage: {result['usage']}")
            
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)


def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "max_batch_size": 8,
            "batch_timeout": 0.1,
            "max_concurrent": 100
        },
        "models": {
            "llama3-8b": "/path/to/llama3-8b",
            "llama3-8b-instruct": "/path/to/llama3-8b-instruct",
            "phi3-mini": "/path/to/phi3-mini"
        },
        "generation": {
            "auto_draft": True,
            "draft_layers_ratio": 0.5,
            "trust_remote_code": False
        }
    }
    
    with open("mlx_speculative_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created sample configuration file: mlx_speculative_config.json")


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "serve":
        serve_command(args)
    elif args.command == "load-model":
        load_model_command(args)
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "models":
        models_command(args)
    elif args.command == "generate":
        generate_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
