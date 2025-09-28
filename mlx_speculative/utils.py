# Copyright © 2025 Manus AI

import time
from typing import Any, Dict, Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer

from .core import SpeculativeEngine
from .models import ModelConfig, load_model_pair
from .sample_utils import create_sampler


def load(
    model_path: str,
    draft_model_path: Optional[str] = None,
    trust_remote_code: bool = False,
    adapter_path: Optional[str] = None,
    auto_draft: bool = True,
    draft_layers_ratio: float = 0.5,
) -> tuple[SpeculativeEngine, TokenizerWrapper]:
    """
    Load a speculative decoding engine with target and draft models.
    
    Args:
        model_path: Path to the target model
        draft_model_path: Optional path to the draft model
        trust_remote_code: Whether to trust remote code for tokenizer
        adapter_path: Optional path to LoRA adapters
        auto_draft: Whether to automatically create a draft model
        draft_layers_ratio: Ratio of layers for auto-generated draft model
        
    Returns:
        Tuple of (SpeculativeEngine, TokenizerWrapper)
    """
    config = ModelConfig(
        model_path=model_path,
        draft_model_path=draft_model_path,
        trust_remote_code=trust_remote_code,
        adapter_path=adapter_path,
    )
    
    target_model, draft_model, tokenizer = load_model_pair(
        config, auto_draft=auto_draft, draft_layers_ratio=draft_layers_ratio
    )
    
    engine = SpeculativeEngine(
        target_model=target_model,
        draft_model=draft_model,
        num_draft_tokens=4,
        auto_draft=auto_draft,
    )
    
    return engine, tokenizer


def generate(
    engine: SpeculativeEngine,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    verbose: bool = False,
) -> str:
    """
    Generate text using speculative decoding.
    
    Args:
        engine: Speculative decoding engine
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty factor
        verbose: Whether to print generation statistics
        
    Returns:
        Generated text string
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    
    # Encode prompt
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None]  # Add batch dimension
    
    # Create sampler
    sampler = create_sampler(temperature=temperature, top_p=top_p, top_k=top_k)
    
    # Generate tokens
    generated_tokens = []
    start_time = time.time()
    
    for tokens, metadata in engine.generate_step(
        prompts=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        sampler=sampler,
    ):
        generated_tokens.append(tokens[0])  # Take first (and only) sequence
        
        if verbose and metadata["step"] % 10 == 0:
            print(f"Step {metadata['step']}: "
                  f"Acceptance rate: {metadata['acceptance_rate']:.2f}, "
                  f"Throughput: {metadata['throughput']:.1f} tok/s")
    
    # Concatenate all generated tokens
    if generated_tokens:
        all_tokens = mx.concatenate(generated_tokens, axis=0)
        generated_text = tokenizer.decode(all_tokens.tolist())
    else:
        generated_text = ""
    
    if verbose:
        elapsed_time = time.time() - start_time
        total_tokens = sum(t.shape[0] for t in generated_tokens)
        print(f"\nGeneration complete:")
        print(f"Total tokens: {total_tokens}")
        print(f"Time: {elapsed_time:.2f}s")
        print(f"Throughput: {total_tokens / elapsed_time:.1f} tok/s")
        print(f"Final acceptance rate: {engine.stats['acceptance_rate']:.2f}")
    
    return generated_text


def batch_generate(
    engine: SpeculativeEngine,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    verbose: bool = False,
    format_prompts: bool = True,
) -> List[str]:
    """
    Generate text for multiple prompts using batched speculative decoding.
    
    Args:
        engine: Speculative decoding engine
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of input prompt strings
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty factor
        verbose: Whether to print generation statistics
        format_prompts: Whether to apply chat template formatting
        
    Returns:
        List of generated text strings
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    
    # Format prompts if requested
    if format_prompts:
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            formatted_prompts.append(formatted)
    else:
        formatted_prompts = prompts
    
    # Set up tokenizer for left padding
    original_padding_side = getattr(tokenizer._tokenizer, 'padding_side', 'right')
    tokenizer._tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer._tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.pad_token_id = tokenizer.eos_token_id
    
    try:
        # Encode and pad prompts
        encoded = tokenizer._tokenizer(formatted_prompts, padding=True, return_tensors="np")
        prompt_tokens = mx.array(encoded['input_ids'])
        
        # Create sampler
        sampler = create_sampler(temperature=temperature, top_p=top_p, top_k=top_k)
        
        # Generate tokens
        batch_size = len(prompts)
        generated_tokens = [[] for _ in range(batch_size)]
        start_time = time.time()
        
        for tokens, metadata in engine.generate_step(
            prompts=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            sampler=sampler,
        ):
            # tokens shape: [batch_size, num_new_tokens]
            for i in range(batch_size):
                generated_tokens[i].append(tokens[i])
            
            if verbose and metadata["step"] % 10 == 0:
                print(f"Step {metadata['step']}: "
                      f"Acceptance rate: {metadata['acceptance_rate']:.2f}, "
                      f"Throughput: {metadata['throughput']:.1f} tok/s")
        
        # Decode generated text
        responses = []
        for i in range(batch_size):
            if generated_tokens[i]:
                all_tokens = mx.concatenate(generated_tokens[i], axis=0)
                text = tokenizer.decode(all_tokens.tolist())
                # Clean up padding and EOS tokens
                text = text.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
                responses.append(text.strip())
            else:
                responses.append("")
        
        if verbose:
            elapsed_time = time.time() - start_time
            total_tokens = sum(sum(t.shape[0] for t in seq) for seq in generated_tokens)
            print(f"\nBatch generation complete:")
            print(f"Batch size: {batch_size}")
            print(f"Total tokens: {total_tokens}")
            print(f"Time: {elapsed_time:.2f}s")
            print(f"Throughput: {total_tokens / elapsed_time:.1f} tok/s")
            print(f"Final acceptance rate: {engine.stats['acceptance_rate']:.2f}")
        
        return responses
    
    finally:
        # Restore original padding side
        tokenizer._tokenizer.padding_side = original_padding_side


def stream_generate(
    engine: SpeculativeEngine,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
) -> Generator[str, None, None]:
    """
    Generate text using speculative decoding with streaming output.
    
    Args:
        engine: Speculative decoding engine
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty factor
        
    Yields:
        Generated text segments as they become available
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    
    # Encode prompt
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None]  # Add batch dimension
    
    # Create sampler
    sampler = create_sampler(temperature=temperature, top_p=top_p, top_k=top_k)
    
    # Initialize detokenizer
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    
    # Generate tokens and stream text
    for tokens, metadata in engine.generate_step(
        prompts=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        sampler=sampler,
    ):
        # Add tokens to detokenizer
        sequence_tokens = tokens[0]  # Take first (and only) sequence
        for token in sequence_tokens:
            detokenizer.add_token(token.item())
            
        # Yield the new text segment
        yield detokenizer.last_segment
    
    # Finalize and yield any remaining text
    detokenizer.finalize()
    if detokenizer.last_segment:
        yield detokenizer.last_segment


def benchmark(
    engine: SpeculativeEngine,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.0,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark the speculative decoding engine.
    
    Args:
        engine: Speculative decoding engine
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompts to benchmark with
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "runs": [],
        "avg_throughput": 0.0,
        "avg_acceptance_rate": 0.0,
        "avg_latency": 0.0,
    }
    
    for run in range(num_runs):
        start_time = time.time()
        
        responses = batch_generate(
            engine=engine,
            tokenizer=tokenizer,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate metrics
        total_tokens = sum(len(tokenizer.encode(response)) for response in responses)
        throughput = total_tokens / elapsed_time
        acceptance_rate = engine.stats["acceptance_rate"]
        
        run_result = {
            "run": run + 1,
            "elapsed_time": elapsed_time,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "acceptance_rate": acceptance_rate,
            "batch_size": len(prompts),
        }
        
        results["runs"].append(run_result)
    
    # Calculate averages
    results["avg_throughput"] = sum(r["throughput"] for r in results["runs"]) / num_runs
    results["avg_acceptance_rate"] = sum(r["acceptance_rate"] for r in results["runs"]) / num_runs
    results["avg_latency"] = sum(r["elapsed_time"] for r in results["runs"]) / num_runs
    
    return results
