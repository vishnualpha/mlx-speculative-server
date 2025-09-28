# Copyright © 2025 Manus AI

from typing import Optional

import mlx.core as mx


def top_p_sampling(logits: mx.array, top_p: float, temperature: float = 1.0) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.
    
    Args:
        logits: Input logits [batch_size, vocab_size] or [vocab_size]
        top_p: Cumulative probability threshold
        temperature: Sampling temperature
        
    Returns:
        Sampled token indices
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = mx.sort(probs, axis=-1)
    sorted_probs = sorted_probs[..., ::-1]  # Reverse to get descending order
    sorted_indices = sorted_indices[..., ::-1]
    
    # Calculate cumulative probabilities
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
    
    # Create mask for tokens to keep
    mask = cumsum_probs <= top_p
    
    # Always keep at least the first token
    if mask.ndim == 1:
        mask = mx.concatenate([mx.array([True]), mask[1:]])
    else:
        first_col = mx.ones((*mask.shape[:-1], 1), dtype=mx.bool_)
        mask = mx.concatenate([first_col, mask[..., 1:]], axis=-1)
    
    # Zero out probabilities for tokens not in top-p
    filtered_probs = mx.where(mask, sorted_probs, 0.0)
    
    # Renormalize
    filtered_probs = filtered_probs / mx.sum(filtered_probs, axis=-1, keepdims=True)
    
    # Sample from filtered distribution
    sampled_indices = mx.random.categorical(mx.log(filtered_probs + 1e-8), axis=-1)
    
    # Map back to original indices
    if logits.ndim == 1:
        return sorted_indices[sampled_indices]
    else:
        batch_indices = mx.arange(logits.shape[0])
        return sorted_indices[batch_indices, sampled_indices]


def top_k_sampling(logits: mx.array, top_k: int, temperature: float = 1.0) -> mx.array:
    """
    Apply top-k sampling to logits.
    
    Args:
        logits: Input logits [batch_size, vocab_size] or [vocab_size]
        top_k: Number of top tokens to consider
        temperature: Sampling temperature
        
    Returns:
        Sampled token indices
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # Get top-k values and indices
    top_k_logits, top_k_indices = mx.topk(logits, k=top_k, axis=-1)
    
    # Sample from top-k distribution
    probs = mx.softmax(top_k_logits, axis=-1)
    sampled_indices = mx.random.categorical(mx.log(probs + 1e-8), axis=-1)
    
    # Map back to original indices
    if logits.ndim == 1:
        return top_k_indices[sampled_indices]
    else:
        batch_indices = mx.arange(logits.shape[0])
        return top_k_indices[batch_indices, sampled_indices]


def temperature_sampling(logits: mx.array, temperature: float) -> mx.array:
    """
    Apply temperature sampling to logits.
    
    Args:
        logits: Input logits [batch_size, vocab_size] or [vocab_size]
        temperature: Sampling temperature
        
    Returns:
        Sampled token indices
    """
    scaled_logits = logits / temperature
    return mx.random.categorical(scaled_logits, axis=-1)


def greedy_sampling(logits: mx.array) -> mx.array:
    """
    Apply greedy sampling (argmax) to logits.
    
    Args:
        logits: Input logits [batch_size, vocab_size] or [vocab_size]
        
    Returns:
        Token indices with highest probability
    """
    return mx.argmax(logits, axis=-1)


def create_sampler(
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
):
    """
    Create a sampling function based on the given parameters.
    
    Args:
        temperature: Sampling temperature (0.0 for greedy)
        top_p: Top-p threshold for nucleus sampling
        top_k: Top-k threshold for top-k sampling
        
    Returns:
        Sampling function that takes logits and returns token indices
    """
    if temperature == 0.0:
        return greedy_sampling
    elif top_k is not None:
        return lambda logits: top_k_sampling(logits, top_k, temperature)
    elif top_p < 1.0:
        return lambda logits: top_p_sampling(logits, top_p, temperature)
    else:
        return lambda logits: temperature_sampling(logits, temperature)


def apply_repetition_penalty(
    logits: mx.array,
    generated_tokens: mx.array,
    penalty: float = 1.0,
) -> mx.array:
    """
    Apply repetition penalty to logits based on previously generated tokens.
    
    Args:
        logits: Input logits [batch_size, vocab_size]
        generated_tokens: Previously generated tokens [batch_size, seq_len]
        penalty: Repetition penalty factor (> 1.0 discourages repetition)
        
    Returns:
        Modified logits with repetition penalty applied
    """
    if penalty == 1.0 or generated_tokens.size == 0:
        return logits
    
    # Get unique tokens for each sequence in the batch
    batch_size = logits.shape[0]
    
    for i in range(batch_size):
        seq_tokens = generated_tokens[i]
        unique_tokens = mx.unique(seq_tokens)
        
        # Apply penalty to repeated tokens
        for token in unique_tokens:
            if logits[i, token] > 0:
                logits = logits.at[i, token].set(logits[i, token] / penalty)
            else:
                logits = logits.at[i, token].set(logits[i, token] * penalty)
    
    return logits


def apply_frequency_penalty(
    logits: mx.array,
    token_counts: mx.array,
    penalty: float = 0.0,
) -> mx.array:
    """
    Apply frequency penalty to logits based on token occurrence counts.
    
    Args:
        logits: Input logits [batch_size, vocab_size]
        token_counts: Count of each token occurrence [batch_size, vocab_size]
        penalty: Frequency penalty factor
        
    Returns:
        Modified logits with frequency penalty applied
    """
    if penalty == 0.0:
        return logits
    
    return logits - penalty * token_counts


def apply_presence_penalty(
    logits: mx.array,
    token_presence: mx.array,
    penalty: float = 0.0,
) -> mx.array:
    """
    Apply presence penalty to logits based on token presence.
    
    Args:
        logits: Input logits [batch_size, vocab_size]
        token_presence: Binary presence of each token [batch_size, vocab_size]
        penalty: Presence penalty factor
        
    Returns:
        Modified logits with presence penalty applied
    """
    if penalty == 0.0:
        return logits
    
    return logits - penalty * token_presence
