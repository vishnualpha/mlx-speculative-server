# Copyright © 2025 Manus AI

import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from .models import BatchedKVCache
from .sample_utils import top_p_sampling


class SpeculativeEngine:
    """
    Core speculative decoding engine that implements batched speculative generation
    with automatic draft model selection and verification.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: Optional[nn.Module] = None,
        num_draft_tokens: int = 4,
        acceptance_threshold: float = 0.8,
        auto_draft: bool = True,
    ):
        """
        Initialize the speculative decoding engine.
        
        Args:
            target_model: The main model for generation
            draft_model: Optional draft model for speculation
            num_draft_tokens: Number of tokens to speculate ahead
            acceptance_threshold: Threshold for accepting draft tokens
            auto_draft: Whether to automatically derive a draft model
        """
        self.target_model = target_model
        self.draft_model = draft_model or self._create_auto_draft_model(target_model)
        self.num_draft_tokens = num_draft_tokens
        self.acceptance_threshold = acceptance_threshold
        self.auto_draft = auto_draft
        
        # Performance tracking
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "draft_tokens": 0,
            "acceptance_rate": 0.0,
            "throughput": 0.0,
        }
    
    def _create_auto_draft_model(self, target_model: nn.Module) -> nn.Module:
        """
        Automatically create a draft model from the target model.
        This could involve layer pruning, width reduction, or other techniques.
        """
        # For now, return the same model - in practice, this would create
        # a smaller, faster version of the target model
        return target_model
    
    def _get_kv_heads(self, model: nn.Module) -> List[int]:
        """Get the number of KV heads for each layer."""
        if hasattr(model, 'n_kv_heads'):
            if isinstance(model.n_kv_heads, int):
                return [model.n_kv_heads] * len(model.layers)
            else:
                return model.n_kv_heads
        else:
            # Fallback to attention heads
            return [getattr(layer.attention, 'n_heads', 8) for layer in model.layers]
    
    def _create_batched_cache(self, model: nn.Module, batch_size: int) -> List[BatchedKVCache]:
        """Create batched KV cache for the model."""
        kv_heads = self._get_kv_heads(model)
        head_dim = getattr(model, 'head_dim', 64)
        
        return [
            BatchedKVCache(head_dim, n_heads, batch_size)
            for n_heads in kv_heads
        ]
    
    def _draft_step(
        self,
        tokens: mx.array,
        draft_cache: List[BatchedKVCache],
        sampler: Callable,
    ) -> Tuple[mx.array, mx.array]:
        """Generate one token using the draft model."""
        logits = self.draft_model(tokens, cache=draft_cache)
        logits = logits[:, -1, :]  # Take last token logits
        
        # Sample from draft model
        probs = mx.softmax(logits, axis=-1)
        next_tokens = sampler(logits)
        
        return next_tokens, probs
    
    def _target_step(
        self,
        tokens: mx.array,
        target_cache: List[BatchedKVCache],
        sampler: Callable,
        num_tokens: int = 1,
    ) -> Tuple[mx.array, mx.array]:
        """Generate tokens using the target model."""
        logits = self.target_model(tokens, cache=target_cache)
        
        if num_tokens == 1:
            logits = logits[:, -1, :]
            probs = mx.softmax(logits, axis=-1)
            next_tokens = sampler(logits)
            return next_tokens, probs
        else:
            # Multiple tokens for verification
            logits = logits[:, -num_tokens:, :]
            probs = mx.softmax(logits, axis=-1)
            tokens_list = []
            
            for i in range(num_tokens):
                token_logits = logits[:, i, :]
                next_token = sampler(token_logits)
                tokens_list.append(next_token)
            
            return mx.stack(tokens_list, axis=1), probs
    
    def _verify_draft_tokens(
        self,
        draft_tokens: mx.array,
        target_probs: mx.array,
        draft_probs: mx.array,
    ) -> Tuple[mx.array, int]:
        """
        Verify draft tokens against target model probabilities.
        
        Returns:
            accepted_tokens: The tokens that were accepted
            num_accepted: Number of accepted tokens
        """
        batch_size, num_draft = draft_tokens.shape
        accepted_tokens = []
        num_accepted = 0
        
        for i in range(num_draft):
            # Get probabilities for the draft token
            draft_token = draft_tokens[:, i]
            target_prob = target_probs[:, i, draft_token]
            draft_prob = draft_probs[:, i, draft_token]
            
            # Acceptance probability based on target/draft ratio
            acceptance_prob = mx.minimum(mx.ones_like(target_prob), target_prob / draft_prob)
            
            # Sample acceptance decision
            accept = mx.random.bernoulli(acceptance_prob)
            
            if mx.all(accept):
                accepted_tokens.append(draft_token)
                num_accepted += 1
            else:
                break
        
        if accepted_tokens:
            return mx.stack(accepted_tokens, axis=1), num_accepted
        else:
            return mx.zeros((batch_size, 0), dtype=mx.int32), 0
    
    def generate_step(
        self,
        prompts: mx.array,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        sampler: Optional[Callable] = None,
    ) -> Generator[Tuple[mx.array, Dict[str, Any]], None, None]:
        """
        Generate tokens using speculative decoding.
        
        Args:
            prompts: Input prompts as token arrays [batch_size, seq_len]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            sampler: Custom sampler function
            
        Yields:
            Tuple of (tokens, metadata) for each generation step
        """
        batch_size = prompts.shape[0]
        
        # Create samplers
        if sampler is None:
            if temperature == 0.0:
                sampler = lambda x: mx.argmax(x, axis=-1)
            else:
                sampler = lambda x: top_p_sampling(x, top_p, temperature)
        
        # Initialize caches
        target_cache = self._create_batched_cache(self.target_model, batch_size)
        draft_cache = self._create_batched_cache(self.draft_model, batch_size)
        
        # Current tokens
        current_tokens = prompts
        
        # Performance tracking
        start_time = time.time()
        total_generated = 0
        total_accepted = 0
        total_draft = 0
        
        for step in range(max_tokens):
            # Generate draft tokens
            draft_tokens_list = []
            draft_probs_list = []
            
            # Start with current tokens for draft generation
            draft_input = current_tokens
            
            for _ in range(self.num_draft_tokens):
                draft_token, draft_prob = self._draft_step(draft_input, draft_cache, sampler)
                draft_tokens_list.append(draft_token)
                draft_probs_list.append(draft_prob)
                
                # Append draft token for next iteration
                draft_input = mx.concatenate([draft_input, draft_token[:, None]], axis=1)
                total_draft += batch_size
            
            # Stack draft tokens and probabilities
            draft_tokens = mx.stack(draft_tokens_list, axis=1)  # [batch_size, num_draft]
            draft_probs = mx.stack(draft_probs_list, axis=1)    # [batch_size, num_draft, vocab_size]
            
            # Verify with target model
            target_input = mx.concatenate([current_tokens, draft_tokens], axis=1)
            target_tokens, target_probs = self._target_step(
                target_input, target_cache, sampler, self.num_draft_tokens + 1
            )
            
            # Verify draft tokens
            accepted_tokens, num_accepted = self._verify_draft_tokens(
                draft_tokens, target_probs[:, :-1, :], draft_probs
            )
            
            # Add one more token from target model
            if num_accepted < self.num_draft_tokens:
                # Use target model's next token after rejection point
                extra_token = target_tokens[:, num_accepted:num_accepted+1]
                if accepted_tokens.shape[1] > 0:
                    final_tokens = mx.concatenate([accepted_tokens, extra_token], axis=1)
                else:
                    final_tokens = extra_token
            else:
                # All draft tokens accepted, add target's final token
                extra_token = target_tokens[:, -1:]
                final_tokens = mx.concatenate([accepted_tokens, extra_token], axis=1)
            
            # Update current tokens
            current_tokens = mx.concatenate([current_tokens, final_tokens], axis=1)
            
            # Update statistics
            total_generated += final_tokens.shape[1] * batch_size
            total_accepted += num_accepted * batch_size
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            throughput = total_generated / elapsed_time if elapsed_time > 0 else 0
            acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0
            
            metadata = {
                "step": step,
                "tokens_generated": final_tokens.shape[1],
                "tokens_accepted": num_accepted,
                "acceptance_rate": acceptance_rate,
                "throughput": throughput,
                "total_tokens": total_generated,
            }
            
            yield final_tokens, metadata
            
            # Check for early stopping (EOS tokens, etc.)
            # This would be implemented based on tokenizer
            
        # Update final statistics
        self.stats.update({
            "total_tokens": total_generated,
            "accepted_tokens": total_accepted,
            "draft_tokens": total_draft,
            "acceptance_rate": acceptance_rate,
            "throughput": throughput,
        })
