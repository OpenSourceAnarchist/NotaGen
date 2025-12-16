"""
Numerically Stable Sampling Module for NotaGen

This module provides robust sampling functions for autoregressive text generation,
specifically designed to handle edge cases that cause numerical instability:

- NaN/Inf values from float16 overflow
- Probability distributions that don't sum to exactly 1.0
- Edge cases where all probabilities are zeroed out by filtering

The implementation avoids the `samplings` library in favor of direct NumPy operations
with explicit handling of numerical edge cases.

Author: NotaGen Team
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
import warnings


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    top_k: int = 9
    top_p: float = 0.9
    temperature: float = 1.2
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


def _sanitize_probabilities(
    probs: np.ndarray, 
    eps: float = 1e-10
) -> np.ndarray:
    """
    Sanitize a probability distribution to ensure numerical stability.
    
    Handles:
    - NaN values (replaced with 0)
    - Inf values (replaced with large finite value, then renormalized)
    - Negative values (clipped to 0)
    - Zero sum (falls back to uniform distribution)
    
    Args:
        probs: Input probability array (may be unnormalized or contain invalid values)
        eps: Small epsilon for numerical stability
        
    Returns:
        Valid probability distribution that sums to 1.0
    """
    probs = np.asarray(probs, dtype=np.float64)
    
    # Handle NaN - replace with 0
    nan_mask = np.isnan(probs)
    if np.any(nan_mask):
        probs = np.where(nan_mask, 0.0, probs)
    
    # Handle Inf - replace with large finite value
    inf_mask = np.isinf(probs)
    if np.any(inf_mask):
        max_finite = np.max(probs[~inf_mask]) if np.any(~inf_mask) else 1.0
        probs = np.where(inf_mask, max_finite * 1e6, probs)
    
    # Clip negative values
    probs = np.maximum(probs, 0.0)
    
    # Normalize
    total = probs.sum()
    if total <= eps:
        # Fall back to uniform distribution if everything was zeroed
        warnings.warn("All probabilities were zero or invalid; using uniform distribution")
        return np.ones_like(probs) / len(probs)
    
    probs = probs / total
    
    # Final safety check - ensure sum is exactly 1.0 for numpy.random.choice
    # This handles floating-point accumulation errors
    probs = probs / probs.sum()
    
    return probs


def top_k_filtering(
    probs: np.ndarray,
    top_k: int
) -> np.ndarray:
    """
    Apply top-k filtering to a probability distribution.
    
    Keeps only the top-k highest probability tokens, setting all others to zero,
    then renormalizes.
    
    Args:
        probs: Probability distribution array
        top_k: Number of top tokens to keep (0 means no filtering)
        
    Returns:
        Filtered and renormalized probability distribution
    """
    if top_k <= 0 or top_k >= len(probs):
        return probs
    
    probs = np.asarray(probs, dtype=np.float64)
    
    # Find the k-th largest value
    # Using argpartition is O(n) vs O(n log n) for full sort
    top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
    
    # Create output with zeros
    filtered = np.zeros_like(probs)
    filtered[top_k_indices] = probs[top_k_indices]
    
    # Renormalize
    total = filtered.sum()
    if total > 0:
        filtered = filtered / total
    else:
        # Edge case: all top-k values were 0
        # Return uniform over top-k positions
        filtered[top_k_indices] = 1.0 / top_k
    
    return filtered


def top_p_filtering(
    probs: np.ndarray,
    top_p: float
) -> np.ndarray:
    """
    Apply nucleus (top-p) filtering to a probability distribution.
    
    Keeps the smallest set of tokens whose cumulative probability exceeds top_p,
    then renormalizes.
    
    Args:
        probs: Probability distribution array
        top_p: Cumulative probability threshold (1.0 means no filtering)
        
    Returns:
        Filtered and renormalized probability distribution
    """
    if top_p >= 1.0:
        return probs
    
    probs = np.asarray(probs, dtype=np.float64)
    
    # Sort in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Find cumulative probabilities
    cumsum = np.cumsum(sorted_probs)
    
    # Find cutoff index (first index where cumsum exceeds top_p)
    cutoff_idx = np.searchsorted(cumsum, top_p, side='right')
    cutoff_idx = max(1, cutoff_idx)  # Keep at least one token
    
    # Create filtered output
    filtered = np.zeros_like(probs)
    keep_indices = sorted_indices[:cutoff_idx]
    filtered[keep_indices] = probs[keep_indices]
    
    # Renormalize
    total = filtered.sum()
    if total > 0:
        filtered = filtered / total
    else:
        # Edge case: shouldn't happen but handle gracefully
        filtered[sorted_indices[0]] = 1.0
    
    return filtered


def apply_temperature(
    probs: np.ndarray,
    temperature: float,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Apply temperature scaling to a probability distribution.
    
    Temperature < 1.0 makes distribution sharper (more confident)
    Temperature > 1.0 makes distribution flatter (more random)
    Temperature = 1.0 leaves distribution unchanged
    
    Implementation note: We work in log-space to avoid numerical issues
    with very small probabilities.
    
    Args:
        probs: Probability distribution array
        temperature: Temperature parameter (must be positive)
        eps: Small epsilon for numerical stability
        
    Returns:
        Temperature-scaled probability distribution
    """
    if temperature == 1.0:
        return probs
    
    probs = np.asarray(probs, dtype=np.float64)
    
    # Add small epsilon to avoid log(0)
    probs_safe = np.maximum(probs, eps)
    
    # Work in log space: new_probs ∝ probs^(1/T) = exp(log(probs)/T)
    log_probs = np.log(probs_safe)
    scaled_log_probs = log_probs / temperature
    
    # Subtract max for numerical stability before exp
    scaled_log_probs = scaled_log_probs - np.max(scaled_log_probs)
    
    # Exponentiate and normalize
    scaled_probs = np.exp(scaled_log_probs)
    
    # Zero out tokens that were originally zero
    scaled_probs = np.where(probs < eps, 0.0, scaled_probs)
    
    # Renormalize
    total = scaled_probs.sum()
    if total > 0:
        scaled_probs = scaled_probs / total
    else:
        # Fallback
        return probs
    
    return scaled_probs


def top_k_top_p_filtering(
    probs: np.ndarray,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    sanitize: bool = True
) -> np.ndarray:
    """
    Apply combined top-k, top-p, and temperature filtering.
    
    This is the main entry point for probability filtering, applying
    all three transformations in the correct order with proper
    numerical stability handling.
    
    Order of operations:
    1. Sanitize input (handle NaN/Inf/negative values)
    2. Top-k filtering (keep only k highest)
    3. Top-p filtering (nucleus sampling)
    4. Temperature scaling
    5. Final sanitization to ensure valid distribution
    
    Args:
        probs: Raw probability distribution (from softmax)
        top_k: Number of top tokens to keep (0 = no filtering)
        top_p: Cumulative probability threshold (1.0 = no filtering)
        temperature: Temperature for sampling (1.0 = no scaling)
        sanitize: Whether to sanitize input/output for numerical stability
        
    Returns:
        Filtered probability distribution ready for sampling
    """
    if sanitize:
        probs = _sanitize_probabilities(probs)
    
    # Apply filters in order
    if top_k > 0:
        probs = top_k_filtering(probs, top_k)
    
    if top_p < 1.0:
        probs = top_p_filtering(probs, top_p)
    
    if temperature != 1.0:
        probs = apply_temperature(probs, temperature)
    
    if sanitize:
        probs = _sanitize_probabilities(probs)
    
    return probs


def sample_token(
    probs: np.ndarray,
    config: Optional[SamplingConfig] = None,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> int:
    """
    Sample a token from a probability distribution with top-k/top-p filtering.
    
    This is the primary sampling function for autoregressive generation.
    It handles all numerical edge cases and provides deterministic sampling
    when a seed is provided.
    
    Args:
        probs: Raw probability distribution (from model softmax)
        config: Optional SamplingConfig object (overrides other params if provided)
        top_k: Number of top tokens to keep (0 = no filtering)
        top_p: Cumulative probability threshold (1.0 = no filtering)
        temperature: Temperature for sampling (1.0 = no scaling)
        seed: Random seed for reproducibility (None = non-deterministic)
        
    Returns:
        Index of sampled token
        
    Example:
        >>> probs = model.get_next_token_probs(context)
        >>> token = sample_token(probs, top_k=9, top_p=0.9, temperature=1.2)
    """
    # Use config if provided
    if config is not None:
        top_k = config.top_k
        top_p = config.top_p
        temperature = config.temperature
        seed = config.seed
    
    # Apply filtering
    filtered_probs = top_k_top_p_filtering(
        probs,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        sanitize=True
    )
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Sample
    try:
        token = np.random.choice(len(filtered_probs), p=filtered_probs)
    except ValueError as e:
        # Last resort fallback - should never happen with proper sanitization
        warnings.warn(f"Sampling failed ({e}), falling back to argmax")
        token = np.argmax(filtered_probs)
    
    return int(token)


# Compatibility aliases for drop-in replacement of samplings library
def top_k_sampling(
    probs: np.ndarray,
    top_k: int,
    seed: Optional[int] = None,
    return_probs: bool = False
) -> Union[int, np.ndarray]:
    """
    Drop-in replacement for samplings.top_k_sampling with added robustness.
    """
    probs = _sanitize_probabilities(probs)
    filtered = top_k_filtering(probs, top_k)
    
    if return_probs:
        return filtered
    
    if seed is not None:
        np.random.seed(seed)
    return int(np.random.choice(len(filtered), p=filtered))


def top_p_sampling(
    probs: np.ndarray,
    top_p: float,
    seed: Optional[int] = None,
    return_probs: bool = False
) -> Union[int, np.ndarray]:
    """
    Drop-in replacement for samplings.top_p_sampling with added robustness.
    """
    probs = _sanitize_probabilities(probs)
    filtered = top_p_filtering(probs, top_p)
    
    if return_probs:
        return filtered
    
    if seed is not None:
        np.random.seed(seed)
    return int(np.random.choice(len(filtered), p=filtered))


def temperature_sampling(
    probs: np.ndarray,
    temperature: float,
    weights: Union[float, np.ndarray] = 1,
    tempered_tokens: list = [],
    seed: Optional[int] = None,
    return_probs: bool = False
) -> Union[int, np.ndarray]:
    """
    Drop-in replacement for samplings.temperature_sampling with added robustness.
    
    Note: The weights and tempered_tokens parameters are preserved for API
    compatibility but the implementation is simplified. If you need the full
    weighted/selective temperature functionality, use the original samplings
    library with proper input sanitization.
    """
    probs = _sanitize_probabilities(probs)
    
    # Simplified implementation - apply temperature to all tokens
    # The original samplings library had complex weighted/selective behavior
    # that's rarely used in practice
    if temperature != 1.0:
        scaled = apply_temperature(probs, temperature)
    else:
        scaled = probs
    
    scaled = _sanitize_probabilities(scaled)
    
    if return_probs:
        return scaled
    
    if seed is not None:
        np.random.seed(seed)
    return int(np.random.choice(len(scaled), p=scaled))


if __name__ == "__main__":
    # Quick self-test
    print("Testing sampling module...")
    
    # Test basic functionality
    np.random.seed(42)
    test_probs = np.random.dirichlet(np.ones(128) * 0.01)
    
    # Test with various edge cases
    test_cases = [
        ("Normal distribution", test_probs),
        ("With NaN", np.array([0.5, np.nan, 0.3, 0.2])),
        ("With Inf", np.array([0.5, np.inf, 0.3, 0.2])),
        ("All zeros", np.zeros(128)),
        ("Negative values", np.array([0.5, -0.1, 0.3, 0.3])),
        ("Very peaked", np.array([0.999999] + [0.000001/127]*127)),
    ]
    
    for name, probs in test_cases:
        probs = np.asarray(probs, dtype=np.float64)
        if len(probs) != 128:
            probs = np.concatenate([probs, np.zeros(128 - len(probs))])
        
        try:
            token = sample_token(probs, top_k=9, top_p=0.9, temperature=1.2)
            print(f"  {name}: OK (token={token})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    
    print("\nAll tests completed!")
