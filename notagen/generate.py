"""
High-level generation API for NotaGen.

This module provides easy-to-use functions for generating music with NotaGen.
It handles model loading, configuration, and generation with sensible defaults.

Example:
    from notagen.generate import generate_music, load_notagen
    
    # Auto-detect weights and generate
    abc_notation = generate_music(period="Romantic", composer="Chopin")
    
    # Or load model once and generate multiple times
    model, patchilizer = load_notagen()
    abc1 = generate_music(model, patchilizer, period="Romantic", composer="Chopin")
    abc2 = generate_music(model, patchilizer, period="Baroque", composer="Bach")
    
    # Batch generation
    prompts = [
        {"period": "Romantic", "composer": "Chopin"},
        {"period": "Baroque", "composer": "Bach"},
    ]
    results = batch_generate(model, patchilizer, prompts)
"""

from __future__ import annotations

import time
import re
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Optional, Tuple, Generator, Callable, List, Dict, Any, 
    Union, Protocol, runtime_checkable
)
from transformers import GPT2Config

from .config import (
    find_weights, parse_weights_filename, 
    PATCH_SIZE, PATCH_LENGTH, PATCH_NUM_LAYERS, CHAR_NUM_LAYERS, HIDDEN_SIZE,
    TOP_K, TOP_P, TEMPERATURE,
)
from .device import get_device
from .patchilizer import Patchilizer
from .utils import NotaGenLMHeadModel, create_model_configs


# =============================================================================
# Type Definitions
# =============================================================================

@dataclass
class GenerationProgress:
    """Progress information for generation callbacks."""
    patches_generated: int
    max_patches: int
    elapsed_seconds: float
    max_time_seconds: float
    chars_generated: int
    current_chars: str
    
    @property
    def patch_progress(self) -> float:
        """Progress as percentage of max patches (0.0 to 1.0)."""
        return self.patches_generated / max(self.max_patches, 1)
    
    @property
    def time_progress(self) -> float:
        """Progress as percentage of max time (0.0 to 1.0)."""
        return self.elapsed_seconds / max(self.max_time_seconds, 1)
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.patches_generated == 0:
            return None
        rate = self.elapsed_seconds / self.patches_generated
        remaining = self.max_patches - self.patches_generated
        return rate * remaining


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""
    def __call__(self, progress: GenerationProgress) -> None: ...


class BaseProgressCallback(ABC):
    """Abstract base class for progress callbacks."""
    
    @abstractmethod
    def __call__(self, progress: GenerationProgress) -> None:
        """Called with progress information."""
        pass
    
    def on_start(self) -> None:
        """Called when generation starts."""
        pass
    
    def on_complete(self, total_chars: int, total_time: float) -> None:
        """Called when generation completes."""
        pass


class ConsoleProgressCallback(BaseProgressCallback):
    """Print progress to console."""
    
    def __init__(self, print_chars: bool = True, interval: int = 10):
        self.print_chars = print_chars
        self.interval = interval
        self._last_print = 0
    
    def __call__(self, progress: GenerationProgress) -> None:
        if self.print_chars:
            print(progress.current_chars, end='', flush=True)
        
        if progress.patches_generated - self._last_print >= self.interval:
            eta = progress.eta_seconds
            eta_str = f"{eta:.1f}s" if eta else "?"
            print(f"\n[Patch {progress.patches_generated}/{progress.max_patches}, "
                  f"ETA: {eta_str}]", flush=True)
            self._last_print = progress.patches_generated
    
    def on_complete(self, total_chars: int, total_time: float) -> None:
        print(f"\n✓ Generated {total_chars} chars in {total_time:.1f}s")


class TqdmProgressCallback(BaseProgressCallback):
    """Show progress bar using tqdm."""
    
    def __init__(self, desc: str = "Generating", leave: bool = True):
        self.desc = desc
        self.leave = leave
        self._pbar = None
    
    def on_start(self) -> None:
        try:
            from tqdm import tqdm
            self._pbar = tqdm(desc=self.desc, unit=" patches", leave=self.leave)
        except ImportError:
            print("tqdm not available, using console output")
            self._pbar = None
    
    def __call__(self, progress: GenerationProgress) -> None:
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix({
                'chars': progress.chars_generated,
                'eta': f"{progress.eta_seconds:.1f}s" if progress.eta_seconds else "?"
            })
    
    def on_complete(self, total_chars: int, total_time: float) -> None:
        if self._pbar is not None:
            self._pbar.close()


# =============================================================================
# Model Cache
# =============================================================================

_cached_model: Optional[NotaGenLMHeadModel] = None
_cached_patchilizer: Optional[Patchilizer] = None
_cached_device: Optional[str] = None


def load_notagen(
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    half_precision: bool = True,
) -> Tuple[NotaGenLMHeadModel, Patchilizer]:
    """
    Load NotaGen model with auto-detection of weights and config.
    
    Args:
        weights_path: Path to model weights. If None, searches common locations.
        device: Device to load model on. If None, auto-detects (CUDA > MPS > CPU).
        half_precision: Use FP16 for faster inference (default: True for GPU).
        
    Returns:
        Tuple of (model, patchilizer)
        
    Raises:
        FileNotFoundError: If no weights file found and no path provided.
    """
    global _cached_model, _cached_patchilizer, _cached_device
    
    # Auto-detect device
    if device is None:
        device = get_device(verbose=True)
    
    # Try to find weights
    if weights_path is None:
        found = find_weights()
        if not found:
            raise FileNotFoundError(
                "No weights file found. Please provide weights_path or place weights in "
                "the current directory or a 'weights/' subdirectory."
            )
        weights_path = str(found[0])
        print(f"Auto-detected weights: {weights_path}")
    
    # Parse config from filename
    parsed = parse_weights_filename(weights_path)
    if parsed:
        patch_size = parsed.get('patch_size', PATCH_SIZE)
        patch_length = parsed.get('patch_length', PATCH_LENGTH)
        patch_num_layers = parsed.get('patch_num_layers', PATCH_NUM_LAYERS)
        char_num_layers = parsed.get('char_num_layers', CHAR_NUM_LAYERS)
        hidden_size = parsed.get('hidden_size', HIDDEN_SIZE)
        print(f"Parsed config from filename: {patch_num_layers} patch layers, {char_num_layers} char layers, {hidden_size} hidden size")
    else:
        patch_size = PATCH_SIZE
        patch_length = PATCH_LENGTH
        patch_num_layers = PATCH_NUM_LAYERS
        char_num_layers = CHAR_NUM_LAYERS
        hidden_size = HIDDEN_SIZE
    
    # Create model configs
    encoder_config, decoder_config = create_model_configs(
        patch_num_layers=patch_num_layers,
        char_num_layers=char_num_layers,
        hidden_size=hidden_size,
        patch_length=patch_length,
        patch_size=patch_size,
    )
    
    # Create model
    model = NotaGenLMHeadModel(encoder_config, decoder_config, patch_size=patch_size)
    
    # Load checkpoint - use weights_only=False since our checkpoints may contain
    # optimizer state and other complex objects (PyTorch 2.6+ compatibility)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    # Handle both direct state_dict and wrapped checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    
    # Use half precision for GPU
    if half_precision and device in ('cuda', 'mps'):
        model = model.half()
    
    model.eval()
    
    # Create patchilizer
    patchilizer = Patchilizer(
        patch_size=patch_size,
        patch_length=patch_length,
        stream=True
    )
    
    # Cache for reuse
    _cached_model = model
    _cached_patchilizer = patchilizer
    _cached_device = device
    
    return model, patchilizer


def create_prompt(
    period: str,
    composer: str,
    instrumentation: Optional[str] = None,
) -> str:
    """
    Create a generation prompt from period, composer, and optional instrumentation.
    
    Args:
        period: Musical period (e.g., "Romantic", "Baroque", "Classical")
        composer: Composer name (e.g., "Chopin", "Bach", "Mozart")
        instrumentation: Optional instrumentation (e.g., "Piano Sonata", "String Quartet")
        
    Returns:
        Formatted prompt string for the model
    """
    # Clean inputs
    period = period.strip()
    composer = composer.strip()
    
    # Build prompt in ABC-like format
    prompt = f"%%period {period}\n%%composer {composer}"
    
    if instrumentation:
        prompt += f"\n%%instrumentation {instrumentation.strip()}"
    
    return prompt


def generate_music(
    model: Optional[NotaGenLMHeadModel] = None,
    patchilizer: Optional[Patchilizer] = None,
    period: str = "Romantic",
    composer: str = "Chopin",
    instrumentation: Optional[str] = None,
    prompt: Optional[str] = None,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    temperature: float = TEMPERATURE,
    max_patches: int = 128,
    max_time_seconds: int = 60,
    callback: Optional[Union[Callable[[str], None], ProgressCallback, BaseProgressCallback]] = None,
    device: Optional[str] = None,
) -> str:
    """
    Generate music in ABC notation.
    
    Args:
        model: Loaded NotaGen model (auto-loads if None)
        patchilizer: Patchilizer instance (auto-loads if None)
        period: Musical period (e.g., "Romantic", "Baroque", "Classical")
        composer: Composer name (e.g., "Chopin", "Bach", "Mozart")
        instrumentation: Optional instrumentation
        prompt: Custom prompt (overrides period/composer/instrumentation if provided)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        temperature: Temperature for sampling
        max_patches: Maximum number of patches to generate
        max_time_seconds: Maximum generation time in seconds
        callback: Callback for progress updates. Can be:
            - Simple callable(str): Called with each generated character
            - ProgressCallback: Called with GenerationProgress object
            - BaseProgressCallback: Full progress callback with start/complete hooks
        device: Device to use (auto-detects if None)
        
    Returns:
        Generated ABC notation string
    """
    global _cached_model, _cached_patchilizer, _cached_device
    
    # Load model if not provided
    if model is None or patchilizer is None:
        if _cached_model is None or _cached_patchilizer is None:
            model, patchilizer = load_notagen(device=device)
        else:
            model = _cached_model
            patchilizer = _cached_patchilizer
            device = _cached_device
    
    if device is None:
        device = next(model.parameters()).device
        if hasattr(device, 'type'):
            device = device.type
    
    # Build prompt
    if prompt is None:
        prompt = create_prompt(period, composer, instrumentation)
    
    # Initialize generation
    input_patches = patchilizer.encode(prompt)
    
    if len(input_patches) == 0:
        return prompt
    
    # Convert to tensor
    input_patches = torch.tensor([input_patches], device=device)
    
    # Track generation
    start_time = time.time()
    generated_abc = prompt
    patches_generated = 0
    total_chars = len(prompt)
    
    # Determine callback type
    is_progress_callback = isinstance(callback, (ProgressCallback, BaseProgressCallback))
    
    # Call on_start if available
    if hasattr(callback, 'on_start'):
        callback.on_start()
    
    # Generate patches
    with torch.no_grad():
        while patches_generated < max_patches:
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_time_seconds:
                break
            
            # Generate next patch
            try:
                new_patch = model.generate(
                    input_patches,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature
                )
            except Exception as e:
                print(f"Generation error: {e}")
                break
            
            # Decode patch to characters
            chars = patchilizer.decode([new_patch])
            
            # Check for end of sequence
            if not chars or chars == '<eos>':
                break
            
            # Add to output
            generated_abc += chars
            total_chars += len(chars)
            patches_generated += 1
            
            # Call callback
            if callback is not None:
                if is_progress_callback:
                    progress = GenerationProgress(
                        patches_generated=patches_generated,
                        max_patches=max_patches,
                        elapsed_seconds=elapsed,
                        max_time_seconds=max_time_seconds,
                        chars_generated=total_chars,
                        current_chars=chars,
                    )
                    callback(progress)
                else:
                    # Simple string callback
                    callback(chars)
            
            # Update input for next patch
            input_patches = patchilizer.encode(generated_abc)
            if len(input_patches) == 0:
                break
            input_patches = torch.tensor([input_patches], device=device)
            
            # Check for natural end markers
            if '\n\n' in chars or generated_abc.endswith('||'):
                break
    
    # Call on_complete if available
    total_time = time.time() - start_time
    if hasattr(callback, 'on_complete'):
        callback.on_complete(total_chars, total_time)
    
    return generated_abc


def generate_music_stream(
    model: Optional[NotaGenLMHeadModel] = None,
    patchilizer: Optional[Patchilizer] = None,
    period: str = "Romantic",
    composer: str = "Chopin",
    instrumentation: Optional[str] = None,
    prompt: Optional[str] = None,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    temperature: float = TEMPERATURE,
    max_patches: int = 128,
    max_time_seconds: int = 60,
    device: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Generate music as a stream of characters.
    
    Same parameters as generate_music(), but yields characters as they're generated.
    
    Yields:
        Individual characters or small groups of characters
    """
    result = []
    
    def collect(chars: str):
        result.append(chars)
    
    # Generate with callback to collect characters
    generated = generate_music(
        model=model,
        patchilizer=patchilizer,
        period=period,
        composer=composer,
        instrumentation=instrumentation,
        prompt=prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_patches=max_patches,
        max_time_seconds=max_time_seconds,
        callback=collect,
        device=device,
    )
    
    # Yield collected characters
    for chars in result:
        yield chars


def batch_generate(
    model: Optional[NotaGenLMHeadModel] = None,
    patchilizer: Optional[Patchilizer] = None,
    prompts: List[Dict[str, Any]] = None,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    temperature: float = TEMPERATURE,
    max_patches: int = 128,
    max_time_seconds: int = 60,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> List[str]:
    """
    Generate multiple pieces efficiently.
    
    Loads the model once and generates multiple pieces sequentially.
    
    Args:
        model: Loaded NotaGen model (auto-loads if None)
        patchilizer: Patchilizer instance (auto-loads if None)
        prompts: List of prompt dictionaries, each containing:
            - period: Musical period (required)
            - composer: Composer name (required)
            - instrumentation: Optional instrumentation
        top_k: Top-k sampling parameter (shared for all)
        top_p: Nucleus sampling parameter (shared for all)
        temperature: Temperature for sampling (shared for all)
        max_patches: Maximum patches per piece
        max_time_seconds: Maximum time per piece
        device: Device to use
        show_progress: Whether to show progress bar
        
    Returns:
        List of generated ABC notation strings
        
    Example:
        prompts = [
            {"period": "Romantic", "composer": "Chopin", "instrumentation": "Piano Sonata"},
            {"period": "Baroque", "composer": "Bach", "instrumentation": "Organ"},
            {"period": "Classical", "composer": "Mozart"},
        ]
        results = batch_generate(prompts=prompts)
    """
    if prompts is None or len(prompts) == 0:
        return []
    
    # Load model once
    if model is None or patchilizer is None:
        model, patchilizer = load_notagen(device=device)
    
    results: List[str] = []
    
    # Setup progress display
    iterator = prompts
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(prompts, desc="Batch generation", unit=" pieces")
        except ImportError:
            print(f"Generating {len(prompts)} pieces...")
    
    for i, prompt_dict in enumerate(iterator):
        period = prompt_dict.get('period', 'Romantic')
        composer = prompt_dict.get('composer', 'Unknown')
        instrumentation = prompt_dict.get('instrumentation')
        
        if not show_progress:
            print(f"[{i+1}/{len(prompts)}] {period} / {composer}", end="... ", flush=True)
        
        try:
            abc = generate_music(
                model=model,
                patchilizer=patchilizer,
                period=period,
                composer=composer,
                instrumentation=instrumentation,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_patches=max_patches,
                max_time_seconds=max_time_seconds,
                callback=None,  # No per-character callback for batch
                device=device,
            )
            results.append(abc)
            
            if not show_progress:
                print(f"✓ ({len(abc)} chars)")
                
        except Exception as e:
            print(f"Error generating {period}/{composer}: {e}")
            results.append("")  # Empty string for failed generation
    
    return results


# =============================================================================
# Convenience aliases
# =============================================================================

NotaGen = load_notagen
generate = generate_music


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main API
    'load_notagen',
    'generate_music',
    'generate_music_stream',
    'batch_generate',
    'create_prompt',
    # Progress callbacks
    'GenerationProgress',
    'ProgressCallback',
    'BaseProgressCallback',
    'ConsoleProgressCallback',
    'TqdmProgressCallback',
    # Aliases
    'NotaGen',
    'generate',
]
