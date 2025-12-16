"""
Inference utilities for NotaGen music generation.

This module re-exports from the unified notagen.utils package.
The actual implementation is centralized for code reuse.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import everything from the unified utils package
from notagen.utils import (
    # Sampling
    top_k_sampling, top_p_sampling, temperature_sampling, sample_token,
    # Model classes
    PatchLevelDecoder, CharLevelDecoder, NotaGenLMHeadModel,
    # Patchilizer
    Patchilizer, create_patchilizer,
    # Config
    PATCH_SIZE, PATCH_LENGTH, PATCH_NUM_LAYERS, CHAR_NUM_LAYERS,
    HIDDEN_SIZE, PATCH_STREAM, PATCH_SAMPLING_BATCH_SIZE,
    # Utilities
    create_model_configs, load_model,
)

# Re-export all for backward compatibility
__all__ = [
    'top_k_sampling', 'top_p_sampling', 'temperature_sampling', 'sample_token',
    'PatchLevelDecoder', 'CharLevelDecoder', 'NotaGenLMHeadModel',
    'Patchilizer', 'create_patchilizer',
    'PATCH_SIZE', 'PATCH_LENGTH', 'PATCH_NUM_LAYERS', 'CHAR_NUM_LAYERS',
    'HIDDEN_SIZE', 'PATCH_STREAM', 'PATCH_SAMPLING_BATCH_SIZE',
    'create_model_configs', 'load_model',
]
