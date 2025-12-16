"""
Inference Configuration for NotaGen

This file imports configuration from the centralized notagen.config module.
Override settings via environment variables or by editing values below.

Environment variables:
    NOTAGEN_WEIGHTS      - Path to weights file
    NOTAGEN_NUM_SAMPLES  - Number of samples to generate
    NOTAGEN_OUTPUT_DIR   - Output directory
    NOTAGEN_VERBOSE      - Print config on import
"""

import os
import sys

# Add parent directory to path for notagen imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from centralized config (auto-detects model from weights filename)
from notagen.config import (
    PATCH_SIZE, PATCH_LENGTH, PATCH_NUM_LAYERS, CHAR_NUM_LAYERS,
    HIDDEN_SIZE, PATCH_STREAM, PATCH_SAMPLING_BATCH_SIZE,
    TOP_K, TOP_P, TEMPERATURE,
    find_weights, print_config,
)

# =============================================================================
# Inference-specific settings
# =============================================================================

NUM_SAMPLES = int(os.environ.get('NOTAGEN_NUM_SAMPLES', '1000'))

# Auto-detect weights or use environment variable
_found_weights = find_weights()
INFERENCE_WEIGHTS_PATH = os.environ.get(
    'NOTAGEN_WEIGHTS',
    str(_found_weights[0]) if _found_weights else ''
)

# Output folders
_weights_name = os.path.splitext(os.path.basename(INFERENCE_WEIGHTS_PATH))[0] if INFERENCE_WEIGHTS_PATH else 'unknown'
ORIGINAL_OUTPUT_FOLDER = os.path.join('../output/original', f'{_weights_name}_k_{TOP_K}_p_{TOP_P}_temp_{TEMPERATURE}')
INTERLEAVED_OUTPUT_FOLDER = os.path.join('../output/interleaved', f'{_weights_name}_k_{TOP_K}_p_{TOP_P}_temp_{TEMPERATURE}')

# Print config on import if verbose
if os.environ.get('NOTAGEN_VERBOSE'):
    print(f"Weights: {INFERENCE_WEIGHTS_PATH}")
    print_config()