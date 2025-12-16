"""
Gradio Demo Configuration for NotaGen

Imports configuration from the centralized notagen.config module.
"""

import os
import sys

# Add parent directory to path for notagen imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from centralized config
from notagen.config import (
    PATCH_SIZE, PATCH_LENGTH, PATCH_NUM_LAYERS, CHAR_NUM_LAYERS,
    HIDDEN_SIZE, PATCH_STREAM,
    TOP_K, TOP_P, TEMPERATURE,
    find_weights,
)

# Auto-detect weights or use environment variable
_found_weights = find_weights()
INFERENCE_WEIGHTS_PATH = os.environ.get(
    'NOTAGEN_WEIGHTS',
    str(_found_weights[0]) if _found_weights else ''
)