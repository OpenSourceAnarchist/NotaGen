"""
Gradio Demo Configuration for NotaGen

Imports configuration from the centralized notagen.config module.
"""

import os
import sys

# Get script directory for reliable path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add parent directory to path for notagen imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from centralized config
from notagen.config import (
    PATCH_SIZE, PATCH_LENGTH, PATCH_NUM_LAYERS, CHAR_NUM_LAYERS,
    HIDDEN_SIZE, PATCH_STREAM,
    TOP_K, TOP_P, TEMPERATURE,
    find_weights,
)

# Search paths for weights - includes Colab common locations
_search_paths = [
    PROJECT_ROOT,                          # Project root
    os.getcwd(),                           # Current working directory
    '/content',                            # Colab default
    '/content/NotaGen',                    # Colab with cloned repo
    os.path.expanduser('~'),               # Home directory
]

# Auto-detect weights or use environment variable
# Check environment variable first (takes priority)
INFERENCE_WEIGHTS_PATH = os.environ.get('NOTAGEN_WEIGHTS', '')

# If not set via env, try auto-detection
if not INFERENCE_WEIGHTS_PATH:
    _found_weights = find_weights(search_paths=_search_paths)
    INFERENCE_WEIGHTS_PATH = str(_found_weights[0]) if _found_weights else ''

# Validate weights path - only warn if truly not found
if not INFERENCE_WEIGHTS_PATH or not os.path.exists(INFERENCE_WEIGHTS_PATH):
    import warnings
    warnings.warn(
        f"""
NotaGen weights not found!

Searched in: {', '.join(_search_paths[:3])}...

Please either:
1. Download weights to the project directory
2. Set NOTAGEN_WEIGHTS environment variable to the weights path

Example:
  wget https://huggingface.co/ElectricAlexis/NotaGen/resolve/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth
  export NOTAGEN_WEIGHTS=/path/to/weights.pth
""",
        RuntimeWarning
    )