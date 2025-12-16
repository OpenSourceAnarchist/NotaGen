"""
Configuration Management for NotaGen

Provides dataclass-based configuration for all NotaGen components,
enabling easy configuration via code, command-line, or config files.

This is the SINGLE SOURCE OF TRUTH for all configuration in NotaGen.
Individual folders (inference/, gradio/, etc.) import from here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
import json
import os
import re
import glob


# =============================================================================
# Weight File Auto-Detection
# =============================================================================

def parse_weights_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse model configuration from weights filename.
    
    Expected format: weights_*_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_*.pth
    
    Returns dict with: patch_size, patch_length, patch_num_layers, char_num_layers, hidden_size
    """
    patterns = {
        'patch_size': r'p_size_(\d+)',
        'patch_length': r'p_length_(\d+)',
        'patch_num_layers': r'p_layers_(\d+)',
        'char_num_layers': r'c_layers_(\d+)',
        'hidden_size': r'h_size_(\d+)',
    }
    
    config = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            config[key] = int(match.group(1))
    
    # Only return if we found at least the key parameters
    if len(config) >= 3:
        return config
    return None


def find_weights(
    search_paths: Optional[List[str]] = None,
    pattern: str = "*.pth"
) -> List[Path]:
    """
    Find weight files in common locations.
    
    Search order:
    1. Custom search_paths if provided
    2. Current directory
    3. Parent directory
    4. Common subdirectories (weights/, checkpoints/, models/)
    5. NOTAGEN_WEIGHTS_DIR environment variable
    
    Returns list of found weight file paths, sorted by modification time (newest first).
    """
    if search_paths is None:
        search_paths = []
    
    # Build search locations
    locations = list(search_paths)
    
    # Current and parent directories
    locations.extend(['.', '..', '../..'])
    
    # Common subdirectories
    for subdir in ['weights', 'checkpoints', 'models', 'pretrained']:
        locations.extend([subdir, f'../{subdir}', f'../../{subdir}'])
    
    # Environment variable
    env_dir = os.environ.get('NOTAGEN_WEIGHTS_DIR')
    if env_dir:
        locations.append(env_dir)
    
    # Find all matching files
    found = []
    seen = set()
    
    for loc in locations:
        try:
            for filepath in glob.glob(os.path.join(loc, pattern)):
                abs_path = Path(filepath).resolve()
                if abs_path not in seen and abs_path.exists():
                    seen.add(abs_path)
                    found.append(abs_path)
        except Exception:
            continue
    
    # Sort by modification time (newest first)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return found


def auto_detect_config(weights_path: Optional[str] = None) -> tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """
    Auto-detect weights file and parse its configuration.
    
    Args:
        weights_path: Explicit path to weights file. If None, searches common locations.
        
    Returns:
        Tuple of (weights_path, config_dict) or (None, None) if not found.
    """
    if weights_path:
        path = Path(weights_path)
        if path.exists():
            config = parse_weights_filename(path.name)
            return path, config
        return None, None
    
    # Search for weights
    found = find_weights()
    if found:
        path = found[0]
        config = parse_weights_filename(path.name)
        return path, config
    
    return None, None


@dataclass
class ModelConfig:
    """Configuration for the NotaGen model architecture."""
    patch_size: int = 16
    patch_length: int = 1024
    patch_num_layers: int = 20
    char_num_layers: int = 6
    hidden_size: int = 1280
    patch_stream: bool = True
    patch_sampling_batch_size: int = 0
    
    @property
    def num_attention_heads(self) -> int:
        """Compute number of attention heads from hidden size."""
        return self.hidden_size // 64
    
    def validate(self):
        """Validate configuration parameters."""
        if self.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size must be divisible by 64, got {self.hidden_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.patch_length <= 0:
            raise ValueError(f"patch_length must be positive, got {self.patch_length}")


@dataclass
class SamplingConfig:
    """Configuration for sampling during generation."""
    top_k: int = 9
    top_p: float = 0.9
    temperature: float = 1.2
    seed: Optional[int] = None
    
    def validate(self):
        """Validate sampling parameters."""
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


@dataclass
class InferenceConfig:
    """Configuration for inference/generation."""
    weights_path: str = ""
    num_samples: int = 1
    max_length: int = 102400
    max_time_seconds: int = 600
    output_dir: str = "output"
    device: str = "cuda"
    use_fp16: bool = True
    
    # Sampling configuration
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # Model configuration  
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def validate(self):
        """Validate all configuration parameters."""
        if not self.weights_path:
            raise ValueError("weights_path must be specified")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        self.sampling.validate()
        self.model.validate()
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "InferenceConfig":
        """Load configuration from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle nested configs
        if 'sampling' in data and isinstance(data['sampling'], dict):
            data['sampling'] = SamplingConfig(**data['sampling'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        
        return cls(**data)
    
    def to_json(self, path: Union[str, Path]):
        """Save configuration to a JSON file."""
        data = {
            'weights_path': self.weights_path,
            'num_samples': self.num_samples,
            'max_length': self.max_length,
            'max_time_seconds': self.max_time_seconds,
            'output_dir': self.output_dir,
            'device': self.device,
            'use_fp16': self.use_fp16,
            'sampling': {
                'top_k': self.sampling.top_k,
                'top_p': self.sampling.top_p,
                'temperature': self.sampling.temperature,
                'seed': self.sampling.seed,
            },
            'model': {
                'patch_size': self.model.patch_size,
                'patch_length': self.model.patch_length,
                'patch_num_layers': self.model.patch_num_layers,
                'char_num_layers': self.model.char_num_layers,
                'hidden_size': self.model.hidden_size,
                'patch_stream': self.model.patch_stream,
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    data_train_index_path: str = ""
    data_eval_index_path: str = ""
    pretrained_path: str = ""
    exp_tag: str = "notagen"
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 1
    num_epochs: int = 10
    warmup_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def validate(self):
        """Validate training configuration."""
        if not self.data_train_index_path:
            raise ValueError("data_train_index_path must be specified")
        self.model.validate()


# Convenient type alias
NotaGenConfig = InferenceConfig


# Default configurations for different model sizes
SMALL_CONFIG = ModelConfig(
    patch_num_layers=12,
    char_num_layers=3,
    hidden_size=768,
    patch_length=2048,
)

MEDIUM_CONFIG = ModelConfig(
    patch_num_layers=16,
    char_num_layers=3,
    hidden_size=1024,
    patch_length=2048,
)

LARGE_CONFIG = ModelConfig(
    patch_num_layers=20,
    char_num_layers=6,
    hidden_size=1280,
    patch_length=1024,
)

# NotaGen-X uses the same as LARGE
NOTAGENX_CONFIG = LARGE_CONFIG


def get_model_config(size: str = "large") -> ModelConfig:
    """Get a predefined model configuration by size name."""
    configs = {
        "small": SMALL_CONFIG,
        "medium": MEDIUM_CONFIG,
        "large": LARGE_CONFIG,
        "notagenx": NOTAGENX_CONFIG,
    }
    if size.lower() not in configs:
        raise ValueError(f"Unknown model size '{size}'. Available: {list(configs.keys())}")
    return configs[size.lower()]


def create_config_from_weights(weights_path: Optional[str] = None) -> tuple[str, ModelConfig]:
    """
    Create configuration by auto-detecting weights file and parsing its filename.
    
    Args:
        weights_path: Explicit path to weights. If None, searches common locations.
        
    Returns:
        Tuple of (weights_path_str, ModelConfig)
        
    Raises:
        FileNotFoundError: If no weights file found
        ValueError: If weights filename cannot be parsed
    """
    path, parsed = auto_detect_config(weights_path)
    
    if path is None:
        raise FileNotFoundError(
            "No weights file found. Please either:\n"
            "  1. Provide weights_path argument\n"
            "  2. Set NOTAGEN_WEIGHTS environment variable\n"
            "  3. Place weights file in current directory"
        )
    
    if parsed:
        config = ModelConfig(
            patch_size=parsed.get('patch_size', 16),
            patch_length=parsed.get('patch_length', 1024),
            patch_num_layers=parsed.get('patch_num_layers', 20),
            char_num_layers=parsed.get('char_num_layers', 6),
            hidden_size=parsed.get('hidden_size', 1280),
        )
    else:
        # Couldn't parse filename, use large config as default
        print(f"Warning: Could not parse config from filename '{path.name}', using large model config")
        config = LARGE_CONFIG
    
    return str(path), config


# =============================================================================
# Legacy Compatibility - Individual config variables
# These are exported so individual folders can do: from notagen.config import PATCH_SIZE
# =============================================================================

def _get_active_config() -> ModelConfig:
    """Get the currently active model configuration."""
    weights_path = os.environ.get('NOTAGEN_WEIGHTS')
    if weights_path:
        _, config = auto_detect_config(weights_path)
        if config:
            return ModelConfig(**config)
    
    # Try to find weights and parse config
    found = find_weights()
    if found:
        config = parse_weights_filename(found[0].name)
        if config:
            return ModelConfig(**config)
    
    # Default to medium config (most common for users)
    return MEDIUM_CONFIG


# Active configuration - auto-detected or from environment
_ACTIVE_CONFIG = _get_active_config()

# Export individual variables for backward compatibility
PATCH_SIZE = _ACTIVE_CONFIG.patch_size
PATCH_LENGTH = _ACTIVE_CONFIG.patch_length  
PATCH_NUM_LAYERS = _ACTIVE_CONFIG.patch_num_layers
CHAR_NUM_LAYERS = _ACTIVE_CONFIG.char_num_layers
HIDDEN_SIZE = _ACTIVE_CONFIG.hidden_size
PATCH_STREAM = _ACTIVE_CONFIG.patch_stream
PATCH_SAMPLING_BATCH_SIZE = _ACTIVE_CONFIG.patch_sampling_batch_size

# Sampling defaults
TOP_K = 9
TOP_P = 0.9
TEMPERATURE = 1.2

# Training defaults
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
ACCUMULATION_STEPS = 1
NUM_EPOCHS = 10
WARMUP_STEPS = 100

# Logging
WANDB_LOGGING = False
WANDB_KEY = ""
WANDB_NAME = "notagen-run"


def print_config():
    """Print the current active configuration."""
    print("=" * 50)
    print("NotaGen Active Configuration")
    print("=" * 50)
    print(f"  PATCH_SIZE:       {PATCH_SIZE}")
    print(f"  PATCH_LENGTH:     {PATCH_LENGTH}")
    print(f"  PATCH_NUM_LAYERS: {PATCH_NUM_LAYERS}")
    print(f"  CHAR_NUM_LAYERS:  {CHAR_NUM_LAYERS}")
    print(f"  HIDDEN_SIZE:      {HIDDEN_SIZE}")
    print(f"  PATCH_STREAM:     {PATCH_STREAM}")
    print("-" * 50)
    print(f"  TOP_K:            {TOP_K}")
    print(f"  TOP_P:            {TOP_P}")
    print(f"  TEMPERATURE:      {TEMPERATURE}")
    print("=" * 50)
