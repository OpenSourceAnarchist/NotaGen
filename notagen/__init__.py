"""
NotaGen: Symbolic Music Generation with Large Language Model Training Paradigms

This package provides shared utilities for the NotaGen music generation model,
including numerically stable sampling, configuration management, device detection,
and the core model architecture.

Quick Start:
    from notagen import load_model, generate_music
    
    model = load_model()  # Auto-detects weights and config
    abc = generate_music(model, period="Romantic", composer="Chopin", style="Piano Sonata")
"""

# Import sampling functions (these have no external dependencies)
from .sampling import (
    top_k_sampling,
    top_p_sampling,
    temperature_sampling,
    top_k_filtering,
    top_p_filtering,
    apply_temperature,
    sample_token,
)

# Import tokenizer
from .patchilizer import Patchilizer

# Import config classes and utilities
from .config import (
    ModelConfig, 
    SamplingConfig, 
    InferenceConfig, 
    TrainingConfig,
    # Auto-detection
    find_weights,
    parse_weights_filename,
    auto_detect_config,
    create_config_from_weights,
    get_model_config,
    print_config,
    # Presets
    SMALL_CONFIG,
    MEDIUM_CONFIG,
    LARGE_CONFIG,
    NOTAGENX_CONFIG,
    # Legacy compatibility constants
    PATCH_SIZE,
    PATCH_LENGTH,
    PATCH_NUM_LAYERS,
    CHAR_NUM_LAYERS,
    HIDDEN_SIZE,
    PATCH_STREAM,
    TOP_K,
    TOP_P,
    TEMPERATURE,
)

# Try to import device utilities (requires torch)
try:
    from .device import (
        get_device,
        get_autocast_device_type,
        is_gpu_available,
        get_device_info,
        set_seed,
        empty_cache,
    )
    _DEVICE_AVAILABLE = True
except ImportError:
    _DEVICE_AVAILABLE = False
    get_device = None
    get_autocast_device_type = None
    is_gpu_available = None
    get_device_info = None
    set_seed = None
    empty_cache = None

# Try to import model components, but don't fail if torch isn't available
try:
    from .model import PatchLevelDecoder, CharLevelDecoder, NotaGenLMHeadModel
    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False
    PatchLevelDecoder = None
    CharLevelDecoder = None
    NotaGenLMHeadModel = None

# Try to import unified utilities (requires torch)
try:
    from .utils import (
        create_patchilizer,
        create_model_configs,
        load_model,
    )
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False
    create_patchilizer = None
    create_model_configs = None
    load_model = None

# Try to import high-level generation API
try:
    from .generate import (
        load_notagen,
        generate_music,
        generate_music_stream,
        batch_generate,
        create_prompt,
        # Progress callbacks
        GenerationProgress,
        ProgressCallback,
        BaseProgressCallback,
        ConsoleProgressCallback,
        TqdmProgressCallback,
    )
    _GENERATE_AVAILABLE = True
except ImportError:
    _GENERATE_AVAILABLE = False
    load_notagen = None
    generate_music = None
    generate_music_stream = None
    batch_generate = None
    create_prompt = None
    GenerationProgress = None
    ProgressCallback = None
    BaseProgressCallback = None
    ConsoleProgressCallback = None
    TqdmProgressCallback = None

# Try to import export utilities
try:
    from .export import (
        abc_to_musicxml,
        abc_to_midi,
        save_output,
        generate_filename,
    )
    _EXPORT_AVAILABLE = True
except ImportError:
    _EXPORT_AVAILABLE = False
    abc_to_musicxml = None
    abc_to_midi = None
    save_output = None
    generate_filename = None

__version__ = "1.0.0"
__all__ = [
    # Version
    "__version__",
    # Sampling functions
    "top_k_sampling",
    "top_p_sampling", 
    "temperature_sampling",
    "top_k_filtering",
    "top_p_filtering",
    "apply_temperature",
    "sample_token",
    # Patchilizer
    "Patchilizer",
    # Config classes
    "ModelConfig",
    "SamplingConfig",
    "InferenceConfig",
    "TrainingConfig",
    # Config utilities
    "find_weights",
    "parse_weights_filename",
    "auto_detect_config",
    "create_config_from_weights",
    "get_model_config",
    "print_config",
    # Config presets
    "SMALL_CONFIG",
    "MEDIUM_CONFIG", 
    "LARGE_CONFIG",
    "NOTAGENX_CONFIG",
    # Legacy constants
    "PATCH_SIZE",
    "PATCH_LENGTH",
    "PATCH_NUM_LAYERS",
    "CHAR_NUM_LAYERS",
    "HIDDEN_SIZE",
    "PATCH_STREAM",
    "TOP_K",
    "TOP_P",
    "TEMPERATURE",
    # Device utilities (may be None if torch not available)
    "get_device",
    "get_autocast_device_type",
    "is_gpu_available",
    "get_device_info",
    "set_seed",
    "empty_cache",
    # Model components (may be None if torch not available)
    "PatchLevelDecoder",
    "CharLevelDecoder",
    "NotaGenLMHeadModel",
    # Unified utilities
    "create_patchilizer",
    "create_model_configs",
    "load_model",
    # High-level generation API
    "load_notagen",
    "generate_music",
    "generate_music_stream",
    "batch_generate",
    "create_prompt",
    # Progress callbacks
    "GenerationProgress",
    "ProgressCallback",
    "BaseProgressCallback",
    "ConsoleProgressCallback",
    "TqdmProgressCallback",
    # Export utilities
    "abc_to_musicxml",
    "abc_to_midi",
    "save_output",
    "generate_filename",
]
