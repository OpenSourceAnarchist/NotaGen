"""
Pretrain configuration for NotaGen.
Imports core model settings from notagen.config.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model configuration from central config
from notagen.config import (
    SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG,
    get_model_config, find_weights, parse_weights_filename
)

# =============================================================================
# Data Configuration
# =============================================================================
DATA_TRAIN_INDEX_PATH = ""  # Path to training data index
DATA_EVAL_INDEX_PATH = ""   # Path to evaluation data index

# =============================================================================
# Model Configuration - Choose a preset or customize
# =============================================================================
# Available presets: SMALL_CONFIG (768), MEDIUM_CONFIG (1024), LARGE_CONFIG (1280)
# For pretraining, SMALL_CONFIG is recommended to start:
_MODEL = SMALL_CONFIG

PATCH_STREAM = True
PATCH_SIZE = _MODEL.patch_size
PATCH_LENGTH = _MODEL.patch_length
PATCH_NUM_LAYERS = _MODEL.patch_num_layers
CHAR_NUM_LAYERS = _MODEL.char_num_layers
HIDDEN_SIZE = _MODEL.hidden_size

# =============================================================================
# Training Configuration
# =============================================================================
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 128
ACCUMULATION_STEPS = 1
PATCH_SAMPLING_BATCH_SIZE = 0  # 0 = use full patches
LOAD_FROM_CHECKPOINT = False

# Wandb logging
WANDB_LOGGING = False
WANDB_KEY = '<your_wandb_key>'

# =============================================================================
# Auto-generated paths
# =============================================================================
EXP_TAG = 'pretrain'
NAME = (
    f"{EXP_TAG}_p_size_{PATCH_SIZE}_p_length_{PATCH_LENGTH}"
    f"_p_layers_{PATCH_NUM_LAYERS}_c_layers_{CHAR_NUM_LAYERS}"
    f"_h_size_{HIDDEN_SIZE}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}"
)

WEIGHTS_PATH = f"weights_notagen_{NAME}.pth"
LOGS_PATH = f"logs_notagen_{NAME}.txt"
WANDB_NAME = NAME

                                                                                   
