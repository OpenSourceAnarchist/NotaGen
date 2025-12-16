"""
Reinforcement Learning (DPO/DPOP) configuration for NotaGen.
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
DATA_INDEX_PATH = ''  # Path to preference data

# =============================================================================
# Model Configuration - Should match pretrained model
# =============================================================================
# For RL, use LARGE_CONFIG to match NotaGen-X pretrained weights:
_MODEL = LARGE_CONFIG

PATCH_STREAM = True
PATCH_SIZE = _MODEL.patch_size
PATCH_LENGTH = _MODEL.patch_length
PATCH_NUM_LAYERS = _MODEL.patch_num_layers
CHAR_NUM_LAYERS = _MODEL.char_num_layers
HIDDEN_SIZE = _MODEL.hidden_size

# =============================================================================
# RL Training Configuration (DPO/DPOP)
# =============================================================================
BETA = 0.1       # Beta in DPO's objective function
LAMBDA = 10      # Lambda in DPOP's objective function
LEARNING_RATE = 1e-6
OPTIMIZATION_STEPS = 10000

# Wandb logging
WANDB_LOGGING = False
WANDB_KEY = '<your_wandb_key>'

# Path to pretrained weights to start RL from
PRETRAINED_PATH = ''

# =============================================================================
# Auto-generated paths
# =============================================================================
EXP_TAG = ''
NAME = (
    f"{EXP_TAG}_beta_{BETA}_lambda_{LAMBDA}"
    f"_p_size_{PATCH_SIZE}_p_length_{PATCH_LENGTH}"
    f"_p_layers_{PATCH_NUM_LAYERS}_c_layers_{CHAR_NUM_LAYERS}"
    f"_h_size_{HIDDEN_SIZE}_lr_{LEARNING_RATE}"
)

WEIGHTS_PATH = f"weights_notagen_{NAME}.pth"
WANDB_NAME = NAME
