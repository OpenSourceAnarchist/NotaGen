"""
Unified utilities for NotaGen.

This module provides all utility classes and functions needed by the 
inference, training, and gradio components. Individual folders can 
import from here instead of maintaining separate implementations.
"""
import sys
import os
import re
import torch
import random
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config, PreTrainedModel
from typing import Optional, List, Tuple

# Import sampling functions
from .sampling import top_k_sampling, top_p_sampling, temperature_sampling, sample_token

# Import Patchilizer base
from .patchilizer import Patchilizer

# Import configs
from .config import (
    ModelConfig, SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG,
    PATCH_SIZE, PATCH_LENGTH, PATCH_NUM_LAYERS, CHAR_NUM_LAYERS, HIDDEN_SIZE,
    PATCH_STREAM, PATCH_SAMPLING_BATCH_SIZE,
)

# Re-export useful items
__all__ = [
    # Sampling
    'top_k_sampling', 'top_p_sampling', 'temperature_sampling', 'sample_token',
    # Model
    'PatchLevelDecoder', 'CharLevelDecoder', 'NotaGenLMHeadModel',
    # Patchilizer
    'Patchilizer', 'create_patchilizer',
    # Config
    'PATCH_SIZE', 'PATCH_LENGTH', 'PATCH_NUM_LAYERS', 'CHAR_NUM_LAYERS',
    'HIDDEN_SIZE', 'PATCH_STREAM', 'PATCH_SAMPLING_BATCH_SIZE',
    # Utils
    'create_model_configs', 'load_model',
]


def create_patchilizer(
    patch_size: int = None,
    patch_length: int = None,
    stream: bool = None
) -> Patchilizer:
    """
    Create a Patchilizer with the given or default configuration.
    
    Args:
        patch_size: Size of each patch (default: from config)
        patch_length: Maximum sequence length in patches (default: from config)
        stream: Whether to use stream mode (default: from config)
        
    Returns:
        Configured Patchilizer instance
    """
    return Patchilizer(
        patch_size=patch_size or PATCH_SIZE,
        patch_length=patch_length or PATCH_LENGTH,
        stream=stream if stream is not None else PATCH_STREAM
    )


class PatchLevelDecoder(PreTrainedModel):
    """
    Patch-level decoder for generating patch embeddings.
    
    Uses a given patch_size for encoding patches into the transformer.
    """
    
    def __init__(self, config: GPT2Config, patch_size: int = None):
        super().__init__(config)
        self._patch_size = patch_size or PATCH_SIZE
        
        # Embedding layer: one-hot patches -> hidden size
        self.patch_embedding = torch.nn.Linear(self._patch_size * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        
        # GPT-2 backbone
        self.base = GPT2Model(config)
    
    def forward(
        self,
        patches: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass of the patch-level decoder.
        
        Args:
            patches: Tensor of shape [batch, seq_len, patch_size] with character IDs
            masks: Optional attention mask of shape [batch, seq_len]
            
        Returns:
            Dictionary with 'last_hidden_state' key containing patch embeddings
        """
        patches_onehot = torch.nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches_flat = patches_onehot.reshape(len(patches), -1, self._patch_size * 128)
        patch_embeds = self.patch_embedding(patches_flat.to(self.device))
        
        if masks is None:
            return self.base(inputs_embeds=patch_embeds)
        else:
            return self.base(inputs_embeds=patch_embeds, attention_mask=masks)


class CharLevelDecoder(PreTrainedModel):
    """
    Character-level decoder for generating characters within a patch.
    """
    
    def __init__(self, config: GPT2Config, patch_sampling_batch_size: int = None):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self._patch_sampling_batch_size = patch_sampling_batch_size or PATCH_SAMPLING_BATCH_SIZE
        self.base = GPT2LMHeadModel(config)
    
    def forward(
        self,
        encoded_patches: torch.Tensor,
        target_patches: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for training."""
        # Prepend BOS token
        target_patches = torch.cat(
            (torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id, target_patches),
            dim=1
        )
        
        # Create labels (mask padding tokens)
        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)
        
        # Optional patch sampling for memory efficiency during training
        if self._patch_sampling_batch_size > 0 and self._patch_sampling_batch_size < target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:self._patch_sampling_batch_size])
            target_patches = target_patches[selected_indices, :]
            target_masks = target_masks[selected_indices, :]
            encoded_patches = encoded_patches[selected_indices, :]
        
        # Get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)
        
        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=target_masks,
            labels=labels
        )
    
    def generate(
        self,
        encoded_patch: torch.Tensor,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """Generate next token probabilities."""
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)
        
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)
        tokens = torch.cat((encoded_patch, tokens[:, 1:, :]), dim=1)
        
        outputs = self.base(inputs_embeds=tokens)
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)
        
        return probs


class NotaGenLMHeadModel(PreTrainedModel):
    """
    NotaGen: Hierarchical Language Model for Symbolic Music Generation.
    """
    
    def __init__(
        self,
        encoder_config: GPT2Config,
        decoder_config: GPT2Config,
        patch_size: int = None,
        patch_sampling_batch_size: int = None
    ):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self._patch_size = patch_size or PATCH_SIZE
        
        self.patch_level_decoder = PatchLevelDecoder(encoder_config, self._patch_size)
        self.char_level_decoder = CharLevelDecoder(decoder_config, patch_sampling_batch_size)
    
    def forward(
        self,
        patches: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for training."""
        patches = patches.reshape(len(patches), -1, self._patch_size)
        encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]
        
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0
        
        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]
        
        return self.char_level_decoder(encoded_patches, patches)
    
    def generate(
        self,
        patches: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0
    ) -> List[int]:
        """Generate the next patch."""
        if patches.shape[-1] % self._patch_size != 0:
            tokens = patches[:, :, -(patches.shape[-1] % self._patch_size):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:, :, :-(patches.shape[-1] % self._patch_size)]
        else:
            tokens = torch.tensor([self.bos_token_id], device=self.device)
        
        patches = patches.reshape(len(patches), -1, self._patch_size)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        generated_patch = []
        
        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature)
            generated_patch.append(token)
            
            if len(tokens) >= self._patch_size:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch


def create_model_configs(
    patch_num_layers: int = None,
    char_num_layers: int = None, 
    hidden_size: int = None,
    patch_length: int = None,
    patch_size: int = None
) -> Tuple[GPT2Config, GPT2Config]:
    """
    Create encoder and decoder GPT2 configs.
    
    Args:
        patch_num_layers: Number of transformer layers for patch-level decoder
        char_num_layers: Number of transformer layers for char-level decoder
        hidden_size: Hidden dimension size
        patch_length: Maximum sequence length in patches
        patch_size: Size of each patch
        
    Returns:
        Tuple of (encoder_config, decoder_config)
    """
    _patch_num_layers = patch_num_layers or PATCH_NUM_LAYERS
    _char_num_layers = char_num_layers or CHAR_NUM_LAYERS
    _hidden_size = hidden_size or HIDDEN_SIZE
    _patch_length = patch_length or PATCH_LENGTH
    _patch_size = patch_size or PATCH_SIZE
    
    num_heads = _hidden_size // 64
    
    encoder_config = GPT2Config(
        num_hidden_layers=_patch_num_layers,
        max_length=_patch_length,
        max_position_embeddings=_patch_length,
        hidden_size=_hidden_size,
        n_embd=_hidden_size,
        num_attention_heads=num_heads,
        vocab_size=1,
    )
    
    decoder_config = GPT2Config(
        num_hidden_layers=_char_num_layers,
        max_length=_patch_size + 1,
        max_position_embeddings=_patch_size + 1,
        hidden_size=_hidden_size,
        num_attention_heads=num_heads,
        vocab_size=128,
    )
    
    return encoder_config, decoder_config


def load_model(
    weights_path: str,
    patch_num_layers: int = None,
    char_num_layers: int = None,
    hidden_size: int = None,
    patch_length: int = None,
    patch_size: int = None,
    patch_sampling_batch_size: int = None,
    device: str = None,
    dtype: torch.dtype = None
) -> NotaGenLMHeadModel:
    """
    Load a NotaGen model from weights.
    
    Args:
        weights_path: Path to the model weights file
        patch_num_layers: Number of patch-level layers (auto-detected from filename if None)
        char_num_layers: Number of char-level layers (auto-detected from filename if None)
        hidden_size: Hidden dimension (auto-detected from filename if None)
        patch_length: Max patch sequence length (auto-detected from filename if None)
        patch_size: Patch size (auto-detected from filename if None)
        patch_sampling_batch_size: Batch size for patch sampling (0 = disabled)
        device: Device to load model on ('cuda', 'mps', 'cpu', or None for auto)
        dtype: Model dtype (torch.float16, torch.float32, etc.)
        
    Returns:
        Loaded NotaGenLMHeadModel
    """
    from .config import parse_weights_filename
    from .device import get_device
    
    # Auto-detect config from filename if not provided
    parsed = parse_weights_filename(weights_path)
    if parsed:
        patch_num_layers = patch_num_layers or parsed.get('patch_num_layers')
        char_num_layers = char_num_layers or parsed.get('char_num_layers')
        hidden_size = hidden_size or parsed.get('hidden_size')
        patch_length = patch_length or parsed.get('patch_length')
        patch_size = patch_size or parsed.get('patch_size')
    
    # Create configs
    encoder_config, decoder_config = create_model_configs(
        patch_num_layers=patch_num_layers,
        char_num_layers=char_num_layers,
        hidden_size=hidden_size,
        patch_length=patch_length,
        patch_size=patch_size,
    )
    
    # Create model
    model = NotaGenLMHeadModel(
        encoder_config,
        decoder_config,
        patch_size=patch_size or PATCH_SIZE,
        patch_sampling_batch_size=patch_sampling_batch_size or 0
    )
    
    # Load weights
    _device = device or get_device()
    # Load checkpoint - use weights_only=False since our checkpoints may contain
    # optimizer state and other complex objects (PyTorch 2.6+ compatibility)
    checkpoint = torch.load(weights_path, map_location=_device, weights_only=False)
    # Handle both direct state_dict and wrapped checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set dtype
    model = model.to(_device)
    if dtype:
        model = model.to(dtype)
    elif _device in ('cuda', 'mps'):
        model = model.half()  # Use fp16 for GPU
    
    model.eval()
    return model
