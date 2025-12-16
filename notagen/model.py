"""
NotaGen Model Architecture

Hierarchical Language Model for Symbolic Music Generation:
- Patch-level decoder: Generates patch embeddings autoregressively
- Character-level decoder: Generates characters within each patch

Based on the bGPT architecture with GPT-2 as the backbone.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config, PreTrainedModel
from typing import Optional, Tuple, List

from .config import ModelConfig
from .sampling import sample_token


class PatchLevelDecoder(PreTrainedModel):
    """
    Patch-level decoder for generating patch embeddings.
    
    Takes sequences of patches (each patch is a sequence of character IDs)
    and produces contextualized embeddings for each patch position.
    """
    
    def __init__(self, config: GPT2Config, patch_size: int = 16):
        super().__init__(config)
        self.patch_size = patch_size
        
        # Embedding layer: one-hot patches -> hidden size
        self.patch_embedding = nn.Linear(patch_size * 128, config.n_embd)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        
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
        # One-hot encode patches: [batch, seq, patch_size] -> [batch, seq, patch_size * 128]
        patches_onehot = nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches_flat = patches_onehot.reshape(patches.shape[0], -1, self.patch_size * 128)
        
        # Project to embedding space
        patch_embeds = self.patch_embedding(patches_flat.to(self.device))
        
        # Run through transformer
        if masks is None:
            return self.base(inputs_embeds=patch_embeds)
        else:
            return self.base(inputs_embeds=patch_embeds, attention_mask=masks)


class CharLevelDecoder(PreTrainedModel):
    """
    Character-level decoder for generating characters within a patch.
    
    Takes a patch embedding and generates characters autoregressively
    until the patch is complete or EOS is reached.
    """
    
    SPECIAL_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.base = GPT2LMHeadModel(config)
    
    def forward(
        self,
        encoded_patches: torch.Tensor,
        target_patches: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            encoded_patches: Patch embeddings from patch-level decoder [num_patches, hidden_size]
            target_patches: Target character IDs [num_patches, patch_size]
            
        Returns:
            Language modeling loss
        """
        # Prepend BOS token
        bos_tokens = torch.ones_like(target_patches[:, 0:1]) * self.BOS_TOKEN_ID
        target_patches = torch.cat([bos_tokens, target_patches], dim=1)
        
        # Create labels (mask padding tokens)
        padding_mask = target_patches == self.SPECIAL_TOKEN_ID
        labels = target_patches.clone()
        labels[padding_mask] = -100
        
        # Create attention mask
        attention_mask = torch.ones_like(labels)
        attention_mask[labels == -100] = 0
        
        # Get input embeddings
        inputs_embeds = nn.functional.embedding(
            target_patches,
            self.base.transformer.wte.weight
        )
        
        # Replace first position with encoded patch
        inputs_embeds = torch.cat([
            encoded_patches.unsqueeze(1),
            inputs_embeds[:, 1:, :]
        ], dim=1)
        
        # Forward pass
        output = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return output
    
    def generate(
        self,
        encoded_patch: torch.Tensor,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate next token probabilities.
        
        Args:
            encoded_patch: Single patch embedding [hidden_size]
            tokens: Already generated tokens [seq_len]
            
        Returns:
            Probability distribution over next token [128]
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)
        
        # Get input embeddings
        token_embeds = nn.functional.embedding(
            tokens,
            self.base.transformer.wte.weight
        )
        
        # Replace first position with encoded patch
        inputs_embeds = torch.cat([
            encoded_patch,
            token_embeds[:, 1:, :]
        ], dim=1)
        
        # Forward pass
        outputs = self.base(inputs_embeds=inputs_embeds)
        
        # Get probabilities for next token
        logits = outputs.logits.squeeze(0)[-1]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        return probs


class NotaGenLMHeadModel(PreTrainedModel):
    """
    NotaGen: Hierarchical Language Model for Symbolic Music Generation.
    
    Combines patch-level and character-level decoders for efficient
    generation of long musical sequences.
    """
    
    SPECIAL_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    
    def __init__(
        self,
        encoder_config: GPT2Config,
        decoder_config: GPT2Config,
        patch_size: int = 16
    ):
        super().__init__(encoder_config)
        self.patch_size = patch_size
        
        self.patch_level_decoder = PatchLevelDecoder(encoder_config, patch_size)
        self.char_level_decoder = CharLevelDecoder(decoder_config)
    
    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "NotaGenLMHeadModel":
        """Create model from a ModelConfig."""
        patch_config = GPT2Config(
            num_hidden_layers=config.patch_num_layers,
            max_length=config.patch_length,
            max_position_embeddings=config.patch_length,
            n_embd=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            vocab_size=1,
        )
        
        char_config = GPT2Config(
            num_hidden_layers=config.char_num_layers,
            max_length=config.patch_size + 1,
            max_position_embeddings=config.patch_size + 1,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            vocab_size=128,
        )
        
        return cls(patch_config, char_config, config.patch_size)
    
    def forward(
        self,
        patches: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            patches: Input patches [batch, seq_len * patch_size]
            masks: Attention mask [batch, seq_len]
            
        Returns:
            Language modeling loss
        """
        # Reshape to [batch, seq_len, patch_size]
        patches = patches.reshape(len(patches), -1, self.patch_size)
        
        # Get patch embeddings
        encoded = self.patch_level_decoder(patches, masks)["last_hidden_state"]
        
        # Create left-shifted masks for next-token prediction
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks_for_targets = masks.clone()
        masks_for_targets[:, 0] = 0
        
        # Select patches with valid masks
        encoded_patches = encoded[left_shift_masks == 1]
        target_patches = patches[masks_for_targets == 1]
        
        return self.char_level_decoder(encoded_patches, target_patches)
    
    def generate(
        self,
        patches: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0
    ) -> List[int]:
        """
        Generate the next patch given context patches.
        
        Args:
            patches: Context patches [1, 1, seq_len * patch_size]
            top_k: Top-k filtering parameter (0 = disabled)
            top_p: Nucleus sampling parameter (1.0 = disabled)
            temperature: Temperature for sampling
            
        Returns:
            List of character IDs for the generated patch
        """
        # Handle partial patches at the end
        if patches.shape[-1] % self.patch_size != 0:
            remainder = patches.shape[-1] % self.patch_size
            tokens = patches[:, :, -remainder:].squeeze(0).squeeze(0)
            tokens = torch.cat([
                torch.tensor([self.BOS_TOKEN_ID], device=self.device),
                tokens
            ], dim=-1)
            patches = patches[:, :, :-remainder]
        else:
            tokens = torch.tensor([self.BOS_TOKEN_ID], device=self.device)
        
        # Reshape and encode patches
        patches = patches.reshape(1, -1, self.patch_size)
        encoded = self.patch_level_decoder(patches)["last_hidden_state"]
        
        # Get the last patch embedding for generation
        last_patch_embed = encoded[0, -1]
        
        # Generate characters
        generated = []
        while True:
            # Get next token probabilities
            probs = self.char_level_decoder.generate(last_patch_embed, tokens)
            probs = probs.cpu().detach().numpy()
            
            # Sample next token with filtering
            token = sample_token(
                probs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            
            generated.append(token)
            
            # Check stopping conditions
            if len(tokens) >= self.patch_size:
                break
            
            # Append token for next iteration
            tokens = torch.cat([
                tokens,
                torch.tensor([token], device=self.device)
            ], dim=0)
        
        return generated
    
    def generate_stream(
        self,
        patches: torch.Tensor,
        patchilizer,
        max_patches: int = 1024,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        callback=None
    ):
        """
        Generate a stream of patches with context window management.
        
        Args:
            patches: Initial context patches
            patchilizer: Patchilizer instance for encoding
            max_patches: Maximum number of patches to generate
            top_k: Top-k filtering parameter
            top_p: Nucleus sampling parameter
            temperature: Temperature for sampling
            callback: Optional callback(char) called for each generated character
            
        Yields:
            Generated characters one at a time
        """
        # This would be implemented similarly to the inference loop
        # in the original code, but as a generator
        raise NotImplementedError("Stream generation not yet implemented")
