"""
Patchilizer: Tokenization for NotaGen's Patch-based Architecture

The Patchilizer converts ABC notation music into patches for the hierarchical
decoder architecture. It handles:
- Splitting music into bars based on bar lines
- Creating fixed-size patches for the model
- Stream encoding for long pieces
- Proper handling of metadata vs tune body
"""

from __future__ import annotations
import re
import bisect
import random
from typing import List, Optional, Tuple
from .config import ModelConfig


class Patchilizer:
    """
    Tokenizer for NotaGen's patch-based architecture.
    
    Converts ABC notation text into sequences of fixed-size patches,
    where each patch is a sequence of character IDs.
    """
    
    # Bar line delimiters in ABC notation
    DELIMITERS = ["|:", "::", ":|", "[|", "||", "|]", "|"]
    
    # Special token IDs
    SPECIAL_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    
    def __init__(
        self,
        patch_size: int = 16,
        patch_length: int = 1024,
        stream: bool = True
    ):
        """
        Initialize the Patchilizer.
        
        Args:
            patch_size: Number of characters per patch
            patch_length: Maximum number of patches in a sequence
            stream: Whether to use stream encoding for long pieces
        """
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.stream = stream
        
        # Compile regex pattern for bar splitting
        self.bar_pattern = re.compile(
            '(' + '|'.join(re.escape(d) for d in self.DELIMITERS) + ')'
        )
        
        # Aliases for compatibility
        self.bos_token_id = self.BOS_TOKEN_ID
        self.eos_token_id = self.EOS_TOKEN_ID
        self.special_token_id = self.SPECIAL_TOKEN_ID
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> "Patchilizer":
        """Create a Patchilizer from a ModelConfig."""
        return cls(
            patch_size=config.patch_size,
            patch_length=config.patch_length,
            stream=config.patch_stream,
        )
    
    def split_bars(self, body_lines: List[str]) -> List[str]:
        """
        Split tune body lines into individual bars.
        
        Args:
            body_lines: List of tune body lines
            
        Returns:
            List of bars (each bar is a string)
        """
        bars = []
        try:
            for line in body_lines:
                line_bars = self.bar_pattern.split(line)
                line_bars = [b for b in line_bars if b]  # Remove empty strings
                
                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.DELIMITERS:
                        # Line starts with bar line - pair them up
                        new_line_bars = [
                            line_bars[i] + line_bars[i + 1]
                            for i in range(0, len(line_bars) - 1, 2)
                        ]
                        if len(line_bars) % 2 == 1:
                            new_line_bars.append(line_bars[-1])
                    else:
                        # Line starts with content
                        new_line_bars = [line_bars[0]] + [
                            line_bars[i] + line_bars[i + 1]
                            for i in range(1, len(line_bars) - 1, 2)
                        ]
                        if len(line_bars) % 2 == 0:
                            new_line_bars.append(line_bars[-1])
                    
                    # Absorb trailing bar line + newline
                    if len(new_line_bars) >= 2 and 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]
                        new_line_bars = new_line_bars[:-1]
                
                bars.extend(new_line_bars)
        except Exception:
            pass
        
        return bars
    
    def split_patches(
        self,
        text: str,
        generate_last: bool = False
    ) -> List[str]:
        """
        Split text into fixed-size patches.
        
        Args:
            text: Text to split
            generate_last: If True, don't pad the last patch (for generation)
            
        Returns:
            List of patch strings
        """
        if not generate_last and len(text) % self.patch_size != 0:
            text += chr(self.EOS_TOKEN_ID)
        
        return [
            text[i:i + self.patch_size]
            for i in range(0, len(text), self.patch_size)
        ]
    
    def patch2chars(self, patch: List[int]) -> str:
        """
        Convert a patch (list of character IDs) back to a string.
        
        Args:
            patch: List of character IDs
            
        Returns:
            Decoded string
        """
        chars = []
        for idx in patch:
            if idx == self.EOS_TOKEN_ID:
                break
            if idx > self.EOS_TOKEN_ID:
                chars.append(chr(idx))
        return ''.join(chars)
    
    def patchilize_metadata(self, metadata_lines: List[str]) -> List[str]:
        """Convert metadata lines to patches."""
        patches = []
        for line in metadata_lines:
            patches.extend(self.split_patches(line))
        return patches
    
    def patchilize_tunebody(
        self,
        tunebody_lines: List[str],
        encode_mode: str = 'train'
    ) -> List[str]:
        """
        Convert tune body lines to patches.
        
        Args:
            tunebody_lines: List of tune body lines
            encode_mode: 'train' or 'generate'
            
        Returns:
            List of patch strings
        """
        bars = self.split_bars(tunebody_lines)
        patches = []
        
        if encode_mode == 'train':
            for bar in bars:
                patches.extend(self.split_patches(bar))
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                patches.extend(self.split_patches(bar))
            if bars:
                patches.extend(self.split_patches(bars[-1], generate_last=True))
        
        return patches
    
    def _find_tunebody_index(
        self,
        lines: List[str],
        for_generation: bool = False
    ) -> Optional[int]:
        """Find the index where tune body starts."""
        for i, line in enumerate(lines):
            if for_generation:
                if line.startswith('[V:') or line.startswith('[r:'):
                    return i
            else:
                if '[V:' in line:
                    return i
        return None if for_generation else -1
    
    def encode_train(
        self,
        abc_text: str,
        add_special_patches: bool = True,
        cut: bool = True
    ) -> List[List[int]]:
        """
        Encode ABC text for training.
        
        Args:
            abc_text: Full ABC notation text
            add_special_patches: Whether to add BOS/EOS patches
            cut: Whether to truncate to patch_length
            
        Returns:
            List of patches, where each patch is a list of character IDs
        """
        lines = [line + '\n' for line in abc_text.split('\n') if line]
        
        tunebody_index = self._find_tunebody_index(lines)
        if tunebody_index is None or tunebody_index < 0:
            tunebody_index = len(lines)
        
        metadata_lines = lines[:tunebody_index]
        tunebody_lines = lines[tunebody_index:]
        
        # Add stream markers if enabled
        if self.stream:
            tunebody_lines = [
                f'[r:{i}/{len(tunebody_lines) - i - 1}]{line}'
                for i, line in enumerate(tunebody_lines)
            ]
        
        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')
        
        # Add special patches
        if add_special_patches:
            bos_patch = chr(self.BOS_TOKEN_ID) * (self.patch_size - 1) + chr(self.EOS_TOKEN_ID)
            eos_patch = chr(self.BOS_TOKEN_ID) + chr(self.EOS_TOKEN_ID) * (self.patch_size - 1)
            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]
        
        # Handle stream cutting for long pieces
        if self.stream and len(metadata_patches) + len(tunebody_patches) > self.patch_length:
            patches = self._stream_cut(
                metadata_patches, tunebody_patches, tunebody_lines,
                add_special_patches
            )
        else:
            patches = metadata_patches + tunebody_patches
        
        if cut:
            patches = patches[:self.patch_length]
        
        # Convert to IDs
        return [
            [ord(c) for c in patch] + [self.SPECIAL_TOKEN_ID] * (self.patch_size - len(patch))
            for patch in patches
        ]
    
    def _stream_cut(
        self,
        metadata_patches: List[str],
        tunebody_patches: List[str],
        tunebody_lines: List[str],
        add_special_patches: bool
    ) -> List[str]:
        """Handle stream cutting for pieces longer than patch_length."""
        available_cut_indexes = [0] + [
            i + 1 for i, patch in enumerate(tunebody_patches) if '\n' in patch
        ]
        line_index_for_cut_index = list(range(len(available_cut_indexes)))
        
        end_index = len(metadata_patches) + len(tunebody_patches) - self.patch_length
        biggest_index = bisect.bisect_left(available_cut_indexes, end_index)
        available_cut_indexes = available_cut_indexes[:biggest_index + 1]
        
        if len(available_cut_indexes) == 1:
            choices = ['head']
        elif len(available_cut_indexes) == 2:
            choices = ['head', 'tail']
        else:
            choices = ['head', 'tail', 'middle']
        
        choice = random.choice(choices)
        
        if choice == 'head':
            return metadata_patches + tunebody_patches
        
        if choice == 'tail':
            cut_index = len(available_cut_indexes) - 1
        else:
            cut_index = random.choice(range(1, len(available_cut_indexes) - 1))
        
        line_index = line_index_for_cut_index[cut_index]
        stream_tunebody_lines = tunebody_lines[line_index:]
        
        stream_tunebody_patches = self.patchilize_tunebody(
            stream_tunebody_lines, encode_mode='train'
        )
        
        if add_special_patches:
            eos_patch = chr(self.BOS_TOKEN_ID) + chr(self.EOS_TOKEN_ID) * (self.patch_size - 1)
            stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
        
        return metadata_patches + stream_tunebody_patches
    
    def encode_generate(
        self,
        abc_code: str,
        add_special_patches: bool = True
    ) -> List[List[int]]:
        """
        Encode ABC text for generation (continuing from a prefix).
        
        Args:
            abc_code: Partial ABC notation text to continue from
            add_special_patches: Whether to add BOS patch
            
        Returns:
            List of patches, where each patch is a list of character IDs
        """
        lines = [line for line in abc_code.split('\n') if line]
        
        tunebody_index = self._find_tunebody_index(lines, for_generation=True)
        
        metadata_lines = lines[:tunebody_index]
        tunebody_lines = lines[tunebody_index:]
        
        # Add newlines
        metadata_lines = [line + '\n' for line in metadata_lines]
        
        if self.stream:
            if not abc_code.endswith('\n'):
                tunebody_lines = (
                    [line + '\n' for line in tunebody_lines[:-1]] +
                    [tunebody_lines[-1]] if tunebody_lines else []
                )
            else:
                tunebody_lines = [line + '\n' for line in tunebody_lines]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]
        
        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')
        
        if add_special_patches:
            bos_patch = chr(self.BOS_TOKEN_ID) * (self.patch_size - 1) + chr(self.EOS_TOKEN_ID)
            metadata_patches = [bos_patch] + metadata_patches
        
        patches = (metadata_patches + tunebody_patches)[:self.patch_length]
        
        # Convert to IDs - handle partial patches for generation
        id_patches = []
        for patch in patches:
            if len(patch) < self.patch_size and patch and patch[-1] != chr(self.EOS_TOKEN_ID):
                # Partial patch for generation - don't pad
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.SPECIAL_TOKEN_ID] * (self.patch_size - len(patch))
            id_patches.append(id_patch)
        
        return id_patches
    
    def decode(self, patches: List[List[int]]) -> str:
        """
        Decode patches back to ABC notation text.
        
        Args:
            patches: List of patches (each is a list of character IDs)
            
        Returns:
            Decoded ABC notation string
        """
        return ''.join(self.patch2chars(patch) for patch in patches)
    
    # Alias for compatibility
    encode = encode_train
