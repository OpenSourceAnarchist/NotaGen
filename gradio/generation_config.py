# =============================================================================
# Generation Configuration and Presets for NotaGen
# =============================================================================
"""
Centralized configuration for music generation including:
- Sampling presets (temperature, top_k, top_p)
- Composition presets (tempo, key, time signature, style)
- ABC notation preamble generation
"""

from dataclasses import dataclass
from typing import Optional, List

# =============================================================================
# Sampling Presets
# =============================================================================

SAMPLING_PRESETS = {
    "Balanced (Default)": {"top_k": 9, "top_p": 0.9, "temperature": 1.2},
    "Conservative": {"top_k": 5, "top_p": 0.85, "temperature": 0.9},
    "Creative": {"top_k": 15, "top_p": 0.95, "temperature": 1.5},
    "Very Conservative": {"top_k": 3, "top_p": 0.7, "temperature": 0.7},
    "Experimental": {"top_k": 20, "top_p": 0.98, "temperature": 1.8},
}

# =============================================================================
# Tempo Presets (with BPM ranges)
# =============================================================================

TEMPO_PRESETS = {
    "Largo": (40, 60),
    "Adagio": (66, 76),
    "Andante": (76, 108),
    "Moderato": (108, 120),
    "Allegretto": (112, 120),
    "Allegro": (120, 156),
    "Vivace": (156, 176),
    "Presto": (168, 200),
}

# =============================================================================
# Key Signature Presets
# =============================================================================

KEY_SIGNATURES = [
    # Major keys
    "C", "G", "D", "A", "E", "B", "F#",
    "F", "Bb", "Eb", "Ab", "Db",
    # Minor keys
    "Am", "Em", "Bm", "F#m", "C#m",
    "Dm", "Gm", "Cm", "Fm", "Bbm",
]

# =============================================================================
# Time Signature Presets
# =============================================================================

TIME_SIGNATURES = [
    "4/4", "3/4", "2/4", "6/8",
    "2/2", "3/8", "9/8", "12/8", "5/4",
]

# =============================================================================
# Style/Expression Presets
# =============================================================================

STYLE_PRESETS = {
    "Cantabile": "cantabile",
    "Legato": "legato",
    "Staccato": "staccato",
    "Dolce": "dolce",
    "Espressivo": "espressivo",
    "Maestoso": "maestoso",
    "Grazioso": "grazioso",
    "Con brio": "con brio",
    "Con fuoco": "con fuoco",
    "Tranquillo": "tranquillo",
}

# =============================================================================
# Unit Length Presets
# =============================================================================

UNIT_LENGTHS = {
    "1/8": "1/8",
    "1/4": "1/4",
    "1/16": "1/16",
}


@dataclass
class GenerationConfig:
    """Configuration for a single generation request."""
    # Basic prompt
    period: str = ""
    composer: str = ""
    instrumentation: str = ""
    
    # Custom prompt (overrides basic prompt if set)
    custom_prompt: str = ""
    use_custom_prompt: bool = False
    
    # Additional preamble (appended after prompt)
    additional_preamble: str = ""
    
    # Composition settings
    tempo: Optional[str] = None
    tempo_bpm: Optional[int] = None
    key_signature: Optional[str] = None
    time_signature: Optional[str] = None
    unit_length: Optional[str] = None
    style_marking: Optional[str] = None
    
    # Sampling parameters
    top_k: int = 9
    top_p: float = 0.9
    temperature: float = 1.2
    
    # Generation limits
    max_patches: int = 2048
    max_time_seconds: int = 600
    max_bytes: int = 102400
    
    # Termination settings
    stop_on_end_marker: bool = True
    max_pieces: int = 1  # Stop after N complete pieces
    
    # Batch settings
    num_variations: int = 1
    
    def build_prompt_lines(self) -> List[str]:
        """Build the prompt lines for generation."""
        lines: List[str] = []
        
        if self.use_custom_prompt and self.custom_prompt.strip():
            # Use custom prompt directly
            for line in self.custom_prompt.strip().split('\n'):
                if not line.endswith('\n'):
                    line += '\n'
                lines.append(line)
        else:
            # Use standard prompt format
            if self.period:
                lines.append(f'%{self.period}\n')
            if self.composer:
                lines.append(f'%{self.composer}\n')
            if self.instrumentation:
                lines.append(f'%{self.instrumentation}\n')
        
        # Add additional preamble if specified
        if self.additional_preamble.strip():
            for line in self.additional_preamble.strip().split('\n'):
                if not line.endswith('\n'):
                    line += '\n'
                lines.append(line)
        
        return lines
    
    def build_metadata_hints(self) -> str:
        """Build ABC notation metadata hints based on composition settings."""
        hints: List[str] = []
        
        # Tempo marking with BPM
        if self.tempo_bpm:
            hints.append(f'Q:1/4={self.tempo_bpm}')
        
        # Time signature
        if self.time_signature:
            hints.append(f'M:{self.time_signature}')
        
        # Unit length
        if self.unit_length:
            hints.append(f'L:{self.unit_length}')
        
        # Key signature
        if self.key_signature:
            hints.append(f'K:{self.key_signature}')
        
        return '\n'.join(hints) + '\n' if hints else ''


def get_tempo_bpm(tempo_preset: str) -> Optional[int]:
    """Get a reasonable BPM value for a tempo preset."""
    if tempo_preset and tempo_preset in TEMPO_PRESETS:
        bpm_range = TEMPO_PRESETS[tempo_preset]
        if bpm_range and isinstance(bpm_range, tuple):
            # Return middle of range
            low, high = bpm_range
            return (low + high) // 2
    return None


def get_tempo_marking(tempo_preset: str) -> Optional[str]:
    """Get the tempo marking text for a preset."""
    if tempo_preset and tempo_preset in TEMPO_PRESETS:
        # The key itself is the marking
        return tempo_preset
    return None
