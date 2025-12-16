"""
Export utilities for NotaGen.

Provides functions to convert ABC notation to MusicXML and MIDI formats.
Uses the built-in abc2xml converter and optionally music21 for MIDI.
"""

from __future__ import annotations

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

# Add parent directory for abc2xml import
_SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SCRIPT_DIR / 'gradio'))
sys.path.insert(0, str(_SCRIPT_DIR / 'data'))


def abc_to_musicxml(
    abc_content: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Convert ABC notation to MusicXML format.
    
    Uses the built-in abc2xml.py converter from the project.
    
    Args:
        abc_content: ABC notation string
        output_path: Output file path. If None, returns XML as string.
        title: Optional title for the output file naming
        
    Returns:
        Path to the output file, or XML string if output_path is None
        
    Raises:
        RuntimeError: If conversion fails
    """
    # Try to import the built-in abc2xml
    try:
        # Import the converter module
        import abc2xml as abc2xml_module
        
        # The abc2xml module has a vertaal function for conversion
        if hasattr(abc2xml_module, 'vertaal'):
            xml_result = abc2xml_module.vertaal(abc_content, 0)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(xml_result)
                return output_path
            return xml_result
    except ImportError:
        pass
    
    # Fallback: Use subprocess to call abc2xml.py script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.abc', delete=False) as f:
        f.write(abc_content)
        temp_abc = f.name
    
    try:
        # Find abc2xml.py script
        abc2xml_paths = [
            _SCRIPT_DIR / 'gradio' / 'abc2xml.py',
            _SCRIPT_DIR / 'data' / 'abc2xml.py',
            _SCRIPT_DIR / 'notebook' / 'abc2xml.py',
        ]
        
        abc2xml_script = None
        for p in abc2xml_paths:
            if p.exists():
                abc2xml_script = str(p)
                break
        
        if abc2xml_script is None:
            raise RuntimeError("abc2xml.py script not found")
        
        # Determine output path
        if output_path is None:
            output_path = temp_abc.replace('.abc', '.xml')
        
        output_dir = os.path.dirname(output_path) or '.'
        
        # Run conversion
        result = subprocess.run(
            [sys.executable, abc2xml_script, '-o', output_dir, temp_abc],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"abc2xml conversion failed: {result.stderr}")
        
        # abc2xml outputs to same directory with .xml extension
        expected_xml = temp_abc.replace('.abc', '.xml')
        if os.path.exists(expected_xml):
            if expected_xml != output_path:
                os.rename(expected_xml, output_path)
        
        return output_path
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_abc):
            os.remove(temp_abc)


def abc_to_midi(
    abc_content: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Convert ABC notation to MIDI format.
    
    First converts to MusicXML, then to MIDI using music21.
    
    Args:
        abc_content: ABC notation string
        output_path: Output file path. If None, generates temp file.
        title: Optional title for the piece
        
    Returns:
        Path to the output MIDI file
        
    Raises:
        RuntimeError: If conversion fails
        ImportError: If music21 is not installed
    """
    try:
        import music21
    except ImportError:
        raise ImportError(
            "music21 is required for MIDI export. "
            "Install it with: pip install music21"
        )
    
    # First convert to MusicXML
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
        temp_xml = f.name
    
    try:
        abc_to_musicxml(abc_content, temp_xml, title)
        
        # Parse MusicXML with music21
        score = music21.converter.parse(temp_xml)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = temp_xml.replace('.xml', '.mid')
        
        # Write MIDI
        score.write('midi', fp=output_path)
        
        return output_path
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_xml):
            os.remove(temp_xml)


def save_output(
    abc_content: str,
    output_path: str,
    format: str = 'abc',
    period: Optional[str] = None,
    composer: Optional[str] = None,
    instrumentation: Optional[str] = None,
) -> str:
    """
    Save generated content to a file in the specified format.
    
    Args:
        abc_content: ABC notation string
        output_path: Output file path (extension may be adjusted based on format)
        format: Output format ('abc', 'xml', 'musicxml', 'midi')
        period: Musical period (for metadata)
        composer: Composer name (for metadata)
        instrumentation: Instrumentation (for metadata)
        
    Returns:
        Path to the saved file
    """
    # Ensure correct extension
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    
    if format == 'abc':
        output_path = base_path + '.abc'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(abc_content)
        return output_path
        
    elif format in ('xml', 'musicxml'):
        output_path = base_path + '.xml'
        return abc_to_musicxml(abc_content, output_path)
        
    elif format == 'midi':
        output_path = base_path + '.mid'
        return abc_to_midi(abc_content, output_path)
        
    else:
        raise ValueError(f"Unknown format: {format}. Use 'abc', 'xml', 'musicxml', or 'midi'.")


def generate_filename(
    period: str,
    composer: str,
    instrumentation: Optional[str] = None,
    extension: str = 'abc',
    include_timestamp: bool = True,
) -> str:
    """
    Generate a descriptive filename for saved output.
    
    Args:
        period: Musical period
        composer: Composer name
        instrumentation: Optional instrumentation
        extension: File extension
        include_timestamp: Whether to include timestamp
        
    Returns:
        Generated filename
    """
    # Clean inputs
    def clean(s: str) -> str:
        return s.replace(' ', '_').replace('/', '-')
    
    parts = [clean(period), clean(composer)]
    if instrumentation:
        parts.append(clean(instrumentation))
    
    filename = '_'.join(parts)
    
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
    
    return f"{filename}.{extension}"


# Type aliases for export
__all__ = [
    'abc_to_musicxml',
    'abc_to_midi',
    'save_output',
    'generate_filename',
]
