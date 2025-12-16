# =============================================================================
# IMPORTANT: Suppress TensorFlow/JAX warnings BEFORE any imports
# These must be set before tensorflow/jax are imported (even indirectly)
# =============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['JAX_PLATFORMS'] = ''
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='
# Additional C++ log suppression
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

import sys
import time
import re
import warnings
from typing import Optional, List, Generator, Callable, Any

# Suppress Python-level warnings aggressively
warnings.filterwarnings('ignore')

import torch

# Ensure we can find our modules regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import *
from config import INFERENCE_WEIGHTS_PATH, PATCH_SIZE, PATCH_LENGTH, PATCH_NUM_LAYERS, CHAR_NUM_LAYERS, HIDDEN_SIZE, TOP_K, TOP_P, TEMPERATURE
from transformers import GPT2Config
from abctoolkit.utils import Barline_regexPattern
from abctoolkit.transpose import Note_list
from abctoolkit.duration import calculate_bartext_duration

Note_list = Note_list + ['z', 'x']

# =============================================================================
# End-of-Piece Detection Patterns
# =============================================================================

# Minimum content length before we even consider checking for completion
# This prevents premature termination on just the preamble
MIN_CONTENT_FOR_COMPLETION = 500  # At least 500 chars of actual content

# Pattern for detecting start of a new piece (after first one)
NEW_PIECE_PATTERN = re.compile(r'^%[A-Z][a-z]+\s*$')  # %Classical, %Romantic, etc.

# =============================================================================
# Device Selection
# =============================================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple MPS")
else:
    device = torch.device("cpu")
    print("💻 Using CPU (GPU not available)")

# =============================================================================
# Model Setup
# =============================================================================

patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                          max_length=PATCH_LENGTH,
                          max_position_embeddings=PATCH_LENGTH,
                          n_embd=HIDDEN_SIZE,
                          num_attention_heads=HIDDEN_SIZE // 64,
                          vocab_size=1)
byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                         max_length=PATCH_SIZE + 1,
                         max_position_embeddings=PATCH_SIZE + 1,
                         hidden_size=HIDDEN_SIZE,
                         num_attention_heads=HIDDEN_SIZE // 64,
                         vocab_size=128)

model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config).to(device)


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """Prepare model for inference with mixed precision."""
    model = model.to(dtype=torch.float16)
    for param in model.parameters():
        if param.dtype == torch.float32:
            param.requires_grad = False
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Validate weights path before loading
if not INFERENCE_WEIGHTS_PATH:
    raise FileNotFoundError(
        "\n❌ NotaGen weights file not found!\n\n"
        "Please download the weights first:\n"
        "  wget https://huggingface.co/ElectricAlexis/NotaGen/resolve/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth\n\n"
        "Or set the NOTAGEN_WEIGHTS environment variable:\n"
        "  export NOTAGEN_WEIGHTS=/path/to/weights.pth"
    )

if not os.path.exists(INFERENCE_WEIGHTS_PATH):
    raise FileNotFoundError(
        f"\n❌ Weights file not found at: {INFERENCE_WEIGHTS_PATH}\n\n"
        "Please download the weights or check the path."
    )

print(f"📥 Loading weights from: {INFERENCE_WEIGHTS_PATH}")

checkpoint = torch.load(INFERENCE_WEIGHTS_PATH, map_location=torch.device(device), weights_only=False)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()


def complete_brackets(s):
    stack = []
    bracket_map = {'{': '}', '[': ']', '(': ')'}
    
    # Iterate through each character, handle bracket matching
    for char in s:
        if char in bracket_map:
            stack.append(char)
        elif char in bracket_map.values():
            # Find the corresponding left bracket
            for key, value in bracket_map.items():
                if value == char:
                    if stack and stack[-1] == key:
                        stack.pop()
                    break  # Found matching right bracket, process next character
    
    # Complete missing right brackets (in reverse order of remaining left brackets in stack)
    completion = ''.join(bracket_map[c] for c in reversed(stack))
    return s + completion


def rest_unreduce(abc_lines):

    tunebody_index = None
    for i in range(len(abc_lines)):
        if abc_lines[i].startswith('%%score'):
            abc_lines[i] = complete_brackets(abc_lines[i])
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []  # voices that use z as rest
    x_symbol_list = []  # voices that use x as rest
    for voice_group in voice_group_list:
        z_symbol_list.append('V:' + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append('V:' + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''

        line = re.sub(r'^\[r:[^\]]*\]', '', line)

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        # calculate duration and collect barline
        dur_dict = {}  
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[:-len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except:
            pass    # use last ref_dur

        if i == 0:
            prefix_left_barline = line.split('[V:')[0]
        else:
            prefix_left_barline = ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


# =============================================================================
# End-of-Piece Detection
# =============================================================================

def check_piece_complete(text: str, pieces_found: int = 0) -> tuple[bool, int]:
    """
    Check if the generated text contains a complete piece or multiple pieces.
    
    A piece is considered complete when:
    1. There's substantial content (not just metadata/preamble)
    2. The content ends with a final barline pattern
    3. OR the model has naturally stopped (BOS+EOS token)
    
    Returns:
        (is_complete, piece_count): Whether a piece is complete and how many pieces found
    """
    # Don't consider complete until we have substantial content
    if len(text) < MIN_CONTENT_FOR_COMPLETION:
        return False, 0
    
    lines = text.strip().split('\n')
    
    # Must have actual tunebody content (lines with [V: or [r: patterns)
    has_tunebody = any('[V:' in line or '[r:' in line for line in lines)
    if not has_tunebody:
        return False, 0
    
    # Count piece markers (period headers like %Classical, %Romantic, etc.)
    piece_starts = 0
    for line in lines:
        if line.startswith('%') and not line.startswith('%%'):
            content = line[1:].strip()
            if content and content[0].isupper():
                if content in ['Classical', 'Baroque', 'Romantic', 'Renaissance', 'Modern', '20th Century', 'Medieval']:
                    piece_starts += 1
    
    # Check for definitive end markers - must be at the ACTUAL end of the text
    # Look at the last few non-empty lines
    non_empty_lines = [l for l in lines if l.strip()]
    if not non_empty_lines:
        return False, 0
    
    # Get the last line with actual content
    last_content_line = non_empty_lines[-1].strip()
    
    # Check if it ends with a final barline
    # Final barlines: |] or ||  (but NOT :|] which is just a repeat, need context)
    has_final_barline = (
        last_content_line.endswith('|]') or 
        last_content_line.endswith('||') or
        last_content_line.endswith(':|]')
    )
    
    # Also check for explicit %end marker at the very end
    has_end_marker = text.strip().endswith('%end') or text.strip().lower().endswith('%end')
    
    is_complete = has_final_barline or has_end_marker
    
    return is_complete, max(1, piece_starts)


def should_stop_generation(text: str, max_pieces: int = 1) -> bool:
    """
    Determine if generation should stop based on content.
    
    Args:
        text: Generated text so far
        max_pieces: Maximum number of pieces to generate
        
    Returns:
        True if generation should stop
    """
    is_complete, piece_count = check_piece_complete(text)
    
    # Stop if we've completed the desired number of pieces
    if is_complete and piece_count >= max_pieces:
        return True
    
    # Stop if we see a new piece starting after completing one
    if piece_count > max_pieces:
        return True
    
    return False


def truncate_to_complete_piece(text: str) -> str:
    """
    Truncate text to end at the last complete piece.
    Removes any partial content after the last end marker.
    """
    # Find the last occurrence of common end markers
    last_end = -1
    
    # Look for final barlines
    for match in re.finditer(r'\|]\s*\n', text):
        last_end = match.end()
    
    # Look for %end marker
    end_match = re.search(r'%end\s*\n?', text, re.IGNORECASE)
    if end_match and end_match.end() > last_end:
        last_end = end_match.end()
    
    if last_end > 0:
        return text[:last_end].strip() + '\n'
    
    return text


# =============================================================================
# Main Inference Function (Enhanced)
# =============================================================================

def inference_patch(
    period: str = None,
    composer: str = None,
    instrumentation: str = None,
    top_k: int = None,
    top_p: float = None,
    temperature: float = None,
    # Custom prompt support
    custom_prompt: str = None,
    additional_preamble: str = None,
    # Termination settings
    max_pieces: int = 1,
    max_time: int = 600,
    max_bytes: int = 102400,
    stop_on_complete: bool = True,
    # Callbacks
    on_token: Callable[[str], None] = None,
) -> Optional[str]:
    """
    Generate music notation using the NotaGen model.
    
    Args:
        period: Musical period (e.g., 'Classical', 'Romantic')
        composer: Composer name
        instrumentation: Instrument description
        top_k: Top-k sampling parameter (default: from config)
        top_p: Top-p (nucleus) sampling parameter (default: from config)
        temperature: Sampling temperature (default: from config)
        custom_prompt: Custom prompt lines (overrides period/composer/instrumentation)
        additional_preamble: Additional ABC notation to append after prompt
        max_pieces: Maximum number of pieces to generate (default: 1)
        max_time: Maximum generation time in seconds (default: 600)
        max_bytes: Maximum bytes to generate (default: 102400)
        stop_on_complete: Stop when a complete piece is detected (default: True)
        on_token: Callback function called for each generated token
    
    Returns:
        Generated ABC notation string, or None if generation failed
    """
    # Use config defaults if not specified
    if top_k is None:
        top_k = TOP_K
    if top_p is None:
        top_p = TOP_P
    if temperature is None:
        temperature = TEMPERATURE

    # Build prompt lines
    if custom_prompt and custom_prompt.strip():
        # Use custom prompt directly
        prompt_lines = []
        for line in custom_prompt.strip().split('\n'):
            if not line.endswith('\n'):
                line += '\n'
            prompt_lines.append(line)
    else:
        # Use standard prompt format
        prompt_lines = []
        if period:
            prompt_lines.append('%' + period + '\n')
        if composer:
            prompt_lines.append('%' + composer + '\n')
        if instrumentation:
            prompt_lines.append('%' + instrumentation + '\n')
    
    # Add additional preamble if specified
    if additional_preamble and additional_preamble.strip():
        for line in additional_preamble.strip().split('\n'):
            if not line.endswith('\n'):
                line += '\n'
            prompt_lines.append(line)

    # Termination tracking
    pieces_generated = 0
    generation_start_time = time.time()
    total_bytes = 0

    while True:

        failure_flag = False

        bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

        start_time = time.time()

        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        context_tunebody_byte_list = []
        metadata_byte_list = []

        print(''.join(byte_list), end='')

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

        end_flag = False
        cut_index = None

        tunebody_flag = False

        with torch.inference_mode():
            
            while True:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    predicted_patch = model.generate(input_patches.unsqueeze(0),
                                                    top_k=top_k,
                                                    top_p=top_p,
                                                    temperature=temperature)
                if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # 初次进入tunebody，必须以[r:0/开头
                    tunebody_flag = True
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                    predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                    top_k=top_k,
                                                    top_p=top_p,
                                                    temperature=temperature)
                    predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
                if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                    end_flag = True
                    break
                next_patch = patchilizer.decode([predicted_patch])

                for char in next_patch:
                    byte_list.append(char)
                    if tunebody_flag:
                        context_tunebody_byte_list.append(char)
                    else:
                        metadata_byte_list.append(char)
                    print(char, end='', flush=True)
                    # Callback for token progress
                    if on_token:
                        on_token(char)

                patch_end_flag = False
                for j in range(len(predicted_patch)):
                    if patch_end_flag:
                        predicted_patch[j] = patchilizer.special_token_id
                    if predicted_patch[j] == patchilizer.eos_token_id:
                        patch_end_flag = True

                predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)
                input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

                # Check termination conditions
                current_text = ''.join(byte_list)
                total_bytes = len(byte_list)
                elapsed_time = time.time() - generation_start_time
                
                # User-configurable limits
                if total_bytes > max_bytes:
                    print(f'\n[Stopped: max bytes {max_bytes} reached]')
                    failure_flag = True
                    break
                if elapsed_time > max_time:
                    print(f'\n[Stopped: max time {max_time}s reached]')
                    failure_flag = True
                    break
                
                # Smart termination: stop when piece is complete
                if stop_on_complete and should_stop_generation(current_text, max_pieces):
                    print('\n[Piece complete - stopping generation]')
                    end_flag = True
                    break

                if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
                    print('Stream generating...')

                    metadata = ''.join(metadata_byte_list)
                    context_tunebody = ''.join(context_tunebody_byte_list)

                    if '\n' not in context_tunebody:
                        break   # Generated content is all metadata, abandon

                    context_tunebody_lines = context_tunebody.split('\n')
                    if not context_tunebody.endswith('\n'):
                        context_tunebody_lines = [context_tunebody_lines[i] + '\n' for i in range(len(context_tunebody_lines) - 1)] + [context_tunebody_lines[-1]]
                    else:
                        context_tunebody_lines = [context_tunebody_lines[i] + '\n' for i in range(len(context_tunebody_lines))]

                    cut_index = len(context_tunebody_lines) // 2
                    abc_code_slice = metadata + ''.join(context_tunebody_lines[-cut_index:])

                    input_patches = patchilizer.encode_generate(abc_code_slice)

                    input_patches = [item for sublist in input_patches for item in sublist]
                    input_patches = torch.tensor([input_patches], device=device)
                    input_patches = input_patches.reshape(1, -1)

                    context_tunebody_byte_list = list(''.join(context_tunebody_lines[-cut_index:]))

            if not failure_flag:
                abc_text = ''.join(byte_list)

                # unreduce
                abc_lines = abc_text.split('\n')
                abc_lines = list(filter(None, abc_lines))
                abc_lines = [line + '\n' for line in abc_lines]
                try:
                    unreduced_abc_lines = rest_unreduce(abc_lines)
                except:
                    failure_flag = True
                    pass
                else:
                    unreduced_abc_lines = [line for line in unreduced_abc_lines if not(line.startswith('%') and not line.startswith('%%'))]
                    unreduced_abc_lines = ['X:1\n'] + unreduced_abc_lines
                    unreduced_abc_text = ''.join(unreduced_abc_lines)
                    
                    # Truncate to complete piece if needed
                    if stop_on_complete:
                        unreduced_abc_text = truncate_to_complete_piece(unreduced_abc_text)
                    
                    return unreduced_abc_text
    
    # If we exit the outer loop without returning, return None
    return None


def inference_batch(
    period: str = None,
    composer: str = None,
    instrumentation: str = None,
    num_variations: int = 3,
    top_k: int = None,
    top_p: float = None,
    temperature: float = None,
    custom_prompt: str = None,
    additional_preamble: str = None,
    max_time_per_piece: int = 300,
    stop_on_complete: bool = True,
    on_progress: Callable[[int, int, str], None] = None,
) -> List[str]:
    """
    Generate multiple variations using the NotaGen model.
    
    Args:
        period: Musical period
        composer: Composer name  
        instrumentation: Instrument description
        num_variations: Number of variations to generate
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        temperature: Sampling temperature
        custom_prompt: Custom prompt (overrides period/composer/instrumentation)
        additional_preamble: Additional ABC notation to append
        max_time_per_piece: Maximum time per piece in seconds
        stop_on_complete: Stop when piece is complete
        on_progress: Callback(current, total, status) for progress updates
        
    Returns:
        List of generated ABC notation strings
    """
    results = []
    
    for i in range(num_variations):
        if on_progress:
            on_progress(i + 1, num_variations, f"Generating variation {i + 1}/{num_variations}...")
        
        try:
            result = inference_patch(
                period=period,
                composer=composer,
                instrumentation=instrumentation,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                custom_prompt=custom_prompt,
                additional_preamble=additional_preamble,
                max_time=max_time_per_piece,
                stop_on_complete=stop_on_complete,
            )
            
            if result:
                results.append(result)
                if on_progress:
                    on_progress(i + 1, num_variations, f"✓ Variation {i + 1} complete")
            else:
                if on_progress:
                    on_progress(i + 1, num_variations, f"✗ Variation {i + 1} failed")
                    
        except Exception as e:
            print(f"Error generating variation {i + 1}: {e}")
            if on_progress:
                on_progress(i + 1, num_variations, f"✗ Error: {str(e)[:50]}")
    
    return results


if __name__ == '__main__':
    inference_patch('Classical', 'Beethoven, Ludwig van', 'Keyboard')
