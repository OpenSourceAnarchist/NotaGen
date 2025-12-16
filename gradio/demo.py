# =============================================================================
# CRITICAL: Prevent TensorFlow from registering CUDA before PyTorch
# These MUST be set before ANY imports that could trigger TF loading
# =============================================================================
import os

# Completely disable TensorFlow's CUDA support to prevent factory registration conflicts
# This is the ROOT CAUSE of the cuDNN/cuBLAS "already registered" errors
os.environ['CUDA_VISIBLE_DEVICES_FOR_TF'] = ''  # Hide GPUs from TF specifically
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = ''
os.environ['TF_CUDA_PATHS'] = ''

# Tell TensorFlow to use CPU only (prevents CUDA factory registration)
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Temporarily hide GPUs

# Now import TensorFlow-related suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

# Disable JAX GPU
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = ''

# Abseil logging (the source of those E0000/W0000 messages)
os.environ['ABSL_LOGGING_LEVEL'] = 'FATAL'

import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow first (if it gets imported) to let it initialize on CPU
try:
    import tensorflow as tf
    # Force TF to CPU mode
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass

# NOW restore CUDA visibility for PyTorch
# Get the original CUDA devices (or default to all)
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('ORIGINAL_CUDA_VISIBLE_DEVICES', '0')

import gradio as gr
import sys
import threading
import queue
from io import TextIOBase
import datetime
import subprocess

# =============================================================================
# Path Setup - Ensure we can find all resources regardless of cwd
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project root to path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change to script directory so relative imports work
os.chdir(SCRIPT_DIR)

# Now import after path is set up
from inference import inference_patch, inference_batch
from generation_config import (
    SAMPLING_PRESETS, TEMPO_PRESETS, KEY_SIGNATURES, 
    TIME_SIGNATURES, STYLE_PRESETS as COMPOSITION_STYLES, UNIT_LENGTHS
)

# =============================================================================
# Style Presets (Sampling)
# =============================================================================

STYLE_PRESETS = {
    "Balanced (Default)": {"top_k": 9, "top_p": 0.9, "temperature": 1.2},
    "Conservative": {"top_k": 5, "top_p": 0.85, "temperature": 0.9},
    "Creative": {"top_k": 15, "top_p": 0.95, "temperature": 1.5},
    "Very Conservative": {"top_k": 3, "top_p": 0.7, "temperature": 0.7},
    "Experimental": {"top_k": 20, "top_p": 0.98, "temperature": 1.8},
}

# =============================================================================
# Composition Presets for Custom Prompts
# =============================================================================

# Tempo options for dropdown
TEMPO_OPTIONS = ["(Auto)"] + list(TEMPO_PRESETS.keys())

# Key signature options
KEY_OPTIONS = ["(Auto)"] + KEY_SIGNATURES

# Time signature options  
TIME_SIG_OPTIONS = ["(Auto)"] + TIME_SIGNATURES

# Musical style/expression options
EXPRESSION_OPTIONS = ["(None)"] + list(COMPOSITION_STYLES.keys())

# Unit note length options
UNIT_LENGTH_OPTIONS = ["(Auto)"] + list(UNIT_LENGTHS.keys())

# =============================================================================
# Load Valid Prompts
# =============================================================================

# Predefined valid combinations set - use absolute path
PROMPTS_FILE = os.path.join(SCRIPT_DIR, 'prompts.txt')
with open(PROMPTS_FILE, 'r') as f:
    prompts = f.readlines()
valid_combinations = set()
for prompt in prompts:
    prompt = prompt.strip()
    parts = prompt.split('_')
    valid_combinations.add((parts[0], parts[1], parts[2]))

# Generate available options
periods = sorted({p for p, _, _ in valid_combinations})
composers = sorted({c for _, c, _ in valid_combinations})
instruments = sorted({i for _, _, i in valid_combinations})


# =============================================================================
# UI Update Functions
# =============================================================================

def update_preset(preset_name):
    """Update sliders when a preset is selected."""
    if preset_name in STYLE_PRESETS:
        preset = STYLE_PRESETS[preset_name]
        return preset["top_k"], preset["top_p"], preset["temperature"]
    return 9, 0.9, 1.2  # Default values


def build_custom_preamble(tempo, key, time_sig, expression, unit_length):
    """Build ABC notation preamble from composition settings."""
    lines = []
    
    # Add tempo marking
    if tempo and tempo != "(Auto)":
        tempo_info = TEMPO_PRESETS.get(tempo)
        if tempo_info and isinstance(tempo_info, tuple):
            bpm = (tempo_info[0] + tempo_info[1]) // 2
            lines.append(f'Q:1/4={bpm}')
    
    # Add key signature (KEY_SIGNATURES is a list, value is used directly)
    if key and key != "(Auto)":
        lines.append(f'K:{key}')
    
    # Add time signature (TIME_SIGNATURES is a list, value is used directly)
    if time_sig and time_sig != "(Auto)":
        lines.append(f'M:{time_sig}')
    
    # Add unit note length
    if unit_length and unit_length != "(Auto)":
        unit = UNIT_LENGTHS.get(unit_length, unit_length)
        if unit:
            lines.append(f'L:{unit}')
    
    # Add expression/style as a text annotation
    if expression and expression != "(None)":
        style_value = COMPOSITION_STYLES.get(expression, expression)
        if style_value:
            lines.append(f'%%text {style_value}')
    
    return '\n'.join(lines) if lines else None

# Dynamic component updates
def update_components(period, composer):
    if not period:
        return [
            gr.Dropdown(choices=[], value=None, interactive=False),
            gr.Dropdown(choices=[], value=None, interactive=False)
        ]
    
    valid_composers = sorted({c for p, c, _ in valid_combinations if p == period})
    valid_instruments = sorted({i for p, c, i in valid_combinations if p == period and c == composer}) if composer else []
    
    return [
        gr.Dropdown(
            choices=valid_composers,
            value=composer if composer in valid_composers else None,
            interactive=True  
        ),
        gr.Dropdown(
            choices=valid_instruments,
            value=None,
            interactive=bool(valid_instruments)  
        )
    ]


class RealtimeStream(TextIOBase):
    def __init__(self, queue):
        self.queue = queue
    
    def write(self, text):
        self.queue.put(text)
        return len(text)


def save_and_convert(abc_content, period, composer, instrumentation):
    if not all([period, composer, instrumentation]):
        raise gr.Error("Please complete a valid generation first before saving")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_str = f"{period}_{composer}_{instrumentation}"
    filename_base = f"{timestamp}_{prompt_str}"
    
    # Save ABC file in script directory
    abc_filename = os.path.join(SCRIPT_DIR, f"{filename_base}.abc")
    with open(abc_filename, "w", encoding="utf-8") as f:
        f.write(abc_content)

    xml_filename = os.path.join(SCRIPT_DIR, f"{filename_base}.xml")
    abc2xml_script = os.path.join(SCRIPT_DIR, 'abc2xml.py')
    try:
        subprocess.run(
            [sys.executable, abc2xml_script, '-o', SCRIPT_DIR, abc_filename],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error"
        raise gr.Error(f"ABC to XML conversion failed: {error_msg}. Please try to generate another composition.")
    
    return f"Saved successfully: {os.path.basename(abc_filename)} -> {os.path.basename(xml_filename)}"



def generate_music_ui(period, composer, instrumentation, top_k, top_p, temperature,
                       num_variations, use_custom_prompt, custom_prompt_text,
                       tempo, key_sig, time_sig, expression, unit_length, stop_on_complete):
    """Generate music with custom sampling parameters and optional custom prompt."""
    
    # Determine if using custom prompt or standard
    if use_custom_prompt and custom_prompt_text and custom_prompt_text.strip():
        # Custom prompt mode - bypass validation
        effective_prompt = custom_prompt_text.strip()
        period_for_file = "Custom"
        composer_for_file = "Custom"
        instrumentation_for_file = "Custom"
    else:
        # Standard mode - validate combination
        if (period, composer, instrumentation) not in valid_combinations:
            raise gr.Error("Invalid prompt combination! Please re-select from the period options")
        effective_prompt = None
        period_for_file = period
        composer_for_file = composer
        instrumentation_for_file = instrumentation
    
    # Build composition preamble
    composition_preamble = build_custom_preamble(tempo, key_sig, time_sig, expression, unit_length)
    
    output_queue = queue.Queue()
    original_stdout = sys.stdout
    sys.stdout = RealtimeStream(output_queue)
    
    result_container = []
    
    def run_inference():
        try:
            if num_variations > 1:
                # Batch generation
                results = inference_batch(
                    period=period if not use_custom_prompt else None,
                    composer=composer if not use_custom_prompt else None,
                    instrumentation=instrumentation if not use_custom_prompt else None,
                    num_variations=int(num_variations),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    temperature=float(temperature),
                    custom_prompt=effective_prompt,
                    additional_preamble=composition_preamble,
                    stop_on_complete=stop_on_complete,
                )
                result_container.extend(results)
            else:
                # Single generation
                result = inference_patch(
                    period=period if not use_custom_prompt else None,
                    composer=composer if not use_custom_prompt else None,
                    instrumentation=instrumentation if not use_custom_prompt else None,
                    top_k=int(top_k),
                    top_p=float(top_p),
                    temperature=float(temperature),
                    custom_prompt=effective_prompt,
                    additional_preamble=composition_preamble,
                    stop_on_complete=stop_on_complete,
                )
                if result:
                    result_container.append(result)
        finally:
            sys.stdout = original_stdout
    
    thread = threading.Thread(target=run_inference)
    thread.start()
    
    process_output = ""
    while thread.is_alive():
        try:
            text = output_queue.get(timeout=0.1)
            process_output += text
            yield process_output, None  
        except queue.Empty:
            continue
    
    while not output_queue.empty():
        text = output_queue.get()
        process_output += text
        yield process_output, None
    
    # Format results
    if len(result_container) == 0:
        final_result = "Generation failed. Please try again."
    elif len(result_container) == 1:
        final_result = result_container[0]
    else:
        # Multiple variations - separate with headers
        final_result = ""
        for i, abc in enumerate(result_container, 1):
            final_result += f"% ===== Variation {i} =====\n{abc}\n\n"
    
    yield process_output, final_result


# =============================================================================
# Gradio UI
# =============================================================================

with gr.Blocks(title="NotaGen - Music Generation") as demo:
    gr.Markdown("## 🎵 NotaGen - Symbolic Music Generation")
    gr.Markdown("Generate classical music in ABC notation using AI. Select a period, composer, and instrumentation to get started.")
    
    with gr.Row():
        # Left column - Input controls
        with gr.Column(scale=1):
            gr.Markdown("### 🎼 Music Style")
            period_dd = gr.Dropdown(
                choices=periods,
                value=None, 
                label="Period",
                interactive=True,
                info="Select a musical period"
            )
            composer_dd = gr.Dropdown(
                choices=[],
                value=None,
                label="Composer",
                interactive=False,
                info="Select a composer from the chosen period"
            )
            instrument_dd = gr.Dropdown(
                choices=[],
                value=None,
                label="Instrumentation",
                interactive=False,
                info="Select an instrumentation"
            )
            
            gr.Markdown("### ⚙️ Generation Settings")
            
            preset_dd = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()),
                value="Balanced (Default)",
                label="Style Preset",
                info="Quick presets for different generation styles"
            )
            
            with gr.Accordion("Sampling Settings", open=False):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=9,
                    step=1,
                    label="Top-K",
                    info="Higher = more diverse vocabulary (1-50)"
                )
                top_p_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P (Nucleus)",
                    info="Higher = more diverse tokens (0.1-1.0)"
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random/creative (0.1-2.0)"
                )
            
            with gr.Accordion("🎯 Batch & Termination", open=False):
                num_variations_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=1,
                    step=1,
                    label="Number of Variations",
                    info="Generate multiple variations (1-100)"
                )
                stop_on_complete_checkbox = gr.Checkbox(
                    value=True,
                    label="Stop when piece is complete",
                    info="Auto-stop at final barline (|] or ||)"
                )
            
            with gr.Accordion("🎹 Composition Controls", open=False):
                gr.Markdown("*Optionally set musical attributes. These hints help guide generation.*")
                with gr.Row():
                    tempo_dd = gr.Dropdown(
                        choices=TEMPO_OPTIONS,
                        value="(Auto)",
                        label="Tempo",
                        info="Set tempo marking"
                    )
                    key_dd = gr.Dropdown(
                        choices=KEY_OPTIONS,
                        value="(Auto)",
                        label="Key Signature",
                        info="Set key signature"
                    )
                with gr.Row():
                    time_sig_dd = gr.Dropdown(
                        choices=TIME_SIG_OPTIONS,
                        value="(Auto)",
                        label="Time Signature",
                        info="Set time signature"
                    )
                    unit_length_dd = gr.Dropdown(
                        choices=UNIT_LENGTH_OPTIONS,
                        value="(Auto)",
                        label="Note Unit",
                        info="Default note length"
                    )
                expression_dd = gr.Dropdown(
                    choices=EXPRESSION_OPTIONS,
                    value="(None)",
                    label="Expression/Style",
                    info="Musical character"
                )
            
            with gr.Accordion("✏️ Custom Prompt", open=False):
                gr.Markdown("*Override the dropdowns with your own prompt. Use ABC notation format.*")
                use_custom_prompt_checkbox = gr.Checkbox(
                    value=False,
                    label="Use Custom Prompt",
                    info="Enable to use custom text instead of dropdowns"
                )
                custom_prompt_textbox = gr.Textbox(
                    label="Custom Prompt Lines",
                    placeholder="%Classical\n%Mozart, Wolfgang Amadeus\n%Keyboard\nM:4/4\nK:C",
                    lines=5,
                    info="Enter prompt lines (one per line). Include % for metadata."
                )
            
            generate_btn = gr.Button("🎹 Generate Music!", variant="primary", size="lg")
            
            gr.Markdown("### 📊 Generation Progress")
            process_output = gr.Textbox(
                label="",
                interactive=False,
                lines=12,
                max_lines=12,
                placeholder="Generation progress will be shown here...",
                elem_classes="process-output"
            )

        # Right column - Output
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Generated ABC Notation")
            final_output = gr.Textbox(
                label="",
                interactive=True,
                lines=20,
                placeholder="Post-processed ABC scores will be shown here...",
                elem_classes="final-output"
            )
            
            with gr.Row():
                save_btn = gr.Button("💾 Save as ABC & XML", variant="secondary")
            
            save_status = gr.Textbox(
                label="Save Status",
                interactive=False,
                visible=True,
                max_lines=2
            )
    
    # Event handlers
    period_dd.change(
        update_components,
        inputs=[period_dd, composer_dd],
        outputs=[composer_dd, instrument_dd]
    )
    composer_dd.change(
        update_components,
        inputs=[period_dd, composer_dd],
        outputs=[composer_dd, instrument_dd]
    )
    
    # Preset updates sliders
    preset_dd.change(
        update_preset,
        inputs=[preset_dd],
        outputs=[top_k_slider, top_p_slider, temperature_slider]
    )
    
    # Generate with all parameters
    generate_btn.click(
        generate_music_ui,
        inputs=[
            period_dd, composer_dd, instrument_dd, 
            top_k_slider, top_p_slider, temperature_slider,
            num_variations_slider, use_custom_prompt_checkbox, custom_prompt_textbox,
            tempo_dd, key_dd, time_sig_dd, expression_dd, unit_length_dd,
            stop_on_complete_checkbox
        ],
        outputs=[process_output, final_output]
    )
    
    save_btn.click(
        save_and_convert,
        inputs=[final_output, period_dd, composer_dd, instrument_dd],
        outputs=[save_status]
    )


css = """
.process-output {
    background-color: #f0f0f0;
    font-family: monospace;
    padding: 10px;
    border-radius: 5px;
}
.final-output {
    background-color: #ffffff;
    font-family: sans-serif;
    padding: 10px;
    border-radius: 5px;
}

.process-output textarea {
    max-height: 500px !important;
    overflow-y: auto !important;
    white-space: pre-wrap;
}

"""
css += """
button#💾-save-convert:hover {
    background-color: #ffe6e6;
}
"""

demo.css = css

if __name__ == "__main__":
    # Enable share=True via environment variable for Colab/remote access
    # Set GRADIO_SHARE=1 or COLAB=1 to enable sharing
    share = os.environ.get('GRADIO_SHARE', '').lower() in ('1', 'true', 'yes') or \
            os.environ.get('COLAB', '').lower() in ('1', 'true', 'yes') or \
            'google.colab' in sys.modules
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=share
    )