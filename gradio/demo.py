# =============================================================================
# IMPORTANT: Suppress TensorFlow/JAX warnings BEFORE any imports
# =============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['JAX_PLATFORMS'] = ''
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
from inference import inference_patch

# =============================================================================
# Style Presets
# =============================================================================

STYLE_PRESETS = {
    "Balanced (Default)": {"top_k": 9, "top_p": 0.9, "temperature": 1.2},
    "Conservative": {"top_k": 5, "top_p": 0.85, "temperature": 0.9},
    "Creative": {"top_k": 15, "top_p": 0.95, "temperature": 1.5},
    "Very Conservative": {"top_k": 3, "top_p": 0.7, "temperature": 0.7},
    "Experimental": {"top_k": 20, "top_p": 0.98, "temperature": 1.8},
}

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



def generate_music_ui(period, composer, instrumentation, top_k, top_p, temperature):
    """Generate music with custom sampling parameters."""
    if (period, composer, instrumentation) not in valid_combinations:
        raise gr.Error("Invalid prompt combination! Please re-select from the period options")
    
    output_queue = queue.Queue()
    original_stdout = sys.stdout
    sys.stdout = RealtimeStream(output_queue)
    
    result_container = []
    def run_inference():
        try:
            # Pass sampling parameters to inference
            result_container.append(
                inference_patch(
                    period, composer, instrumentation,
                    top_k=int(top_k), top_p=float(top_p), temperature=float(temperature)
                )
            )
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
    
    final_result = result_container[0] if result_container else ""
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
            
            with gr.Accordion("Advanced Settings", open=False):
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
        inputs=[period_dd, composer_dd, instrument_dd, top_k_slider, top_p_slider, temperature_slider],
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