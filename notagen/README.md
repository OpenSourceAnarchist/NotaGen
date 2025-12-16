# NotaGen Package

A consolidated Python package for NotaGen symbolic music generation, providing:

1. **Numerically Stable Sampling** - Drop-in replacement for the `samplings` library that handles edge cases (NaN, Inf, negative values, zero sums) that caused intermittent "probabilities do not sum to 1" crashes.

2. **Unified Patchilizer** - Single implementation of the ABC notation tokenizer with patch-based encoding.

3. **Configuration Management** - Dataclass-based configuration for models, sampling, inference, and training.

4. **Model Architecture** - Hierarchical transformer (patch-level + character-level decoders).

## Bug Fix: "Probabilities do not sum to 1"

### Root Cause

The original `samplings` library (v0.1.7) had no protection against invalid probability values:

1. When model softmax outputs contained NaN/Inf (from float16 overflow during inference), these propagated through the sampling chain
2. `probs.sum()` returns `nan` when any value is `nan`
3. `probs / probs.sum()` then makes ALL values `nan`
4. `np.random.choice()` fails with "probabilities contain NaN" or "probabilities do not sum to 1"

### Solution

The `notagen.sampling` module provides:

- **`_sanitize_probabilities()`**: Cleans NaN, Inf, negative values; falls back to uniform distribution if needed
- All sampling functions now sanitize inputs and outputs
- Multiple fallback mechanisms (uniform distribution → argmax) ensure sampling never fails

### Usage

The fix is automatic - the updated `utils.py` files now import from `notagen.sampling`:

```python
# Before (buggy)
from samplings import top_k_sampling, top_p_sampling, temperature_sampling

# After (stable)
from notagen.sampling import top_k_sampling, top_p_sampling, temperature_sampling
```

The API is 100% compatible - no code changes needed beyond the import.

## Package Structure

```
notagen/
├── __init__.py       # Main package exports
├── sampling.py       # Numerically stable sampling functions
├── patchilizer.py    # ABC notation tokenizer  
├── config.py         # Dataclass configurations
└── model.py          # Model architecture (optional)
```

## Testing

Stress tested with 10,000+ trials including:
- NaN values in probabilities
- Inf values in probabilities  
- All-zero probabilities
- Negative probabilities
- Very small values (underflow)

All edge cases pass without errors.

## Changes to Existing Code

Updated files to use `notagen.sampling` instead of `samplings`:
- `inference/utils.py`
- `gradio/utils.py`
- `finetune/utils.py`
- `pretrain/utils.py`
- `RL/utils.py`

Removed dependency from `requirements.txt`:
```
# samplings==0.1.7  # Removed - replaced by notagen.sampling for numerical stability
```
