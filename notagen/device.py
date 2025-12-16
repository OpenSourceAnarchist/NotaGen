"""
Device Detection and Management for NotaGen

Provides unified device detection across all NotaGen modules,
supporting CUDA, MPS (Apple Silicon), and CPU backends.
"""

from __future__ import annotations
import os
import torch
from typing import Optional
import warnings


def get_device(
    preferred: Optional[str] = None,
    allow_mps: bool = True,
    verbose: bool = False
) -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Priority order:
    1. User-specified preferred device (if available)
    2. CUDA (NVIDIA GPUs)
    3. MPS (Apple Silicon) - if allow_mps is True
    4. CPU (fallback)
    
    Args:
        preferred: Preferred device string ('cuda', 'mps', 'cpu', or 'cuda:0', etc.)
        allow_mps: Whether to use MPS backend on Apple Silicon
        verbose: Print device selection information
        
    Returns:
        torch.device for the selected backend
        
    Examples:
        >>> device = get_device()  # Auto-detect best device
        >>> device = get_device('cpu')  # Force CPU
        >>> device = get_device('cuda:1')  # Specific GPU
    """
    # If user specified a device, try to use it
    if preferred is not None:
        device_str = preferred.lower()
        
        if device_str.startswith('cuda'):
            if torch.cuda.is_available():
                device = torch.device(preferred)
                if verbose:
                    print(f"Using requested CUDA device: {device}")
                return device
            else:
                warnings.warn(f"CUDA requested but not available, falling back to auto-detection")
        elif device_str == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                if verbose:
                    print("Using requested MPS device (Apple Silicon)")
                return device
            else:
                warnings.warn("MPS requested but not available, falling back to auto-detection")
        elif device_str == 'cpu':
            if verbose:
                print("Using requested CPU device")
            return torch.device('cpu')
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA device: {gpu_name}")
        return device
    
    if allow_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print("Using MPS device (Apple Silicon)")
        return device
    
    if verbose:
        print("Using CPU device (no GPU acceleration available)")
    return torch.device('cpu')


def get_autocast_device_type(device: torch.device) -> str:
    """
    Get the device type string for torch.autocast().
    
    Args:
        device: The torch device
        
    Returns:
        Device type string compatible with torch.autocast
    """
    device_type = str(device).split(':')[0]
    
    # MPS doesn't support autocast as of PyTorch 2.0, fall back to no-op
    if device_type == 'mps':
        return 'cpu'  # Will effectively disable autocast
    
    return device_type


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or MPS) is available."""
    return torch.cuda.is_available() or (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    )


def get_device_info() -> dict:
    """
    Get comprehensive device information.
    
    Returns:
        Dictionary with device availability and details
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'selected_device': str(get_device()),
    }
    
    if info['cuda_available']:
        info['cuda_devices'] = [
            {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info


def set_seed(seed: int, device: Optional[torch.device] = None):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        device: Device to set seed for (auto-detected if None)
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device is None:
        device = get_device(verbose=False)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == 'mps':
        # MPS-specific seeding if needed in future PyTorch versions
        pass


def empty_cache(device: Optional[torch.device] = None):
    """
    Clear memory cache for the given device.
    
    Args:
        device: Device to clear cache for (auto-detected if None)
    """
    if device is None:
        device = get_device(verbose=False)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # MPS cache clearing if available in future PyTorch versions
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
