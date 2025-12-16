#!/usr/bin/env python
"""
NotaGen Command Line Interface

Usage:
    python -m notagen generate --period Romantic --composer Chopin
    python -m notagen generate --period Baroque --composer Bach --output output.abc
    python -m notagen generate --period Classical --composer Mozart --format musicxml
    python -m notagen batch --input prompts.txt --output-dir ./generated
    python -m notagen info  # Show available periods, composers, config
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, TextIO

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_generate(args: argparse.Namespace) -> int:
    """Handle the generate command."""
    from notagen import load_notagen, generate_music, print_config
    from notagen.export import save_output
    from notagen.generate import TqdmProgressCallback
    
    # Show config if verbose
    if args.verbose:
        print_config()
        print()
    
    # Load model
    print(f"Loading model from: {args.weights or 'auto-detect'}...")
    model, patchilizer = load_notagen(
        weights_path=args.weights,
        device=args.device,
        half_precision=not args.fp32,
    )
    
    # Setup progress callback
    callback = None
    if args.progress:
        callback = TqdmProgressCallback(desc="Generating")
    
    # Generate
    print(f"\nGenerating: {args.period} / {args.composer}" + 
          (f" / {args.instrumentation}" if args.instrumentation else ""))
    print("-" * 50)
    
    abc_notation = generate_music(
        model=model,
        patchilizer=patchilizer,
        period=args.period,
        composer=args.composer,
        instrumentation=args.instrumentation,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_patches=args.max_patches,
        max_time_seconds=args.max_time,
        callback=callback,
    )
    
    # Output
    if args.output:
        output_path = save_output(
            abc_notation,
            output_path=args.output,
            format=args.format,
            period=args.period,
            composer=args.composer,
            instrumentation=args.instrumentation,
        )
        print(f"\nSaved to: {output_path}")
    else:
        print("\n" + "=" * 50)
        print("Generated ABC Notation:")
        print("=" * 50)
        print(abc_notation)
    
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Handle the batch command."""
    from notagen import load_notagen
    from notagen.generate import batch_generate, TqdmProgressCallback
    from notagen.export import save_output
    
    # Parse prompts from input file
    prompts: List[dict] = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse "Period_Composer_Instrumentation" format
            parts = line.split('_')
            if len(parts) >= 2:
                prompt = {
                    'period': parts[0],
                    'composer': parts[1],
                    'instrumentation': parts[2] if len(parts) > 2 else None,
                }
                prompts.append(prompt)
    
    if not prompts:
        print("Error: No valid prompts found in input file")
        return 1
    
    print(f"Found {len(prompts)} prompts to generate")
    
    # Load model
    print(f"Loading model...")
    model, patchilizer = load_notagen(
        weights_path=args.weights,
        device=args.device,
        half_precision=not args.fp32,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all
    results = batch_generate(
        model=model,
        patchilizer=patchilizer,
        prompts=prompts,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_patches=args.max_patches,
        max_time_seconds=args.max_time,
        show_progress=True,
    )
    
    # Save results
    for i, (prompt, abc) in enumerate(zip(prompts, results)):
        filename = f"{i+1:03d}_{prompt['period']}_{prompt['composer']}"
        if prompt.get('instrumentation'):
            filename += f"_{prompt['instrumentation']}"
        
        output_path = output_dir / f"{filename}.{args.format}"
        save_output(
            abc,
            output_path=str(output_path),
            format=args.format,
            **prompt,
        )
        print(f"  Saved: {output_path}")
    
    print(f"\nBatch complete: {len(results)} pieces generated")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show available options and current configuration."""
    from notagen import print_config, find_weights
    from notagen.config import SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG
    
    print("=" * 60)
    print("NotaGen - Symbolic Music Generation")
    print("=" * 60)
    
    # Show weights
    weights = find_weights()
    if weights:
        print(f"\nFound weights file(s):")
        for w in weights[:5]:
            print(f"  - {w}")
    else:
        print("\nNo weights file found. Please download model weights.")
    
    print("\n" + "-" * 60)
    print("Model Size Presets:")
    print("-" * 60)
    print(f"  small:  {SMALL_CONFIG.patch_num_layers} patch layers, {SMALL_CONFIG.char_num_layers} char layers, {SMALL_CONFIG.hidden_size} hidden (~110M params)")
    print(f"  medium: {MEDIUM_CONFIG.patch_num_layers} patch layers, {MEDIUM_CONFIG.char_num_layers} char layers, {MEDIUM_CONFIG.hidden_size} hidden (~244M params)")
    print(f"  large:  {LARGE_CONFIG.patch_num_layers} patch layers, {LARGE_CONFIG.char_num_layers} char layers, {LARGE_CONFIG.hidden_size} hidden (~516M params)")
    
    print("\n" + "-" * 60)
    print("Current Configuration:")
    print("-" * 60)
    print_config()
    
    print("\n" + "-" * 60)
    print("Example Usage:")
    print("-" * 60)
    print("  python -m notagen generate --period Romantic --composer Chopin")
    print("  python -m notagen generate --period Baroque --composer Bach --format musicxml")
    print("  python -m notagen batch --input prompts.txt --output-dir ./output")
    
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert ABC file to other formats."""
    from notagen.export import abc_to_musicxml, abc_to_midi
    
    # Read input
    with open(args.input, 'r') as f:
        abc_content = f.read()
    
    output_path = args.output or args.input.rsplit('.', 1)[0]
    
    if args.format in ('xml', 'musicxml'):
        result = abc_to_musicxml(abc_content, output_path + '.xml')
        print(f"Converted to MusicXML: {result}")
    elif args.format == 'midi':
        result = abc_to_midi(abc_content, output_path + '.mid')
        print(f"Converted to MIDI: {result}")
    else:
        print(f"Unknown format: {args.format}")
        return 1
    
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog='notagen',
        description='NotaGen - Symbolic Music Generation with LLM Training Paradigms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m notagen generate --period Romantic --composer Chopin
  python -m notagen generate --period Baroque --composer Bach --output bach.abc
  python -m notagen generate --period Classical --composer Mozart --format musicxml
  python -m notagen batch --input prompts.txt --output-dir ./generated
  python -m notagen convert --input piece.abc --format midi
  python -m notagen info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # =========================================================================
    # Generate command
    # =========================================================================
    gen_parser = subparsers.add_parser('generate', help='Generate a single piece')
    gen_parser.add_argument('--period', '-p', required=True,
                           help='Musical period (e.g., Romantic, Baroque, Classical)')
    gen_parser.add_argument('--composer', '-c', required=True,
                           help='Composer name (e.g., Chopin, Bach, Mozart)')
    gen_parser.add_argument('--instrumentation', '-i',
                           help='Instrumentation (e.g., Piano Sonata, String Quartet)')
    gen_parser.add_argument('--output', '-o',
                           help='Output file path (prints to stdout if not specified)')
    gen_parser.add_argument('--format', '-f', default='abc',
                           choices=['abc', 'xml', 'musicxml', 'midi'],
                           help='Output format (default: abc)')
    gen_parser.add_argument('--weights', '-w',
                           help='Path to model weights (auto-detects if not specified)')
    gen_parser.add_argument('--device', '-d',
                           choices=['cuda', 'mps', 'cpu'],
                           help='Device to use (auto-detects if not specified)')
    gen_parser.add_argument('--top-k', type=int, default=9,
                           help='Top-k sampling parameter (default: 9)')
    gen_parser.add_argument('--top-p', type=float, default=0.9,
                           help='Top-p (nucleus) sampling parameter (default: 0.9)')
    gen_parser.add_argument('--temperature', type=float, default=1.2,
                           help='Temperature for sampling (default: 1.2)')
    gen_parser.add_argument('--max-patches', type=int, default=128,
                           help='Maximum patches to generate (default: 128)')
    gen_parser.add_argument('--max-time', type=int, default=300,
                           help='Maximum generation time in seconds (default: 300)')
    gen_parser.add_argument('--fp32', action='store_true',
                           help='Use FP32 instead of FP16')
    gen_parser.add_argument('--progress', action='store_true',
                           help='Show progress bar')
    gen_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')
    gen_parser.set_defaults(func=cmd_generate)
    
    # =========================================================================
    # Batch command
    # =========================================================================
    batch_parser = subparsers.add_parser('batch', help='Generate multiple pieces')
    batch_parser.add_argument('--input', '-i', required=True,
                             help='Input file with prompts (one per line: Period_Composer_Instrumentation)')
    batch_parser.add_argument('--output-dir', '-o', default='./generated',
                             help='Output directory (default: ./generated)')
    batch_parser.add_argument('--format', '-f', default='abc',
                             choices=['abc', 'xml', 'musicxml', 'midi'],
                             help='Output format (default: abc)')
    batch_parser.add_argument('--weights', '-w',
                             help='Path to model weights')
    batch_parser.add_argument('--device', '-d',
                             choices=['cuda', 'mps', 'cpu'],
                             help='Device to use')
    batch_parser.add_argument('--top-k', type=int, default=9,
                             help='Top-k sampling parameter')
    batch_parser.add_argument('--top-p', type=float, default=0.9,
                             help='Top-p sampling parameter')
    batch_parser.add_argument('--temperature', type=float, default=1.2,
                             help='Temperature for sampling')
    batch_parser.add_argument('--max-patches', type=int, default=128,
                             help='Maximum patches per piece')
    batch_parser.add_argument('--max-time', type=int, default=300,
                             help='Maximum time per piece in seconds')
    batch_parser.add_argument('--fp32', action='store_true',
                             help='Use FP32 instead of FP16')
    batch_parser.set_defaults(func=cmd_batch)
    
    # =========================================================================
    # Convert command
    # =========================================================================
    conv_parser = subparsers.add_parser('convert', help='Convert ABC to other formats')
    conv_parser.add_argument('--input', '-i', required=True,
                            help='Input ABC file')
    conv_parser.add_argument('--output', '-o',
                            help='Output file (default: same name as input)')
    conv_parser.add_argument('--format', '-f', required=True,
                            choices=['xml', 'musicxml', 'midi'],
                            help='Output format')
    conv_parser.set_defaults(func=cmd_convert)
    
    # =========================================================================
    # Info command
    # =========================================================================
    info_parser = subparsers.add_parser('info', help='Show configuration and available options')
    info_parser.set_defaults(func=cmd_info)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
