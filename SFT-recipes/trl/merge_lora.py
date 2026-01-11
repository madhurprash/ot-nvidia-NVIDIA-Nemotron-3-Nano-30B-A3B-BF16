"""
Generic LoRA Adapter Merge Script

This script merges a LoRA adapter with its base model.
Configuration is loaded from ../../vLLM/config.yaml

Usage:
    python merge_lora.py [--config /path/to/config.yaml]
"""
import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_config(config_path: str = None) -> dict:
    """Load configuration from config.yaml file."""
    if config_path is None:
        # Default to the vLLM config.yaml
        script_dir = Path(__file__).parent.absolute()
        config_path = script_dir / "../../vLLM/config.yaml"

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def merge_lora_adapter(
    base_model: str,
    adapter_path: str,
    output_path: str
):
    """
    Merge LoRA adapter with base model.

    Args:
        base_model: HuggingFace model ID or local path to base model
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
    """
    print("="*60)
    print("LoRA Adapter Merge Script")
    print("="*60)
    print(f"Base model: {base_model}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Output directory: {output_path}")
    print("="*60)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print(f"Saving tokenizer to {output_path}...")
    tokenizer.save_pretrained(output_path)

    # Load base model in bfloat16
    print(f"\nLoading base model in bfloat16...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Base model loaded.")

    # Wrap with PEFT adapter
    print(f"\nWrapping base model with PEFT LoRA from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    # Move to bfloat16 for merging
    print("Moving PEFT model to bfloat16 for merging...")
    peft_model = peft_model.to(torch.bfloat16)

    # Merge
    print("\nMerging LoRA weights into base model...")
    merged = peft_model.merge_and_unload()

    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    merged.save_pretrained(output_path, safe_serialization=True)
    print(f"✅ Merged model saved to: {output_path}")
    print("\n" + "="*60)
    print("Merge completed successfully!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ../../vLLM/config.yaml)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model from config"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override adapter path from config"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override output path from config"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    lora_config = config.get('lora_merge', {})

    # Get parameters from config or arguments
    base_model = args.base_model or lora_config.get('base_model')
    adapter_path = args.adapter_path or lora_config.get('adapter_path')
    output_path = args.output_path or lora_config.get('output_path')

    # Check required parameters
    if not all([base_model, adapter_path, output_path]):
        print("Error: Missing required configuration")
        print(f"Base model: {base_model}")
        print(f"Adapter path: {adapter_path}")
        print(f"Output path: {output_path}")
        sys.exit(1)

    # Merge the adapter
    merge_lora_adapter(base_model, adapter_path, output_path)


if __name__ == "__main__":
    main()
