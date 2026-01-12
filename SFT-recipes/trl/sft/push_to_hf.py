"""
Push LoRA adapter or merged model to Hugging Face Hub

This script can:
1. Push a LoRA adapter directly to HuggingFace
2. Merge LoRA adapter with base model and push the merged model
3. Push an already merged model

Configuration is loaded from ../../vLLM/config.yaml
"""
import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi, login


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


def merge_and_push(
    base_model: str,
    adapter_path: str,
    output_path: str,
    hf_repo_id: str,
    push_merged: bool = True,
    push_adapter: bool = False,
    hf_token: str = None
):
    """
    Merge LoRA adapter with base model and optionally push to HuggingFace.

    Args:
        base_model: HuggingFace model ID or local path to base model
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model locally
        hf_repo_id: HuggingFace repository ID (e.g., "username/model-name")
        push_merged: Whether to push the merged model to HuggingFace
        push_adapter: Whether to push the adapter to HuggingFace
        hf_token: HuggingFace API token (optional, will use cached token if not provided)
    """
    # Login to HuggingFace if token is provided
    if hf_token:
        login(token=hf_token)

    print("="*60)
    print("LoRA Adapter Merge and Push to HuggingFace")
    print("="*60)
    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    print(f"HuggingFace repo: {hf_repo_id}")
    print(f"Push merged model: {push_merged}")
    print(f"Push adapter only: {push_adapter}")
    print("="*60)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # If only pushing adapter, skip merge
    if push_adapter and not push_merged:
        print("\nPushing adapter only (no merge)...")
        print(f"Uploading adapter from {adapter_path} to {hf_repo_id}")

        api = HfApi()
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=hf_repo_id,
            repo_type="model",
        )
        print(f"\n✅ Adapter uploaded to: https://huggingface.co/{hf_repo_id}")
        return

    # Load tokenizer
    print(f"\nLoading tokenizer from {base_model}...")
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

    # Save merged model locally
    print(f"\nSaving merged model to {output_path}...")
    merged.save_pretrained(output_path, safe_serialization=True)
    print(f"✅ Merged model saved locally to: {output_path}")

    # Push to HuggingFace if requested
    if push_merged:
        print(f"\nPushing merged model to HuggingFace: {hf_repo_id}")
        merged.push_to_hub(hf_repo_id, use_temp_dir=False)
        tokenizer.push_to_hub(hf_repo_id, use_temp_dir=False)
        print(f"\n✅ Merged model pushed to: https://huggingface.co/{hf_repo_id}")

    print("\n" + "="*60)
    print("Process completed successfully!")
    print("="*60)


def push_existing_model(model_path: str, hf_repo_id: str, hf_token: str = None):
    """
    Push an existing model directory to HuggingFace Hub.

    Args:
        model_path: Path to the model directory
        hf_repo_id: HuggingFace repository ID
        hf_token: HuggingFace API token
    """
    if hf_token:
        login(token=hf_token)

    print("="*60)
    print("Pushing Existing Model to HuggingFace")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"HuggingFace repo: {hf_repo_id}")
    print("="*60)

    # Check if path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Upload to HuggingFace
    print(f"\nUploading model to {hf_repo_id}...")
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_repo_id,
        repo_type="model",
    )
    print(f"\n✅ Model uploaded to: https://huggingface.co/{hf_repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push LoRA adapter or merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ../../vLLM/config.yaml)"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/model-name)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, will use cached token if not provided)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["merge", "adapter", "existing"],
        default="merge",
        help="Mode: 'merge' (merge and push), 'adapter' (push adapter only), 'existing' (push existing merged model)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to existing model (used with --mode existing)"
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

    if args.mode == "existing":
        # Push existing model
        model_path = args.model_path or output_path
        if not model_path:
            print("Error: --model-path is required for 'existing' mode")
            sys.exit(1)
        push_existing_model(model_path, args.hf_repo_id, args.hf_token)

    elif args.mode == "adapter":
        # Push adapter only
        if not adapter_path:
            print("Error: adapter_path not found in config or --adapter-path not provided")
            sys.exit(1)
        merge_and_push(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            hf_repo_id=args.hf_repo_id,
            push_merged=False,
            push_adapter=True,
            hf_token=args.hf_token
        )

    else:  # merge mode
        # Merge and push
        if not all([base_model, adapter_path, output_path]):
            print("Error: Missing required configuration")
            print(f"Base model: {base_model}")
            print(f"Adapter path: {adapter_path}")
            print(f"Output path: {output_path}")
            sys.exit(1)

        merge_and_push(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            hf_repo_id=args.hf_repo_id,
            push_merged=True,
            push_adapter=False,
            hf_token=args.hf_token
        )


if __name__ == "__main__":
    main()
