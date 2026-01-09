"""
vLLM OpenAI-Compatible API Server for any vLLM-supported model

This script starts an OpenAI-compatible API server using vLLM.
The server will be accessible at http://localhost:8000
Configuration is loaded from config.yaml
"""
import os
import subprocess
import sys
from utils import load_config

def start_api_server():
    """Start the vLLM OpenAI-compatible API server."""

    # Load configuration
    config = load_config()
    model_config = config['model_information']['model_config']
    vllm_config = config['model_information']['vllm_engine_config']

    # Handle None vllm_config by using empty dict
    if vllm_config is None:
        vllm_config = {}

    # Set PyTorch CUDA memory allocator configuration if specified
    if "pytorch_cuda_alloc_conf" in vllm_config:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = vllm_config["pytorch_cuda_alloc_conf"]

    # Determine which model to use based on is_model_local flag
    is_model_local = model_config.get("is_model_local", False)
    if is_model_local:
        model_to_use = model_config.get("model_path", model_config["model_id"])
    else:
        model_to_use = model_config["model_id"]

    # Build the vLLM server command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_to_use,
        "--dtype", model_config["dtype"],
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    if model_config.get("trust_remote_code", False):
        cmd.append("--trust-remote-code")

    if vllm_config.get("enforce_eager", False):
        cmd.append("--enforce-eager")

    # Add vLLM engine parameters
    if "max_model_len" in vllm_config:
        cmd.extend(["--max-model-len", str(vllm_config["max_model_len"])])

    if "kv_cache_dtype" in vllm_config:
        cmd.extend(["--kv-cache-dtype", vllm_config["kv_cache_dtype"]])

    if "gpu_memory_utilization" in vllm_config:
        cmd.extend(["--gpu-memory-utilization", str(vllm_config["gpu_memory_utilization"])])

    if "tensor_parallel_size" in vllm_config:
        cmd.extend(["--tensor-parallel-size", str(vllm_config["tensor_parallel_size"])])

    if "tool_call_parser" in vllm_config:
        cmd.extend(["--tool-call-parser", vllm_config["tool_call_parser"]])

    if vllm_config.get("enable_auto_tool_choice", False):
        cmd.append("--enable-auto-tool-choice")

    # LoRA adapter support
    if vllm_config.get("enable_lora", False):
        cmd.append("--enable-lora")

        if "lora_modules" in vllm_config and vllm_config["lora_modules"]:
            for lora_name, lora_path in vllm_config["lora_modules"].items():
                cmd.extend(["--lora-modules", f"{lora_name}={lora_path}"])

        if "max_loras" in vllm_config:
            cmd.extend(["--max-loras", str(vllm_config["max_loras"])])

        if "max_lora_rank" in vllm_config:
            cmd.extend(["--max-lora-rank", str(vllm_config["max_lora_rank"])])

    print("="*60)
    print("Starting vLLM OpenAI-Compatible API Server")
    print("="*60)
    print(f"Model: {model_to_use}")
    print(f"Model source: {'Local path' if is_model_local else 'HuggingFace Hub'}")
    print(f"Max model length: {vllm_config.get('max_model_len', 8192)}")
    print(f"GPU memory utilization: {vllm_config.get('gpu_memory_utilization', 0.85)}")
    print(f"\nServer will be available at: http://localhost:8000")
    print(f"API docs at: http://localhost:8000/docs")
    print(f"\nCommand: {' '.join(cmd)}")
    print("="*60)
    print("\nStarting server... (Press Ctrl+C to stop)\n")

    # Run the server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_api_server()
