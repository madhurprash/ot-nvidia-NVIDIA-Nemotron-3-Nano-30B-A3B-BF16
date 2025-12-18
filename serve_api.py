"""
vLLM OpenAI-Compatible API Server for NVIDIA Nemotron-3-Nano-30B-A3B-BF16

This script starts an OpenAI-compatible API server using vLLM.
The server will be accessible at http://localhost:8000
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

    # Set PyTorch CUDA memory allocator configuration if specified
    if "pytorch_cuda_alloc_conf" in vllm_config:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = vllm_config["pytorch_cuda_alloc_conf"]

    # Build the vLLM server command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_config["model_id"],
        "--dtype", model_config["dtype"],
        "--max-model-len", str(vllm_config.get("max_model_len", 8192)),
        "--gpu-memory-utilization", str(vllm_config.get("gpu_memory_utilization", 0.85)),
        "--swap-space", str(vllm_config.get("swap_space", 4)),
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    if model_config.get("trust_remote_code", False):
        cmd.append("--trust-remote-code")

    if vllm_config.get("enforce_eager", False):
        cmd.append("--enforce-eager")

    print("="*60)
    print("Starting vLLM OpenAI-Compatible API Server")
    print("="*60)
    print(f"Model: {model_config['model_id']}")
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
