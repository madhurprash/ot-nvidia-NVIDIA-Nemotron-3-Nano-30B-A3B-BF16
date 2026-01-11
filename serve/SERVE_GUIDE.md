# vLLM Serving Guide with LoRA Adapters

This guide explains how to serve models with LoRA adapters using vLLM's OpenAI-compatible API server.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Starting the Server](#starting-the-server)
- [Making Requests](#making-requests)
- [Dynamic Adapter Management](#dynamic-adapter-management)
- [Troubleshooting](#troubleshooting)

## Quick Start

1. Configure your model and LoRA adapters in `config.yaml`
2. Start the server: `python serve.py`
3. Make requests to `http://localhost:8000`

## Configuration

### Base Model Configuration

Edit `config.yaml` to configure your base model:

```yaml
model_information:
  model_config:
    # Use local model or HuggingFace repo
    is_model_local: false
    model_id: "mistralai/Devstral-Small-2-24B-Instruct-2512"
    # If is_model_local is true, use this path
    model_path: "/path/to/local/model"
    trust_remote_code: true
    dtype: "auto"
```

### LoRA Adapter Configuration

Configure LoRA adapters in the `vllm_engine_config` section:

```yaml
vllm_engine_config:
  # Enable LoRA support
  enable_lora: true

  # Define LoRA modules
  lora_modules:
    # Option 1: Local path
    my-adapter-local: "/path/to/local/adapter"

    # Option 2: HuggingFace repository
    my-adapter-hf: "username/repo-name"

    # Example: Your pushed LoRA adapter
    devstral-sft: "Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts"

  # Maximum number of LoRA adapters that can be used concurrently
  max_loras: 2

  # Maximum rank of LoRA adapters (set to match or exceed your adapter's rank)
  max_lora_rank: 16
```

**Important Notes:**
- `max_lora_rank`: Set this to at least the rank of your LoRA adapter. Check your adapter config for the rank value.
- `max_loras`: Number of adapters that can be active simultaneously. Increase if you need more concurrent adapters.
- LoRA modules can be specified as local paths or HuggingFace repository IDs (they will be auto-downloaded).

### Memory and Performance Settings

```yaml
vllm_engine_config:
  # Context window size
  max_model_len: 32768

  # GPU memory usage (0.0 to 1.0)
  gpu_memory_utilization: 0.9

  # Multi-GPU support
  tensor_parallel_size: 8
```

## Starting the Server

Run the server with:

```bash
python serve.py
```

The server will:
1. Load the base model
2. Register all configured LoRA adapters
3. Start listening on `http://0.0.0.0:8000`

You'll see output like:
```
============================================================
Starting vLLM OpenAI-Compatible API Server
============================================================
Model: mistralai/Devstral-Small-2-24B-Instruct-2512
Model source: HuggingFace Hub
Max model length: 32768
GPU memory utilization: 0.9

Server will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs
============================================================
```

## Making Requests

### Using the Base Model

To use the base model (without LoRA adapter):

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "prompt": "def fibonacci(n):",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Using a LoRA Adapter

To use a specific LoRA adapter, set the `model` parameter to the adapter name from your config:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "devstral-sft",
    "prompt": "def fibonacci(n):",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Chat Completions with LoRA

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "devstral-sft",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    "max_tokens": 512,
    "temperature": 0.6
  }'
```

### Using Python OpenAI Client

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require authentication
)

# Use LoRA adapter
response = client.chat.completions.create(
    model="devstral-sft",  # Your LoRA adapter name
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain async/await in Python"}
    ],
    max_tokens=1024,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Dynamic Adapter Management

vLLM supports loading and unloading LoRA adapters at runtime without restarting the server.

### Enable Dynamic Loading

Set the environment variable before starting the server:

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
python serve.py
```

### Load a New Adapter

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "new-adapter",
    "lora_path": "username/repo-name"
  }'
```

Or with a local path:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "new-adapter",
    "lora_path": "/path/to/local/adapter"
  }'
```

### Unload an Adapter

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "adapter-to-remove"
  }'
```

### Python Example for Dynamic Loading

```python
import requests

BASE_URL = "http://localhost:8000"

# Load adapter
load_response = requests.post(
    f"{BASE_URL}/v1/load_lora_adapter",
    json={
        "lora_name": "my-new-adapter",
        "lora_path": "Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts"
    }
)
print(f"Load status: {load_response.status_code}")

# Use the adapter
# ... make inference requests with model="my-new-adapter" ...

# Unload adapter when done
unload_response = requests.post(
    f"{BASE_URL}/v1/unload_lora_adapter",
    json={"lora_name": "my-new-adapter"}
)
print(f"Unload status: {unload_response.status_code}")
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce `gpu_memory_utilization` in config.yaml:
   ```yaml
   gpu_memory_utilization: 0.85  # Try lower values like 0.8, 0.75
   ```

2. Reduce `max_model_len`:
   ```yaml
   max_model_len: 16384  # Or 8192 for very constrained memory
   ```

3. Reduce `max_loras` to load fewer adapters concurrently:
   ```yaml
   max_loras: 1
   ```

### Adapter Not Found

If you get "model not found" errors:

1. Check that the adapter name in your request matches the name in `lora_modules`
2. Verify the adapter path/repo exists and is accessible
3. For HuggingFace repos, ensure you have access and are authenticated if needed:
   ```bash
   huggingface-cli login
   ```

### Slow First Request

The first request after starting the server may be slow as the model warms up. Subsequent requests will be faster.

### Checking LoRA Rank

To find your LoRA adapter's rank, check `adapter_config.json` in your adapter directory:

```bash
cat /path/to/adapter/adapter_config.json
```

Look for the `r` or `rank` field and ensure `max_lora_rank` in your config is set to at least this value.

## API Documentation

Once the server is running, you can access interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Workflow

Here's a complete workflow for serving a fine-tuned model:

1. **Train and save your LoRA adapter** (or download from HuggingFace)

2. **Update config.yaml**:
   ```yaml
   model_config:
     is_model_local: false
     model_id: "mistralai/Devstral-Small-2-24B-Instruct-2512"

   vllm_engine_config:
     enable_lora: true
     lora_modules:
       my-finetuned-model: "Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts"
     max_loras: 1
     max_lora_rank: 16
   ```

3. **Start the server**:
   ```bash
   python serve.py
   ```

4. **Make requests**:
   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

   response = client.chat.completions.create(
       model="my-finetuned-model",
       messages=[{"role": "user", "content": "Hello!"}],
       max_tokens=256
   )

   print(response.choices[0].message.content)
   ```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM LoRA Support](https://docs.vllm.ai/en/stable/features/lora/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
