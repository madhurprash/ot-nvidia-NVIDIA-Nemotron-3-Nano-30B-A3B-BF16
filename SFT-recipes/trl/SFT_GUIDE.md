# Supervised Fine-Tuning (SFT) Guide

This guide covers the complete workflow for fine-tuning models with LoRA adapters, merging them with base models, and deploying them using vLLM. This repository is tested on a `Devstral-Small-2-24B-Instruct-2512` model which is Supervised Fine-tuned using the `open-thoughts/OpenThoughts-Agent-v1-SFT` dataset. The `LoRA` adapter is then directly pushed under `Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts` [here](https://huggingface.co/Madhurprash/Devstral-Small-2-24B-Instruct-2512-SFT-LoRA-OpenThoughts). This adapter can then be directly merged into the base model and tested on the Terminal Bench 2.0 benchmark.

## Table of Contents

1. [Fine-tuning with LoRA](#fine-tuning-with-lora)
2. [Merging LoRA Adapters](#merging-lora-adapters)
3. [Pushing Models to HuggingFace](#pushing-models-to-huggingface)
4. [Serving with vLLM](#serving-with-vllm)
5. [Configuration](#configuration)

---

## Fine-tuning with LoRA

### Prerequisites

- Base model (e.g., Devstral, Mistral, Llama)
- Training dataset prepared
- Sufficient GPU memory
- Python environment with required packages

### Training Process

1. Prepare your training data in the required format
2. Configure your training parameters
3. Run the training script
4. Monitor training progress

The LoRA adapter will be saved to the output directory specified in your training configuration.

**Note:** Specific training scripts should be configured based on your model architecture and dataset requirements.

---

## Merging LoRA Adapters

After training, you have two options:

### Option 1: Merge LoRA Adapter with Base Model

Use the generic merge script that loads configuration from `vLLM/config.yaml`:

```bash
python merge_lora.py
```

**With custom configuration:**

```bash
python merge_lora.py --config /path/to/config.yaml
```

**Override specific parameters:**

```bash
python merge_lora.py \
  --base-model "mistralai/Devstral-Small-2-24B-Instruct-2512" \
  --adapter-path "./outputs/devstral-sft" \
  --output-path "./outputs/merged-devstral-sft"
```

### Option 2: Use Mistral-Specific Merge Script

For Mistral models specifically:

```bash
python merge_mistral_lora.py
```

This script is hardcoded for Mistral3ForConditionalGeneration and uses the original configuration.

---

## Pushing Models to HuggingFace

The `push_to_hf.py` script provides three modes for uploading to HuggingFace Hub:

### Mode 1: Merge and Push (Default)

Merge the LoRA adapter with the base model and push the merged model:

```bash
python push_to_hf.py \
  --hf-repo-id "your-username/your-model-name" \
  --mode merge
```

**With HuggingFace token:**

```bash
python push_to_hf.py \
  --hf-repo-id "your-username/your-model-name" \
  --hf-token "your_hf_token_here" \
  --mode merge
```

### Mode 2: Push Adapter Only

Push only the LoRA adapter without merging:

```bash
python push_to_hf.py --hf-repo-id yourusername/repo-id --mode adapter --adapter-path adapter-path --hf-token your-hf-token
```

### Mode 3: Push Existing Merged Model

Push an already merged model:

```bash
python push_to_hf.py \
  --hf-repo-id "your-username/your-model-name" \
  --mode existing \
  --model-path "/path/to/merged/model"
```

### Authentication

You can provide your HuggingFace token in three ways:

1. **As an argument:** `--hf-token "your_token"`
2. **Cached login:** Run `huggingface-cli login` beforehand
3. **Environment variable:** Set `HF_TOKEN` in your environment

---

## Serving with vLLM

The vLLM server supports serving merged models or using LoRA adapters dynamically.

### Start the Server

Navigate to the vLLM directory and run:

```bash
cd ../../vLLM
python serve.py
```

The server will:
- Read configuration from `config.yaml`
- Start on `http://localhost:8000`
- Provide OpenAI-compatible API endpoints

### Configuration Options

Edit `vLLM/config.yaml` to configure:

#### For Merged Models:

```yaml
model_information:
  model_config:
    is_model_local: true
    model_path: "/path/to/merged/model"
  vllm_engine_config:
    enable_lora: false
```

#### For LoRA Adapters:

```yaml
model_information:
  model_config:
    is_model_local: false
    model_id: "mistralai/Devstral-Small-2-24B-Instruct-2512"
  vllm_engine_config:
    enable_lora: true
    lora_modules:
      devstral-sft: "/path/to/adapter"
    max_loras: 1
    max_lora_rank: 8
```

### API Usage

Once the server is running, you can use it like any OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="devstral-sft",  # or your model name
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
```

---

## Configuration

### Main Configuration File: `vLLM/config.yaml`

```yaml
general:
  name: "agentic-SLM-vllm-deployment"
  description: "vLLM deployment configuration"

model_information:
  model_config:
    is_model_local: false
    model_id: "your-model-id"
    model_path: "/path/to/local/model"
    trust_remote_code: true
    dtype: "auto"

  vllm_engine_config:
    max_model_len: 32768
    tensor_parallel_size: 8
    tool_call_parser: "mistral"
    enable_auto_tool_choice: true
    enable_lora: false
    lora_modules:
      adapter-name: "/path/to/adapter"
    max_loras: 1
    max_lora_rank: 8

  inference_parameters:
    temperature: 0.6
    max_tokens: 8192

lora_merge:
  base_model: "mistralai/Devstral-Small-2-24B-Instruct-2512"
  adapter_path: "/path/to/adapter"
  output_path: "/path/to/output"
```

### Key Configuration Parameters

- **is_model_local**: Set to `true` to load from local path, `false` for HuggingFace Hub
- **model_id**: HuggingFace model ID (when `is_model_local: false`)
- **model_path**: Local path to model (when `is_model_local: true`)
- **enable_lora**: Set to `true` to enable dynamic LoRA adapter loading
- **lora_modules**: Dictionary of adapter names and paths
- **max_model_len**: Maximum context length
- **tensor_parallel_size**: Number of GPUs for tensor parallelism

---

## Complete Workflow Example

Here's a complete example workflow:

### 1. Fine-tune Model

```bash
# Your training script here
python train.py --output-dir ./outputs/devstral-sft
```

### 2. Update Configuration

Edit `../../vLLM/config.yaml`:

```yaml
lora_merge:
  base_model: "mistralai/Devstral-Small-2-24B-Instruct-2512"
  adapter_path: "./outputs/devstral-sft"
  output_path: "./outputs/merged-devstral-sft"
```

### 3. Merge LoRA Adapter

```bash
python merge_lora.py
```

### 4. Push to HuggingFace

```bash
python push_to_hf.py \
  --hf-repo-id "your-username/devstral-sft" \
  --mode merge \
  --hf-token "your_token"
```

### 5. Configure vLLM Server

Update `../../vLLM/config.yaml` to use your model:

```yaml
model_information:
  model_config:
    is_model_local: false
    model_id: "your-username/devstral-sft"
```

### 6. Start vLLM Server

```bash
cd ../../vLLM
python serve.py
```

### 7. Test the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-username/devstral-sft",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `max_model_len`
   - Reduce `tensor_parallel_size`
   - Use a smaller batch size

2. **HuggingFace Authentication Failed**
   - Run `huggingface-cli login`
   - Or provide token with `--hf-token`

3. **vLLM Server Won't Start**
   - Check GPU availability
   - Verify model path is correct
   - Check `config.yaml` syntax

4. **LoRA Adapter Not Loading**
   - Verify adapter path exists
   - Check `enable_lora: true` in config
   - Ensure `max_lora_rank` matches your adapter

---

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM LoRA Support](https://docs.vllm.ai/en/stable/features/lora/)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

---

## Scripts Reference

- `merge_lora.py` - Generic LoRA merge script (uses config.yaml)
- `merge_mistral_lora.py` - Mistral-specific merge script
- `push_to_hf.py` - Upload models/adapters to HuggingFace
- `../../vLLM/serve.py` - Start vLLM API server
- `../../vLLM/config.yaml` - Main configuration file
