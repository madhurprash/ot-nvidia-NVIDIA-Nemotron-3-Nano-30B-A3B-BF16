# OpenThoughts Benchmark Guide for NVIDIA Nemotron-3-Nano-30B

This guide walks you through benchmarking the NVIDIA Nemotron-3-Nano-30B model on the OpenThoughts-TB-dev dataset.

## Overview

The benchmarking process involves:
1. Starting the vLLM API server with your model
2. Setting up the model as a local agent that harbor can communicate with
3. Downloading the OpenThoughts benchmark dataset
4. Running the benchmark and analyzing results

---

## Step 1: Start the vLLM API Server

First, start your model as an OpenAI-compatible API server:

```bash
# Start the server (runs in foreground)
python serve_api.py
```

The server will be available at:
- **API Base URL**: `http://localhost:8000/v1`
- **OpenAI Compatible**: `/v1/chat/completions` endpoint
- **API Docs**: http://localhost:8000/docs

**Important**: Keep this terminal open. The server must be running during benchmarking.

### Verify the Server is Running

In a **new terminal**, test the server:

```bash
# Check if server is responding
curl http://localhost:8000/v1/models

# Test a completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

If you see a response, the server is working correctly!

---

## Step 2: Install Harbor CLI

Harbor is the benchmarking tool for OpenThoughts datasets.

```bash
# Install harbor
pip install harbor-ai

# Verify installation
harbor --version
```

---

## Step 3: Download the OpenThoughts Dataset

### 3.1 Download the Dataset Script

```bash
# Download the dataset downloader script
curl -L https://raw.githubusercontent.com/open-thoughts/OpenThoughts-Agent/refs/heads/main/eval/tacc/snapshot_download.py -o snapshot_download.py

# Make it executable
chmod +x snapshot_download.py
```

### 3.2 Download the OpenThoughts-TB-dev Dataset

```bash
# Download the dataset to a local directory
python snapshot_download.py open-thoughts/OpenThoughts-TB-dev --local-dir ./openthoughts_dataset
```

This will download the dataset to `./openthoughts_dataset/`. The download may take several minutes depending on the dataset size.

---

## Step 4: Configure Your Model as an Agent

Harbor needs to know how to communicate with your model. The agent configuration tells harbor:
- Where your model API is running
- What model name to use
- How to format requests

### Create an Agent Configuration File

Create a file called `agent_config.json`:

```json
{
  "agent_name": "nvidia-nemotron-nano",
  "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
  "api_base": "http://localhost:8000/v1",
  "api_type": "openai",
  "temperature": 0.6,
  "max_tokens": 2048
}
```

**Note**: Adjust `temperature` and `max_tokens` based on your config.yaml settings.

---

## Step 4B: Using a Custom External Agent (Alternative)

As an alternative to the basic agent configuration above, you can use a custom Harbor external agent that provides more control and better integration with your local vLLM server.

### What is an External Agent?

External agents are custom Python implementations that interface with Harbor's environment through the `BaseAgent` interface. They provide:
- Better error handling and retry logic
- Custom system prompts and behavior
- Full control over model interaction
- Iterative problem-solving capabilities

### Files Created

This project includes a pre-built external agent:

1. **nvidia_nemotron_agent.py** - Custom agent implementation
   - Implements Harbor's `BaseAgent` interface
   - Connects to your local vLLM server
   - Handles bash command execution and iteration

2. **test_agent.py** - Agent testing script
   - Verifies server connection
   - Tests agent configuration

### Testing the External Agent

Before running benchmarks, test that the agent can connect to your vLLM server:

```bash
python test_agent.py
```

Expected output:
```
Testing NVIDIA Nemotron External Agent
============================================================

Agent Name: nvidia-nemotron-nano
Agent Version: 1.0.0
Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
API Base: http://localhost:8000/v1

Testing connection to vLLM server...
✓ Successfully connected to vLLM server
  Available models: [...]
```

### How the External Agent Works

The external agent implements a sophisticated interaction loop:

1. **Initialization**: Creates an OpenAI client pointing to your local vLLM server
2. **Task Execution**:
   - Receives task instructions from Harbor
   - Sends them to your model via chat completions API
   - Extracts bash commands from model responses
   - Executes commands using Harbor's environment
   - Feeds results back to the model
3. **Iteration**: Continues for up to 10 iterations or until task completion

### Agent Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Harbor CLI     │────────▶│  External Agent  │────────▶│  vLLM Server    │
│                 │         │  (nvidia_nemotron│         │  (localhost:8000)│
│                 │         │   _agent.py)     │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  Task Execution  │
                            │  (Bash Commands) │
                            └──────────────────┘
```

### Customizing the Agent

You can customize the agent by editing `nvidia_nemotron_agent.py`:

```python
# Adjust initialization parameters
agent = NvidiaNemotronAgent(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    api_base="http://localhost:8000/v1",  # Change if using different port
    temperature=0.1,                       # Adjust sampling temperature
    max_tokens=8192,                       # Max tokens per response
)
```

**System Prompt Customization**: Edit the `system_prompt` variable in the `run()` method to change how the agent approaches tasks.

**Iteration Limit**: Modify `max_iterations` in the `run()` method:
```python
max_iterations = 20  # Increase for more complex tasks
```

### Using the External Agent with Harbor

The benchmark script has been updated to automatically use the external agent. When you run:

```bash
./benchmark_openthoughts.sh
```

It will use the `--agent-import-path` parameter to load your custom agent:

```bash
harbor run \
    --path ./openthoughts_dataset \
    --agent-import-path nvidia_nemotron_agent.py \
    --jobs-dir ./benchmark_results \
    --job-name benchmark_nvidia-nemotron-nano_20241218_120000
```

### Manual Usage

You can also use the external agent directly with Harbor CLI:

```bash
harbor run \
    --path /path/to/task \
    --agent-import-path nvidia_nemotron_agent:NvidiaNemotronAgent \
    --jobs-dir ./results \
    --job-name my-test-job \
    --n-concurrent 4
```

### Advantages of External Agent

Compared to the basic configuration:
- ✅ Better error handling and recovery
- ✅ Iterative problem-solving (multiple rounds of interaction)
- ✅ Automatic bash command extraction and execution
- ✅ Custom system prompts for better task performance
- ✅ Full control over model interaction logic
- ✅ Easier debugging and logging

### Troubleshooting External Agent

**Import errors**:
```bash
# Ensure Harbor is installed
pip install harbor
python -c "import harbor; print(harbor.__version__)"
```

**Agent can't connect**:
```bash
# Verify vLLM server is running
curl http://localhost:8000/v1/models
```

**Module not found**:
```bash
# Make sure you're in the correct directory
cd /home/ubuntu/ot-nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
python test_agent.py
```

---

## Step 5: Run the Benchmark

Now you're ready to run the benchmark!

### Using the Automated Script

```bash
# Make the script executable
chmod +x benchmark_openthoughts.sh

# Run the benchmark
./benchmark_openthoughts.sh
```

### Manual Benchmark Command

If you prefer to run harbor manually:

```bash
# Basic benchmark command
harbor run \
  --dataset ./openthoughts_dataset \
  --agent nvidia-nemotron-nano \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --api-base http://localhost:8000/v1 \
  --output ./benchmark_results/results_$(date +%Y%m%d_%H%M%S).json
```

### Advanced Options

```bash
# With custom configuration
harbor run \
  --dataset ./openthoughts_dataset \
  --config agent_config.json \
  --output ./results.json \
  --verbose
```

**Harbor may have different command syntax** - check the documentation:
```bash
harbor --help
harbor run --help
```

---

## Step 6: Monitor the Benchmark

While the benchmark runs:

1. **Watch server logs**: In your server terminal, you'll see API requests
2. **Monitor GPU usage**: Run `nvidia-smi` in another terminal
3. **Estimated time**: Depends on dataset size and model speed

Example monitoring:
```bash
# In a new terminal, watch GPU usage
watch -n 1 nvidia-smi
```

---

## Step 7: Analyze Results

After the benchmark completes, review the results:

```bash
# View results (requires jq for formatting)
cat ./benchmark_results/results_*.json | jq '.'

# Or without jq
cat ./benchmark_results/results_*.json
```

### Typical Metrics

OpenThoughts benchmarks typically evaluate:
- **Accuracy**: How correct the model's reasoning is
- **Completeness**: Whether the model fully addresses the problem
- **Reasoning Quality**: Quality of step-by-step thinking
- **Task Success Rate**: Percentage of tasks completed successfully

---

## Troubleshooting

### Server Not Responding

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# If not, restart the server
python serve_api.py
```

### Out of Memory Errors

Edit `config.yaml` and reduce:
```yaml
vllm_engine_config:
  max_model_len: 4096  # Reduce from 8192
  gpu_memory_utilization: 0.80  # Reduce from 0.85
```

Then restart the server.

### Harbor Not Found

```bash
# Reinstall harbor
pip install --upgrade harbor-ai

# Check PATH
which harbor
```

### Dataset Download Fails

```bash
# Try with huggingface-cli instead
pip install huggingface_hub[cli]

huggingface-cli download open-thoughts/OpenThoughts-TB-dev \
  --local-dir ./openthoughts_dataset \
  --repo-type dataset
```

---

## Quick Reference

### Complete Workflow (One-Command Summary)

```bash
# Terminal 1: Start the server
python serve_api.py

# Terminal 2: Run everything
pip install harbor-ai && \
curl -L https://raw.githubusercontent.com/open-thoughts/OpenThoughts-Agent/refs/heads/main/eval/tacc/snapshot_download.py -o snapshot_download.py && \
chmod +x snapshot_download.py && \
python snapshot_download.py open-thoughts/OpenThoughts-TB-dev --local-dir ./openthoughts_dataset && \
./benchmark_openthoughts.sh
```

### Configuration Files

- `config.yaml` - Model and vLLM configuration
- `agent_config.json` - Agent configuration for harbor
- `serve_api.py` - API server script

### Important URLs

- **Model API**: http://localhost:8000/v1
- **API Docs**: http://localhost:8000/docs
- **OpenThoughts Dataset**: https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev
- **OpenThoughts GitHub**: https://github.com/open-thoughts/OpenThoughts-Agent

---

## Next Steps

After benchmarking:

1. **Compare Results**: Run benchmarks with different models/configurations
2. **Optimize Settings**: Adjust temperature, max_tokens in config.yaml
3. **Analyze Failures**: Review specific tasks where the model struggled
4. **Share Results**: Contribute findings to the OpenThoughts community

---

## Additional Resources

- **vLLM Documentation**: https://docs.vllm.ai
- **OpenThoughts Documentation**: Check the GitHub repository
- **Harbor Documentation**: Run `harbor --help` for more information

---

## Notes

- The benchmark script assumes your server is running on `localhost:8000`
- Adjust `agent_config.json` if you use a different port or host
- Results are saved with timestamps in `./benchmark_results/`
- Keep the server running throughout the entire benchmark process
