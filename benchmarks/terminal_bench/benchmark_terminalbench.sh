#!/bin/bash
#
# Terminal-Bench 2.0 Benchmark Script for vLLM-served models
#
# This script benchmarks any model served via vLLM on Terminal-Bench 2.0 using harbor.
#
# Prerequisites:
#   - Docker must be installed and running
#   - vLLM API server should be running on http://localhost:8000
#   - Start the server with: python serve_api.py
#   - config.yaml should contain your model configuration
#
# Usage:
#   ./benchmark_terminalbench.sh [--n-concurrent N] [--agent-name NAME]
#

set -e  # Exit on any error

# Configuration
API_BASE="http://localhost:8000/v1"
DATASET="terminal-bench@2.0"

# Read model name from config.yaml
MODEL_NAME=$(python3 -c "import yaml; config = yaml.safe_load(open('config.yaml')); print(config['model_information']['model_config']['model_id'])")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
N_CONCURRENT=4
N_ATTEMPTS=5
AGENT_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-concurrent)
            N_CONCURRENT="$2"
            shift 2
            ;;
        --n-attempts)
            N_ATTEMPTS="$2"
            shift 2
            ;;
        --agent-name)
            AGENT_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-concurrent N    Number of concurrent benchmark tasks (default: 4)"
            echo "  --n-attempts N      Number of attempts per trial (default: 5)"
            echo "  --agent-name NAME   Custom agent name for results (default: derived from model)"
            echo "  --help              Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  CUDA_VISIBLE_DEVICES  Control GPU visibility (e.g., CUDA_VISIBLE_DEVICES=0,1)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Derive agent name if not provided
if [ -z "$AGENT_NAME" ]; then
    # Extract a short agent name from model ID (e.g., "mistralai/Model-Name" -> "model-name")
    AGENT_NAME=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g' | cut -d'-' -f1-3)
fi

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_header "Checking Prerequisites"

# Check for Python
if ! command_exists python && ! command_exists python3; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command_exists python3 && echo "python3" || echo "python")
echo -e "${GREEN}✓${NC} Python found: $($PYTHON_CMD --version)"

# Check for pip or uv
if command_exists uv; then
    PIP_CMD="uv pip"
    echo -e "${GREEN}✓${NC} uv found (will use for package management)"
elif command_exists pip3; then
    PIP_CMD="pip3"
    echo -e "${GREEN}✓${NC} pip found"
elif command_exists pip; then
    PIP_CMD="pip"
    echo -e "${GREEN}✓${NC} pip found"
else
    echo -e "${RED}Error: Neither pip nor uv is installed${NC}"
    exit 1
fi

# Check for Docker
if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Docker is required for Terminal-Bench 2.0"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker is installed and running"

# Display GPU information if available
if command_exists nvidia-smi; then
    echo -e "\n${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl -v 0

    if [ ! -z "${CUDA_VISIBLE_DEVICES}" ]; then
        echo -e "${YELLOW}CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC}  nvidia-smi not found - GPU information not available"
fi

# Check if vLLM API server is running
print_header "Checking vLLM API Server"

if curl -s --max-time 5 "${API_BASE}/models" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} vLLM API server is running at ${API_BASE}"
else
    echo -e "${YELLOW}⚠${NC}  vLLM API server is not responding at ${API_BASE}"
    echo -e "${YELLOW}⚠${NC}  Please start the server in another terminal:"
    echo -e "${YELLOW}   python serve_api.py${NC}"
    echo ""
    read -p "Press Enter once the server is running, or Ctrl+C to exit..."

    # Check again
    if ! curl -s --max-time 5 "${API_BASE}/models" > /dev/null 2>&1; then
        echo -e "${RED}Error: vLLM API server is still not responding${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Server is now responding"
fi

# Install harbor if not present
print_header "Installing Harbor CLI"

# Check if harbor exists in venv or system
HARBOR_CMD=""
if [ -f ".venv/bin/harbor" ]; then
    HARBOR_CMD=".venv/bin/harbor"
    echo -e "${GREEN}✓${NC} Harbor is already installed in .venv"
elif command_exists harbor; then
    HARBOR_CMD="harbor"
    echo -e "${GREEN}✓${NC} Harbor is already installed"
else
    echo "Installing harbor from GitHub..."
    # Install from git+https since harbor-ai package name may not be on PyPI
    $PIP_CMD install git+https://github.com/harbor-ai/harbor.git || \
    $PIP_CMD install harbor || \
    {
        echo -e "${RED}Error: Failed to install Harbor${NC}"
        echo "Please install manually: pip install git+https://github.com/harbor-ai/harbor.git"
        exit 1
    }

    # Check again after installation
    if [ -f ".venv/bin/harbor" ]; then
        HARBOR_CMD=".venv/bin/harbor"
    elif command_exists harbor; then
        HARBOR_CMD="harbor"
    fi
    echo -e "${GREEN}✓${NC} Harbor installed successfully"
fi

# Run benchmark
print_header "Running Terminal-Bench 2.0 Benchmark"

echo "Configuration:"
echo "  Dataset:        ${DATASET}"
echo "  Agent:          custom_harbor_external_agent:VLLMAgent"
echo "  Agent Name:     ${AGENT_NAME}"
echo "  Model:          ${MODEL_NAME}"
echo "  API Base:       ${API_BASE}"
echo "  N-Concurrent:   ${N_CONCURRENT}"
echo "  N-Attempts:     ${N_ATTEMPTS}"
echo ""

# Create results directory with terminalbench structure
# Single folder: terminalbench-YYYYMMDD-HHMMSS
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="./terminalbench-${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
JOB_NAME="tbench_${AGENT_NAME}"

echo "Results will be saved to: ${RESULTS_DIR}/${JOB_NAME}/"
echo ""
echo "Starting Terminal-Bench 2.0 benchmark..."
echo -e "${YELLOW}(This may take a while depending on task count and complexity)${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENT_PATH="custom_harbor_external_agent:VLLMAgent"

echo "Using custom Harbor external agent:"
echo "  Agent Path:  ${AGENT_PATH}"
echo "  Model:       ${MODEL_NAME}"
echo "  API Base:    ${API_BASE}"
echo ""

set +e  # Don't exit on error for the benchmark command

# Add current directory to PYTHONPATH so harbor can import the agent
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run harbor benchmark with Terminal-Bench 2.0
harbor run \
    -d "${DATASET}" \
    --agent-import-path "${AGENT_PATH}" \
    --model "${MODEL_NAME}" \
    --ak "api_base=${API_BASE}" \
    --jobs-dir "${RESULTS_DIR}" \
    --job-name "${JOB_NAME}" \
    -n "${N_CONCURRENT}" \
    --task-name adaptive-rejection-sampler \
    -k "${N_ATTEMPTS}"

BENCHMARK_EXIT_CODE=$?
set -e  # Re-enable exit on error

# Check if benchmark completed successfully
if [ ${BENCHMARK_EXIT_CODE} -eq 0 ]; then
    print_header "Benchmark Complete!"
    echo -e "${GREEN}✓${NC} Terminal-Bench 2.0 benchmark completed successfully"
    echo ""
    echo "Results saved to: ${RESULTS_DIR}/${JOB_NAME}/"
    echo ""

    # Show summary if results directory exists
    JOB_DIR="${RESULTS_DIR}/${JOB_NAME}"
    if [ -d "${JOB_DIR}" ]; then
        echo "Job directory contents:"
        ls -lh "${JOB_DIR}"
        echo ""

        # Look for summary or results files
        if [ -f "${JOB_DIR}/summary.json" ]; then
            echo "Summary:"
            if command_exists jq; then
                jq '.' "${JOB_DIR}/summary.json" | head -30
            else
                echo "Install 'jq' for formatted JSON output: sudo apt install jq"
                head -30 "${JOB_DIR}/summary.json"
            fi
        fi

        # Check for aggregate results
        if [ -f "${JOB_DIR}/aggregate.json" ]; then
            echo ""
            echo "Aggregate Results:"
            if command_exists jq; then
                jq '.' "${JOB_DIR}/aggregate.json"
            else
                cat "${JOB_DIR}/aggregate.json"
            fi
        fi
    fi
else
    echo -e "${RED}✗${NC} Benchmark failed with exit code: ${BENCHMARK_EXIT_CODE}"
    exit 1
fi

print_header "Next Steps"
echo "1. Review results in: ${RESULTS_DIR}/${JOB_NAME}/"
echo "2. Analyze performance metrics and task completion rates"
echo "3. Compare with other models/agents"
echo "4. Submit to leaderboard (optional):"
echo "   https://huggingface.co/datasets/alexgshaw/terminal-bench-2-leaderboard"
echo ""
