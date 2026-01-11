#!/bin/bash
#
# OpenThoughts Benchmark Script for vLLM-served models
#
# This script benchmarks any model served via vLLM on the OpenThoughts-TB-dev dataset using harbor.
#
# Prerequisites:
#   - vLLM API server should be running on http://localhost:8000
#   - Start the server with: python serve_api.py
#   - config.yaml should contain your model configuration
#
# Usage:
#   ./benchmark_openthoughts.sh [--local-dir DIR] [--skip-download]
#

set -e  # Exit on any error

# Configuration
DATASET_REPO="open-thoughts/OpenThoughts-TB-dev"
DEFAULT_LOCAL_DIR="./openthoughts_dataset"
API_BASE="http://localhost:8000/v1"

# Read model name from config.yaml
MODEL_NAME=$(python3 -c "import yaml; config = yaml.safe_load(open('config.yaml')); print(config['model_information']['model_config']['model_id'])")
# Extract a short agent name from model ID (e.g., "mistralai/Model-Name" -> "model-name")
AGENT_NAME=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g' | cut -d'-' -f1-3)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
LOCAL_DIR="${DEFAULT_LOCAL_DIR}"
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local-dir DIR     Directory to store the dataset (default: ./openthoughts_dataset)"
            echo "  --skip-download     Skip dataset download if already present"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

if command_exists harbor; then
    echo -e "${GREEN}✓${NC} Harbor is already installed"
    harbor --version || echo "Harbor installed (version check not available)"
else
    echo "Installing harbor from PyPI..."
    $PIP_CMD install harbor-ai
    echo -e "${GREEN}✓${NC} Harbor installed successfully"
fi

# Download dataset script
if [ "$SKIP_DOWNLOAD" = false ]; then
    print_header "Downloading Dataset Script"

    if [ -f "snapshot_download.py" ]; then
        echo -e "${YELLOW}⚠${NC}  snapshot_download.py already exists. Overwriting..."
    fi

    curl -L https://raw.githubusercontent.com/open-thoughts/OpenThoughts-Agent/refs/heads/main/eval/tacc/snapshot_download.py -o snapshot_download.py
    chmod +x snapshot_download.py
    echo -e "${GREEN}✓${NC} Downloaded snapshot_download.py"

    # Download the dataset
    print_header "Downloading OpenThoughts Dataset"

    echo "Dataset will be saved to: ${LOCAL_DIR}"
    echo "This may take several minutes depending on dataset size..."
    echo ""

    $PYTHON_CMD snapshot_download.py "${DATASET_REPO}" --local-dir "${LOCAL_DIR}"
    echo -e "${GREEN}✓${NC} Dataset downloaded successfully"
else
    echo -e "${YELLOW}⚠${NC}  Skipping dataset download (--skip-download flag set)"

    if [ ! -d "${LOCAL_DIR}" ]; then
        echo -e "${RED}Error: Dataset directory does not exist: ${LOCAL_DIR}${NC}"
        echo "Run without --skip-download to download the dataset first"
        exit 1
    fi
fi

# Run benchmark
print_header "Running Benchmark with Harbor"

echo "Configuration:"
echo "  Dataset:    ${LOCAL_DIR}"
echo "  Agent:      ${AGENT_NAME}"
echo "  Model:      ${MODEL_NAME}"
echo "  API Base:   ${API_BASE}"
echo ""

# Create results directory with terminalbench structure
# Main folder: terminalbench
# Subfolder: date/time based (e.g., 20260107/143022)
MAIN_DIR="./terminalbench"
DATE_DIR=$(date +%Y%m%d)
TIME_DIR=$(date +%H%M%S)
RESULTS_DIR="${MAIN_DIR}/${DATE_DIR}/${TIME_DIR}"
mkdir -p "${RESULTS_DIR}"
JOB_NAME="benchmark_${AGENT_NAME}"

echo "Results will be saved to: ${RESULTS_DIR}/${JOB_NAME}/"
echo ""
echo "Starting benchmark..."
echo -e "${YELLOW}(This may take a while depending on dataset size)${NC}"
echo ""

# Run harbor benchmark with custom external agent
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENT_PATH="nvidia_nemotron_agent:VLLMAgent"

echo "Using custom external agent:"
echo "  Agent Path:  ${AGENT_PATH}"
echo "  Model:       ${MODEL_NAME}"
echo "  API Base:    ${API_BASE}"
echo ""

set +e  # Don't exit on error for the benchmark command
# Add current directory to PYTHONPATH so harbor can import the agent
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
harbor run \
    --path "${LOCAL_DIR}" \
    --agent-import-path "${AGENT_PATH}" \
    --agent-args "model_name=${MODEL_NAME}" \
    --jobs-dir "${RESULTS_DIR}" \
    --job-name "${JOB_NAME}" \
    --n-concurrent 4

BENCHMARK_EXIT_CODE=$?
set -e  # Re-enable exit on error

# Check if benchmark completed successfully
if [ ${BENCHMARK_EXIT_CODE} -eq 0 ]; then
    print_header "Benchmark Complete!"
    echo -e "${GREEN}✓${NC} Benchmark completed successfully"
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
                jq '.' "${JOB_DIR}/summary.json" | head -20
            else
                echo "Install 'jq' for formatted JSON output: sudo apt install jq"
                head -20 "${JOB_DIR}/summary.json"
            fi
        fi
    fi
else
    echo -e "${RED}✗${NC} Benchmark failed with exit code: ${BENCHMARK_EXIT_CODE}"
    exit 1
fi

print_header "Next Steps"
echo "1. Review results in: ${RESULTS_DIR}/${JOB_NAME}/"
echo "2. Analyze performance metrics"
echo "3. Compare with other models/agents"
echo ""
