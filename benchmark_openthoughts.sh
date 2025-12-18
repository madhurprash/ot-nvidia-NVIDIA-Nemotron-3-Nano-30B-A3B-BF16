#!/bin/bash
#
# OpenThoughts Benchmark Script for NVIDIA Nemotron-3-Nano-30B-A3B-BF16
#
# This script benchmarks the model on the OpenThoughts-TB-dev dataset using harbor.
#
# Prerequisites:
#   - vLLM API server should be running on http://localhost:8000
#   - Start the server with: python serve_api.py
#
# Usage:
#   ./benchmark_openthoughts.sh [--local-dir DIR] [--skip-download]
#

set -e  # Exit on any error

# Configuration
DATASET_REPO="open-thoughts/OpenThoughts-TB-dev"
DEFAULT_LOCAL_DIR="./openthoughts_dataset"
AGENT_NAME="nvidia-nemotron-nano"
MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
API_BASE="http://localhost:8000/v1"

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

# Check for pip
if ! command_exists pip && ! command_exists pip3; then
    echo -e "${RED}Error: pip is not installed${NC}"
    exit 1
fi

PIP_CMD=$(command_exists pip3 && echo "pip3" || echo "pip")
echo -e "${GREEN}✓${NC} pip found"

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

# Create results directory
RESULTS_DIR="./benchmark_results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/benchmark_${AGENT_NAME}_${TIMESTAMP}.json"

echo "Results will be saved to: ${RESULTS_FILE}"
echo ""
echo "Starting benchmark..."
echo -e "${YELLOW}(This may take a while depending on dataset size)${NC}"
echo ""

# Run harbor benchmark
# Note: The exact harbor command syntax may vary depending on the version
# Check 'harbor --help' or 'harbor run --help' for the correct syntax
# Common patterns include:
#   - harbor run --dataset <dir> --agent <name> --model <model> --output <file>
#   - harbor eval --dataset <dir> --config <config.json> --output <file>
#
# If the command below doesn't work, you may need to:
# 1. Create an agent_config.json with your settings
# 2. Use: harbor run --dataset "${LOCAL_DIR}" --config agent_config.json --output "${RESULTS_FILE}"

set +e  # Don't exit on error for the benchmark command
harbor run \
    --dataset "${LOCAL_DIR}" \
    --agent "${AGENT_NAME}" \
    --model "${MODEL_NAME}" \
    --api-base "${API_BASE}" \
    --output "${RESULTS_FILE}"

BENCHMARK_EXIT_CODE=$?
set -e  # Re-enable exit on error

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
    print_header "Benchmark Complete!"
    echo -e "${GREEN}✓${NC} Benchmark completed successfully"
    echo ""
    echo "Results saved to: ${RESULTS_FILE}"
    echo ""

    # Show summary if results file exists
    if [ -f "${RESULTS_FILE}" ]; then
        echo "Summary:"
        if command_exists jq; then
            jq '.' "${RESULTS_FILE}" | head -20
        else
            echo "Install 'jq' for formatted JSON output: sudo apt install jq"
            head -20 "${RESULTS_FILE}"
        fi
    fi
else
    echo -e "${RED}✗${NC} Benchmark failed"
    exit 1
fi

print_header "Next Steps"
echo "1. Review results in: ${RESULTS_FILE}"
echo "2. Analyze performance metrics"
echo "3. Compare with other models/agents"
echo ""
