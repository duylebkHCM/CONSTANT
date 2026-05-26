#!/bin/bash
# Build Docker image for CONSTANT handwriting generation model

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
IMAGE_NAME="constant"
IMAGE_TAG="latest"
DOCKERFILE_PATH="docker/Dockerfile"
BUILD_CONTEXT="."
NO_CACHE=""
PLATFORM=""
GPU_DEVICE="all"
HOST_PORT="7860"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help function
print_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Docker image for CONSTANT handwriting generation model.

Options:
    -n, --name NAME           Image name (default: constant)
    -t, --tag TAG             Image tag (default: latest)
    -f, --dockerfile PATH     Path to Dockerfile (default: docker/Dockerfile)
    -c, --context PATH        Build context path (default: .)
    --no-cache                Build without using cache
    --platform PLATFORM       Target platform (e.g., linux/amd64, linux/arm64)
    --device DEVICE           GPU device for docker run (default: all, options: all, 0, 1, 2, etc.)
    --port PORT               Host port mapping (default: 7860)
    -h, --help                Show this help message

Examples:
    # Basic build
    $0

    # Build with custom name and tag
    $0 --name my-constant --tag v1.0

    # Build without cache
    $0 --no-cache

    # Build for specific platform
    $0 --platform linux/amd64

    # Build with device and port configuration for running
    $0 --device 0 --port 8080

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -f|--dockerfile)
            DOCKERFILE_PATH="$2"
            shift 2
            ;;
        -c|--context)
            BUILD_CONTEXT="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="--platform=$2"
            shift 2
            ;;
        --device)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --port)
            HOST_PORT="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# Validate Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE_PATH${NC}"
    exit 1
fi

# Full image reference
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Print build configuration
echo "=================================================="
echo "Docker Build Configuration"
echo "=================================================="
echo "Image name:       ${FULL_IMAGE_NAME}"
echo "Dockerfile:       ${DOCKERFILE_PATH}"
echo "Build context:    ${BUILD_CONTEXT}"
echo "No cache:         ${NO_CACHE:-false}"
echo "Platform:         ${PLATFORM:-default}"
echo "GPU device:       ${GPU_DEVICE}"
echo "Host port:        ${HOST_PORT}"
echo "=================================================="
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Build command
BUILD_CMD="docker build ${PLATFORM} ${NO_CACHE} -t ${FULL_IMAGE_NAME} -f ${DOCKERFILE_PATH} ${BUILD_CONTEXT}"

echo -e "${YELLOW}Building Docker image...${NC}"
echo "Running: ${BUILD_CMD}"
echo

# Run build
if eval "$BUILD_CMD"; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo
    echo "Image details:"
    docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo

    echo
    echo "=================================================="
    echo -e "${GREEN}Done!${NC}"
    echo "=================================================="
    echo
    echo "To run the container:"
    if [ "$GPU_DEVICE" = "all" ]; then
        echo "  docker run --gpus all -it --rm -p ${HOST_PORT}:7860 ${FULL_IMAGE_NAME}"
    else
        echo "  docker run --gpus device=${GPU_DEVICE} -it --rm -p ${HOST_PORT}:7860 ${FULL_IMAGE_NAME}"
    fi
    echo
    echo "To run with volume mounts:"
    if [ "$GPU_DEVICE" = "all" ]; then
        echo "  docker run --gpus all -it --rm \\"
    else
        echo "  docker run --gpus device=${GPU_DEVICE} -it --rm \\"
    fi
    echo "    -v \$(pwd)/data:/workspace/data \\"
    echo "    -v \$(pwd)/output:/workspace/output \\"
    echo "    -v \$(pwd)/pretrained:/workspace/pretrained \\"
    echo "    -v \$(pwd)/ckpt:/workspace/ckpt \\"
    echo "    -p ${HOST_PORT}:7860 \\"
    echo "    ${FULL_IMAGE_NAME}"
    echo

else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
