# Default compiler
set shell := ["bash", "-c"]

NVCC := "nvcc"
CC   := "gcc"

# Flags
NVCC_FLAGS := "-O3 -arch=native"
CC_FLAGS   := "-O3"

BIN_DIR    := "bin"
SRC_FILES  := `ls *.cu`

# Default recipe
default:
    @just --list

# Build all CUDA files
build-all:
    @mkdir -p {{BIN_DIR}}
    @for f in {{SRC_FILES}}; do \
        echo "Compiling $f..."; \
        {{NVCC}} {{NVCC_FLAGS}} "$f" -o "{{BIN_DIR}}/${f%.cu}"; \
    done

# Build a specific file
build name:
    @mkdir -p {{BIN_DIR}}
    @fname="{{name}}"; \
    {{NVCC}} {{NVCC_FLAGS}} "${fname%.cu}.cu" -o "{{BIN_DIR}}/${fname%.cu}"

# Run a specific program
run name:
    @just build {{name}}
    @fname="{{name}}"; \
    ./{{BIN_DIR}}/${fname%.cu}

# Clean build artifacts
clean:
    rm -rf {{BIN_DIR}}

# Check for CUDA devices
check-devices:
    @{{NVCC}} --version
    @nvidia-smi
