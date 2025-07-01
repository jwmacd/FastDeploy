# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

NEVER BUILD THIS IN THE LOCAL ENVIRONMENT

## Project Overview

FastDeploy is an inference and deployment toolkit for Large Language Models (LLMs) and Visual Language Models (VLMs) based on PaddlePaddle. It provides production-ready deployment solutions with advanced acceleration technologies including PD disaggregation, unified KV cache transmission, and comprehensive quantization support.

## Build and Development Commands

### Building FastDeploy

**CRITICAL: Always use Docker for building. The build requires CUDA and specific GPU libraries not available in standard environments.**

```bash
# Docker build for GPU version
docker build -f dockerfiles/Dockerfile.gpu -t fastdeploy-gpu .

# Build script usage (inside Docker only)
bash build.sh 1 python false "[89]"  # Note: architecture must be JSON array format
```

Build script parameters:
1. Build wheel (1=yes, 0=no)
2. Python version (default: "python")
3. CPU BF16 support (default: "false")
4. GPU architectures as JSON array (e.g., "[89]" for SM_89)

### Running Tests

```bash
# Run all unit tests
bash scripts/run_unittest.sh

# Run specific test file
python test/layers/test_attention.py

# Run CI tests with pytest
python -m pytest -sv test/ci_use/test_qwen2_offline.py

# Run all CI tests
bash scripts/run_ci.sh
```

### Code Quality

```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Format code with YAPF
yapf --in-place --verbose fastdeploy/**/*.py

# Lint with Ruff (line length 120)
ruff --fix --line-length=120 fastdeploy/

# Sort imports
isort fastdeploy/
```

## Architecture Overview

### Core Components

1. **Engine Layer** (`fastdeploy/engine/`)
   - `LLMEngine`: Main inference engine managing model operations
   - Resource management for GPU/CPU allocation
   - Request/response data structures

2. **Model Execution** (`fastdeploy/model_executor/`)
   - Model implementations in `models/` (ERNIE 4.5, Qwen, etc.)
   - Neural network layers in `layers/`
   - Hardware-specific operators in `ops/`
   - CUDA graph optimization support

3. **Scheduling System** (`fastdeploy/scheduler/`)
   - Local and distributed scheduling
   - PD disaggregation with splitwise scheduler
   - Request batching and prioritization

4. **Cache Management** (`fastdeploy/cache_manager/`)
   - KV cache with prefix caching
   - Cache transfer via IPC/RDMA
   - Dynamic cache allocation

5. **Communication** (`fastdeploy/distributed/` and `fastdeploy/inter_communicator/`)
   - Multi-GPU parallelism (TP, EP, DP)
   - ZMQ-based inter-process communication
   - IPC signals and queues

### Parallelism Strategies

- **Tensor Parallelism (TP)**: Model layers split across GPUs
- **Expert Parallelism (EP)**: For Mixture of Experts models
- **Data Parallelism (DP)**: Request-level parallelism
- **PD Disaggregation**: Separating prefill and decode phases for efficiency

### Key Design Patterns

- **Registry Pattern**: Automatic model registration system
- **Factory Pattern**: Hardware backend selection
- **Observer Pattern**: Request/response handling
- **Strategy Pattern**: Pluggable scheduling and caching strategies

## Important Implementation Details

### Custom Operators

The `custom_ops/` directory contains hardware-specific implementations. The build system expects GPU architectures as a JSON array in the `FD_BUILDING_ARCS` environment variable.

### Model Support

Models are registered automatically via decorators in `fastdeploy/model_executor/models/registry.py`. Each model implementation extends base classes and implements required interfaces for attention, MLP, and other components.

### Configuration System

Central configuration in `fastdeploy/config.py`:
- `ModelConfig`: Architecture parameters
- `ParallelConfig`: Distributed execution
- `MoEConfig`: Mixture of Experts
- `SpeculativeConfig`: Speculative decoding
- `GraphOptimizationConfig`: CUDA optimizations

### Hardware Backends

Different hardware support is implemented via backend-specific operators:
- NVIDIA GPU: Primary backend with full feature support
- Kunlunxin XPU, Iluvatar GPU, Enflame GCU: Alternative backends
- CPU: Fallback implementation

## Common Issues and Solutions

1. **TypeError in build.sh**: The architecture parameter must be a JSON array (e.g., "[89]" not "89")

2. **CUDA library errors**: Build must use Docker with CUDA stubs. The Dockerfile creates symlinks for libcuda.so.1 during build.

3. **Import errors**: Ensure PaddlePaddle GPU version is installed: `pip install paddlepaddle-gpu==3.1.0`

## API Compatibility

FastDeploy maintains compatibility with vLLM interfaces:
- OpenAI-compatible API server
- Similar `LLM` class interface for offline inference
- Compatible sampling parameters

## Performance Optimization Features

- **Quantization**: W8A16, W4A16, W2A16, FP8, etc.
- **CUDA Graph**: Static graph capture for decode phase
- **Chunked Prefill**: Breaking long sequences into chunks
- **Speculative Decoding**: Multi-token prediction
- **Prefix Caching**: Reusing computed KV cache for common prefixes