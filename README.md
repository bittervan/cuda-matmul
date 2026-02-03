# CUDA Matrix Multiplication

Learning CUDA programming through matrix multiplication implementations.

## Project Structure

```
.
├── src/           # CUDA source files (.cu)
├── include/       # Header files (.h, .cuh)
├── build/         # Build output (ignored by git)
└── CMakeLists.txt # CMake build configuration
```

## Prerequisites

- CUDA Toolkit 12+
- CMake 3.24+
- Clang compiler (for CUDA support)

## Build

```bash
# Set environment variables
export CUDA_HOME=/path/to/cuda
export CUDA_GPU_ARCH=sm_70  # Adjust for your GPU

# Build
mkdir build && cd build
cmake ..
make
```

## Current Implementations

- [ ] SAXPY (Single-Precision A·X Plus Y)
- [ ] Matrix Multiplication (Naive)
- [ ] Matrix Multiplication (Tiled)
- [ ] Matrix Multiplication (Shared Memory)
- [ ] Matrix Multiplication (Cublas)

## License

MIT
