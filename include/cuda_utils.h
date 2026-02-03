#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                              \
  cudaError_t _e = (call);                                 \
  if (_e != cudaSuccess) {                                 \
    fprintf(stderr, "CUDA error %s:%d: %s\n",               \
            __FILE__, __LINE__, cudaGetErrorString(_e));    \
    std::exit(1);                                           \
  }                                                        \
} while (0)

#endif // CUDA_UTILS_H
