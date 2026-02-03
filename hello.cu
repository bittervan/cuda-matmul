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

__global__ void saxpy(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  const int N = 1 << 20;
  const float a = 2.0f;

  float *x, *y;
  CUDA_CHECK(cudaMallocManaged(&x, N * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, N * sizeof(float)));

  for (int i = 0; i < N; ++i) { x[i] = 1.0f; y[i] = 2.0f; }

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  saxpy<<<blocks, threads>>>(N, a, x, y);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // check a few values
  float max_err = 0.0f;
  for (int i = 0; i < 10; ++i) {
    float expected = a * 1.0f + 2.0f; // 4.0
    float err = fabsf(y[i] - expected);
    if (err > max_err) max_err = err;
    printf("y[%d]=%.1f\n", i, y[i]);
  }
  printf("max_err=%.6f\n", max_err);

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  return 0;
}
