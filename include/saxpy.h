#ifndef SAXPY_H
#define SAXPY_H

__global__ void saxpy(int n, float a, const float* x, float* y);

#endif // SAXPY_H
