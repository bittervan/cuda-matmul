#include <matmul.h>
#include <cassert>

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
    float Cvalue = 0;

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int len = A.width;

    for (int k = 0; k < len; k += BLOCK_SIZE) {
        int row_s = threadIdx.y;
        int col_s = threadIdx.x;

        if (row < A.height && col_s + k < A.width) {
            s_A[row_s][col_s] = A.elements[row * A.width + col_s + k];
        } else {
            s_A[row_s][col_s] = 0;
        }

        if (col < B.width && row_s + k < B.height) {
            s_B[row_s][col_s] = B.elements[(row_s + k) * B.width + col];
        } else {
            s_B[row_s][col_s] = 0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            Cvalue += s_A[row_s][i] * s_B[i][col_s];
        }

        __syncthreads();
    }

    if (row < C.height && col < C.width) {
        C.elements[row * C.width + col] = Cvalue;
    }
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A, d_B, d_C;
    size_t size = 0;

    d_A.height = A.height;
    d_A.width = A.width;
    size = d_A.height * d_A.width;
    cudaMalloc(&d_A.elements, size * sizeof(float));
    cudaMemcpy(d_A.elements, A.elements, size * sizeof(float), cudaMemcpyHostToDevice);

    d_B.height = B.height;
    d_B.width = B.width;
    size = d_B.height * d_B.width;
    cudaMalloc(&d_B.elements, size * sizeof(float));
    cudaMemcpy(d_B.elements, B.elements, size * sizeof(float), cudaMemcpyHostToDevice);

    d_C.height = C.height;
    d_C.width = C.width;
    size = d_C.height * d_C.width;
    cudaMalloc(&d_C.elements, size * sizeof(float));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_C.width + dimBlock.x - 1) / dimBlock.x, (d_C.height + dimBlock.y - 1) / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}