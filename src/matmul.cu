#include <matmul.h>

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
    float Cvalue = 0;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    printf("row: %x, col: %x\n", row, col);
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A, d_B, d_C;
    size_t size = 0;

    d_A.height = A.height;
    d_A.width = A.width;
    size = d_A.height * d_A.width;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    d_B.height = B.height;
    d_B.width = B.width;
    size = d_B.height * d_B.width;
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    d_C.height = C.height;
    d_C.width = C.width;
    size = d_C.height * d_C.width;
    cudaMalloc(&d_B.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_C.width + dimBlock.x - 1) / dimBlock.x, (d_C.height + dimBlock.y - 1) / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}