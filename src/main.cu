#include <matmul.h>
#include <cstdio>
#include <cstdlib>

void PrintMatrix(const char* name, const Matrix m) {
    printf("%s (%dx%d):\n", name, m.height, m.width);
    for (int i = 0; i < m.height; i++) {
        for (int j = 0; j < m.width; j++) {
            printf("%6.2f ", m.elements[i * m.width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // 简单的 3x3 矩阵乘法示例
    // A = [1 2 3]    B = [7  8  9]
    //     [4 5 6]        [10 11 12]
    //     [7 8 9]        [13 14 15]
    // 结果 C 应该是 3x3

    const int M = 3, N = 3, K = 3;

    Matrix A, B, C;
    A.height = M;
    A.width = K;
    B.height = K;
    B.width = N;
    C.height = M;
    C.width = N;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    A.elements = (float*)malloc(size_A);
    B.elements = (float*)malloc(size_B);
    C.elements = (float*)malloc(size_C);

    // 初始化矩阵 A 和 B
    for (int i = 0; i < M * K; i++) {
        A.elements[i] = i + 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        B.elements[i] = i + 7.0f;
    }

    // 打印输入矩阵
    PrintMatrix("Matrix A", A);
    PrintMatrix("Matrix B", B);

    // 调用矩阵乘法
    MatMul(A, B, C);

    // 打印结果矩阵
    PrintMatrix("Matrix C (A x B)", C);

    // 释放内存
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}
