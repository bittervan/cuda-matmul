#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>

typedef struct {
    int width;
    int height;
    std::vector<float> elements;
} Matrix;

// CPU 矩阵乘法
void MatMulCPU(const Matrix& A, const Matrix& B, Matrix& C) {
    memset(C.elements.data(), 0, C.elements.size() * sizeof(float));

    for (int row = 0; row < C.height; row++) {
        for (int col = 0; col < C.width; col++) {
            float sum = 0;
            for (int k = 0; k < A.width; k++) {
                sum += A.elements[row * A.width + k] * B.elements[k * B.width + col];
            }
            C.elements[row * C.width + col] = sum;
        }
    }
}

// 保存矩阵到文件
void SaveMatrix(const char* filename, const Matrix& m) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&m.height), sizeof(int));
    out.write(reinterpret_cast<const char*>(&m.width), sizeof(int));
    out.write(reinterpret_cast<const char*>(m.elements.data()),
              m.height * m.width * sizeof(float));
}

// 生成随机矩阵
Matrix GenerateRandomMatrix(int height, int width, float min_val = 0.0f, float max_val = 10.0f) {
    Matrix m;
    m.height = height;
    m.width = width;
    m.elements.resize(height * width);

    for (size_t i = 0; i < m.elements.size(); i++) {
        m.elements[i] = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
    }

    return m;
}

void GenerateTestCase(const char* name, int M, int K, int N) {
    std::cout << "Generating test case: " << name << " (" << M << "x" << K << " x " << K << "x" << N << ")\n";

    Matrix A = GenerateRandomMatrix(M, K);
    Matrix B = GenerateRandomMatrix(K, N);
    Matrix C;
    C.height = M;
    C.width = N;
    C.elements.resize(M * N);

    MatMulCPU(A, B, C);

    std::string prefix = std::string("data/") + name;
    SaveMatrix((prefix + "_A.bin").c_str(), A);
    SaveMatrix((prefix + "_B.bin").c_str(), B);
    SaveMatrix((prefix + "_C.bin").c_str(), C);

    std::cout << "  Saved to: " << prefix << "_*.bin\n";
}

int main() {
    srand(time(nullptr));

    // 创建 data 目录
    system("mkdir -p data");

    // 生成不同尺寸的测试用例
    GenerateTestCase("tiny", 4, 4, 4);
    GenerateTestCase("small", 32, 32, 32);
    GenerateTestCase("medium", 128, 128, 128);
    GenerateTestCase("large", 512, 512, 512);
    GenerateTestCase("xlarge", 1024, 1024, 1024);

    // 非方阵
    GenerateTestCase("rect1", 256, 512, 128);
    GenerateTestCase("rect2", 128, 256, 512);

    // 边界情况
    GenerateTestCase("boundary_16", 16, 16, 16);
    GenerateTestCase("boundary_17", 17, 17, 17);  // 测试不能被 BLOCK_SIZE 整除的情况

    std::cout << "\nAll test cases generated successfully!\n";
    return 0;
}
