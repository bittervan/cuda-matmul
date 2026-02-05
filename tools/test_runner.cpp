#include "loader.h"
#include "matmul.h"
#include <iostream>
#include <chrono>
#include <iomanip>

void PrintMatrixInfo(const char* name, const Matrix& m) {
    std::cout << name << ": " << m.height << "x" << m.width << std::endl;
}

void RunTest(const std::string& test_name) {
    std::cout << "\n========================================\n";
    std::cout << "Running test: " << test_name << "\n";
    std::cout << "========================================\n";

    Matrix A, B, C_expected, C_computed;

    // Load test data
    if (!LoadMatrix("data/" + test_name + "_A.bin", A)) {
        std::cerr << "Failed to load matrix A\n";
        return;
    }
    if (!LoadMatrix("data/" + test_name + "_B.bin", B)) {
        std::cerr << "Failed to load matrix B\n";
        FreeMatrix(A);
        return;
    }
    if (!LoadMatrix("data/" + test_name + "_C.bin", C_expected)) {
        std::cerr << "Failed to load expected result\n";
        FreeMatrix(A);
        FreeMatrix(B);
        return;
    }

    // Allocate computed result
    C_computed.height = C_expected.height;
    C_computed.width = C_expected.width;
    C_computed.elements = new float[C_expected.height * C_expected.width];

    PrintMatrixInfo("A", A);
    PrintMatrixInfo("B", B);
    PrintMatrixInfo("C", C_expected);

    // Run matrix multiplication
    std::cout << "\nRunning MatMul...\n";
    auto start = std::chrono::high_resolution_clock::now();

    MatMul(A, B, C_computed);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Verify result
    std::cout << "\nVerifying result...\n";
    float max_diff = VerifyMatrix(C_computed, C_expected);

    if (max_diff >= 0 && max_diff < 1e-3f) {
        std::cout << "✓ PASSED! Max difference: " << std::scientific << max_diff << std::fixed << "\n";
    } else {
        std::cout << "✗ FAILED! Max difference: " << std::scientific << max_diff << std::fixed << "\n";
    }

    // Calculate GFLOPS
    long long flops = 2LL * A.height * A.width * B.width;
    double gflops = flops / (duration.count() / 1000000.0) / 1e9;
    std::cout << "Time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";

    // Cleanup
    FreeMatrix(A);
    FreeMatrix(B);
    FreeMatrix(C_expected);
    delete[] C_computed.elements;
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "  Matrix Multiplication Test Runner\n";
    std::cout << "==============================================\n";

    // Run fixed test cases
    std::cout << "\n--- Fixed Test Cases ---\n";
    RunTest("tiny");
    RunTest("small");
    RunTest("medium");
    // RunTest("large");      // Uncomment for larger tests
    // RunTest("xlarge");
    RunTest("rect1");
    RunTest("rect2");
    RunTest("boundary_16");
    RunTest("boundary_17");

    // Run 100 random test cases
    std::cout << "\n--- Random Test Cases (100) ---\n";
    int passed = 0;
    int failed = 0;

    for (int i = 0; i < 100; i++) {
        char name[64];
        snprintf(name, sizeof(name), "random_%03d", i);

        // 简化输出，只显示测试名称和结果
        std::cout << "\n[Test " << (i + 1) << "/100] " << name << "... ";

        Matrix A, B, C_expected, C_computed;

        // Load test data
        if (!LoadMatrix("data/" + std::string(name) + "_A.bin", A) ||
            !LoadMatrix("data/" + std::string(name) + "_B.bin", B) ||
            !LoadMatrix("data/" + std::string(name) + "_C.bin", C_expected)) {
            std::cout << "FAILED to load!\n";
            failed++;
            continue;
        }

        // Allocate computed result
        C_computed.height = C_expected.height;
        C_computed.width = C_expected.width;
        C_computed.elements = new float[C_expected.height * C_expected.width];

        // Run matrix multiplication
        MatMul(A, B, C_computed);

        // Verify result (capture max_diff output)
        float max_diff = VerifyMatrix(C_computed, C_expected);

        if (max_diff >= 0) {
            std::cout << "PASSED ✓\n";
            passed++;
        } else {
            std::cout << "FAILED ✗\n";
            failed++;
        }

        // Cleanup
        FreeMatrix(A);
        FreeMatrix(B);
        FreeMatrix(C_expected);
        delete[] C_computed.elements;
    }

    std::cout << "\n==============================================\n";
    std::cout << "  All tests completed!\n";
    std::cout << "  Random tests: " << passed << " passed, " << failed << " failed\n";
    std::cout << "==============================================\n";

    return 0;
}
