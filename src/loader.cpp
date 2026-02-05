#include "loader.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>

bool LoadMatrix(const std::string& filename, Matrix& m) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        return false;
    }

    // Read dimensions
    in.read(reinterpret_cast<char*>(&m.height), sizeof(int));
    in.read(reinterpret_cast<char*>(&m.width), sizeof(int));

    if (!in.good()) {
        std::cerr << "Error: Failed to read matrix dimensions from: " << filename << std::endl;
        return false;
    }

    // Allocate memory
    size_t num_elements = m.height * m.width;
    m.elements = new float[num_elements];

    // Read elements
    in.read(reinterpret_cast<char*>(m.elements), num_elements * sizeof(float));

    if (!in.good()) {
        std::cerr << "Error: Failed to read matrix data from: " << filename << std::endl;
        delete[] m.elements;
        m.elements = nullptr;
        return false;
    }

    in.close();
    return true;
}

void FreeMatrix(Matrix& m) {
    if (m.elements != nullptr) {
        delete[] m.elements;
        m.elements = nullptr;
    }
}

float VerifyMatrix(const Matrix& computed, const Matrix& expected, float tolerance) {
    // Check dimensions
    if (computed.height != expected.height || computed.width != expected.width) {
        std::cerr << "Error: Dimension mismatch - computed ("
                  << computed.height << "x" << computed.width
                  << ") vs expected ("
                  << expected.height << "x" << expected.width << ")" << std::endl;
        return -1.0f;
    }

    size_t num_elements = computed.height * computed.width;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    size_t passed_abs = 0;
    size_t passed_rel = 0;
    size_t failed_count = 0;

    // 绝对误差阈值和相对误差阈值
    const float abs_tolerance = 1e-3f;   // 绝对误差阈值
    const float rel_tolerance = 1e-2f;   // 相对误差阈值 1%

    for (size_t i = 0; i < num_elements; i++) {
        float computed_val = computed.elements[i];
        float expected_val = expected.elements[i];
        float abs_diff = std::abs(computed_val - expected_val);

        // 计算相对误差 (避免除以0)
        float rel_diff = 0.0f;
        if (std::abs(expected_val) > 1e-10f) {
            rel_diff = abs_diff / std::abs(expected_val);
        } else if (abs_diff < abs_tolerance) {
            // 如果期望值接近0，只看绝对误差
            rel_diff = 0.0f;
        } else {
            rel_diff = 1.0f;  // 期望值为0但有误差，相对误差视为100%
        }

        // 记录最大误差
        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;

        // 检查是否通过：只要绝对误差或相对误差有一个符合就算通过
        bool passed = (abs_diff <= abs_tolerance) || (rel_diff <= rel_tolerance);
        if (passed) {
            if (abs_diff <= abs_tolerance) passed_abs++;
            if (rel_diff <= rel_tolerance) passed_rel++;
        } else {
            failed_count++;
            if (failed_count <= 10) {  // 打印前10个错误
                int row = i / computed.width;
                int col = i % computed.width;
                std::cerr << "Error at (" << row << "," << col
                         << "): computed=" << computed_val
                         << ", expected=" << expected_val
                         << ", abs_diff=" << abs_diff
                         << ", rel_diff=" << (rel_diff * 100.0f) << "%" << std::endl;
            }
        }
    }

    std::cout << "Max absolute difference: " << std::scientific << max_abs_diff << std::fixed << std::endl;
    std::cout << "Max relative difference: " << std::scientific << max_rel_diff << std::fixed << std::endl;
    std::cout << "Passed (abs tolerance): " << passed_abs << " / " << num_elements << std::endl;
    std::cout << "Passed (rel tolerance): " << passed_rel << " / " << num_elements << std::endl;
    std::cout << "Failed (both criteria): " << failed_count << " / " << num_elements << std::endl;

    // 返回最大绝对误差用于判断（负数表示失败）
    return failed_count == 0 ? max_abs_diff : -1.0f;
}
