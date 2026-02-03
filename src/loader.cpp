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
    float max_diff = 0.0f;
    size_t error_count = 0;

    for (size_t i = 0; i < num_elements; i++) {
        float diff = std::abs(computed.elements[i] - expected.elements[i]);
        if (diff > tolerance) {
            if (error_count < 10) {  // Print first 10 errors
                int row = i / computed.width;
                int col = i % computed.width;
                std::cerr << "Error at (" << row << "," << col
                         << "): computed=" << computed.elements[i]
                         << ", expected=" << expected.elements[i]
                         << ", diff=" << diff << std::endl;
            }
            error_count++;
        }
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    if (error_count > 0) {
        std::cerr << "Total errors: " << error_count << " / " << num_elements << std::endl;
    }

    return max_diff;
}
