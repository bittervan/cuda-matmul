#pragma once

#include "matmul.h"
#include <string>

// Load matrix from binary file
// Returns true if successful, false otherwise
bool LoadMatrix(const std::string& filename, Matrix& m);

// Free matrix memory (for dynamically allocated matrices)
void FreeMatrix(Matrix& m);

// Verify if two matrices are equal within tolerance
// Returns max absolute difference
float VerifyMatrix(const Matrix& computed, const Matrix& expected, float tolerance = 1e-3);
