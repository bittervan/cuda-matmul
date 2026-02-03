#pragma once 

typedef struct {
  int width;
  int height;
  float *elements;
} Matrix;

void MatMul(const Matrix, const Matrix, Matrix);

#define BLOCK_SIZE 16