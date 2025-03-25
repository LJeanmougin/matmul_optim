#ifndef MAT_TOOLS_H
#define MAT_TOOLS_H

#include <fstream>
#include <unistd.h>

void randomize_matrix(float *mat, int size);
void copy_matrix(float *src, float *dst, int size);
bool verify_matrix(float *A, float *B, int size);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
void cudaCheck(cudaError_t error, const char *file, int line);

#endif