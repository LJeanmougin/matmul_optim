#ifndef SGEMM_H
#define SGEMM_H

#define BDIMX 32
#define BDIMY 32

int div_ceil(int numerator, int denominator);
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            float *A, float *B, float beta, float *C);

void sgemm_gpu(int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C);

void sgemm_cpu(int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C);

#endif