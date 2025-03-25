#include <iostream>
#include <sgemm.cuh>
#include <mat_tools.cuh>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}


__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            float *A, float *B, float beta, float *C)
{
  // printf("Kernel running on block : %u thread : %u\n", blockIdx.x, threadIdx.x);
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < N && y < M)
  {
    float tmp = 0.0f;
    int i;
    for (i = 0; i < K; i++)
      tmp += A[y * K + i] * B[i * N + x];
    C[y * N + x] = alpha * tmp + beta * C[y * N + x];
  }
}

void sgemm_gpu(int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C)
{
  float *d_A, *d_B, *d_C;
  dim3 gridDim(div_ceil(N, 32), div_ceil(M, 32), 1);
  dim3 blockDim(BDIMX, BDIMY, 1);
  
  cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
  cudaCheck(cudaMalloc(&d_B, N * K * sizeof(float)));
  cudaCheck(cudaMalloc(&d_C, N * M * sizeof(float)));

  cudaCheck(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

  cudaCheck(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaCheck(cudaFree(d_A));
  cudaCheck(cudaFree(d_B));
  cudaCheck(cudaFree(d_C));
}

void sgemm_cpu(int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C)
{
  int x, y, i;
  float tmp;
  for (y = 0; y < M; y++)
    {
      for (x = 0; x < N; x++)
      {
        tmp = 0.0f;
        for (i = 0; i < K; i++)
          tmp += A[y * K + i] * B[i * N + x];
        C[y * N + x] = alpha * tmp + beta * C[y * N + x];
      }
    }
}