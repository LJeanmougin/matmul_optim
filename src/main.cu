#include <iostream>
#include <sgemm.cuh>
#include <mat_tools.cuh>
#include <chrono>
#include <filesystem>

#define MAT_SIZE 4092

int main()
{
  float *A, *B, *gpuC, *cpuC;

  A = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  B = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  gpuC = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  cpuC = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));

  randomize_matrix(A, MAT_SIZE * MAT_SIZE);
  randomize_matrix(B, MAT_SIZE * MAT_SIZE);
  randomize_matrix(gpuC, MAT_SIZE * MAT_SIZE);

  copy_matrix(cpuC, gpuC, MAT_SIZE * MAT_SIZE);

  // sgemm_cpu(MAT_SIZE, MAT_SIZE, MAT_SIZE, 0.2, A, B, 0.3, cpuC);
    
  auto start = std::chrono::high_resolution_clock::now();
  
  sgemm_gpu(MAT_SIZE, MAT_SIZE, MAT_SIZE, 0.2, A, B, 0.3, gpuC, sgemm_naive);
  
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "New version : " << duration.count()/1000000.0f << " seconds" << std::endl;

  verify_matrix(cpuC, gpuC, MAT_SIZE * MAT_SIZE);

  free(A);
  free(B);
  free(gpuC);

  return 0;
}