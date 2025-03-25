#include <iostream>
#include <sgemm.cuh>
#include <mat_tools.cuh>
#include <chrono>
#include <filesystem>

#define MAT_SIZE 4092

int main()
{
  float *A, *B, *cpuC, *gpuC;

  A = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  B = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  cpuC = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  gpuC = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));

  randomize_matrix(A, MAT_SIZE * MAT_SIZE);
  randomize_matrix(B, MAT_SIZE * MAT_SIZE);
  randomize_matrix(cpuC, MAT_SIZE * MAT_SIZE);
  copy_matrix(gpuC, cpuC, MAT_SIZE * MAT_SIZE);
  
  verify_matrix(gpuC, cpuC, MAT_SIZE * MAT_SIZE);
  
  // CPU Version for reference
  auto start = std::chrono::high_resolution_clock::now();
  
  sgemm_cpu(MAT_SIZE, MAT_SIZE, MAT_SIZE, 0.2, A, B, 0.3, cpuC);
  
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Monothread CPU duration : " << duration.count()/1000000.0f << " seconds" << std::endl;
  
  // GPU Version (Taking into account copy operations)

  start = std::chrono::high_resolution_clock::now();
  
  sgemm_gpu(MAT_SIZE, MAT_SIZE, MAT_SIZE, 0.2, A, B, 0.3, gpuC);
  
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Kernel duration : " << duration.count()/1000000.0f << " seconds" << std::endl;

  if(verify_matrix(gpuC, cpuC, MAT_SIZE * MAT_SIZE))
  {
    std::cout << "Valid result" << std::endl;
  }

  free(A);
  free(B);
  free(cpuC);
  free(gpuC);

  return 0;
}