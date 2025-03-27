#include <iostream>
#include <sgemm.cuh>
#include <mat_tools.cuh>
#include <chrono>
#include <filesystem>

#define MAT_SIZE 4092

int main()
{
  float *A, *B, *gpuC, *cpuC;
  std::ofstream fs("./log.txt");
  auto start = std::chrono::high_resolution_clock::now();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  A = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  B = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  gpuC = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));
  cpuC = (float *) malloc (MAT_SIZE * MAT_SIZE * sizeof(float));

  randomize_matrix(A, MAT_SIZE * MAT_SIZE);
  randomize_matrix(B, MAT_SIZE * MAT_SIZE);
  randomize_matrix(gpuC, MAT_SIZE * MAT_SIZE);

  copy_matrix(cpuC, gpuC, MAT_SIZE * MAT_SIZE);

  print_matrix(gpuC, MAT_SIZE, MAT_SIZE, fs);

  
    // COALESCED KERNEL
    start = std::chrono::high_resolution_clock::now();
    
    sgemm_gpu_coal(MAT_SIZE, MAT_SIZE, MAT_SIZE, 0.2, A, B, 0.3, cpuC);
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    print_matrix(cpuC, MAT_SIZE, MAT_SIZE, fs);
    std::cout << "Coal version : " << duration.count()/1000000.0f << " seconds" << std::endl;
  // sgemm_cpu(MAT_SIZE, MAT_SIZE, MAT_SIZE, 0.2, A, B, 0.3, cpuC);
    
  start = std::chrono::high_resolution_clock::now();
  
  sgemm_gpu_naive(MAT_SIZE, MAT_SIZE, MAT_SIZE, 0.2, A, B, 0.3, gpuC);
  
  
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  print_matrix(gpuC, MAT_SIZE, MAT_SIZE, fs);
  std::cout << "Naive version : " << duration.count()/1000000.0f << " seconds" << std::endl;


  verify_matrix(cpuC, gpuC, MAT_SIZE * MAT_SIZE);

  free(A);
  free(B);
  free(gpuC);
  free(cpuC);

  return 0;
}