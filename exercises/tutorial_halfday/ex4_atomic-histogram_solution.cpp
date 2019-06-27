//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "RAJA/RAJA.hpp"

#include "memoryManager.hpp"

/*
 *  Exercise #4: Atomic histogram
 *
 *  In this exercise, you will use use RAJA atomic operations to compute
 *  an array which represents a histogram of values in another array.
 *  Given an array of length N containing integers in the interval [0, M), 
 *  you will compute entries in an array 'hist' of length M. Each entry 
 *  hist[i] in the histogram array will equal the number of occurrences of 
 *  the value 'i' in the orginal array.
 *
 *  This file contains sequential and OpenMP variants of the histogram
 *  computation using C-style for-loops. You will fill in RAJA versions of
 *  these variants, plus a RAJA CUDA version if you have access to an NVIDIA
 *  GPU and a CUDA compiler, in empty code sections indicated by comments.
 *
 *  RAJA features you will use:
 *    - `forall` loop iteration template method
 *    - Index range segment
 *    - Atomic add operation
 *
 *  If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
// Functions to check and print result.
//
void checkResult(int* hist, int* histref, int len);
void printArray(int* v, int len);


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise #4: Atomic histogram...\n";

  //
  // Define array bounds and initialize array to compute histogram of values
  // on. 
  //
  int M = 20;
  int N = 100000;

  int* array = memoryManager::allocate<int>(N);
  int* hist = memoryManager::allocate<int>(M);
  int* hist_ref = memoryManager::allocate<int>(M);

  for (int i = 0; i < N; ++i) { 
    array[i] = rand() % M;
  }


//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::cout << "\n\n Running C-style sequential historgram...\n";

  std::memset(hist_ref, 0, M * sizeof(int));

  for (int i = 0; i < N; ++i) {
      hist_ref[ array[i] ]++;
  }

//printArray(hist_ref, M);


//----------------------------------------------------------------------------//
// C-style OpenMP multithreading variant.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(hist, 0, M * sizeof(int));

  std::cout << "\n\n Running C-style OpenMP historgram...\n";

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
      #pragma omp atomic
      hist[ array[i] ]++;
  }

  checkResult(hist, hist_ref, M);
//printArray(hist, M);

#endif 


//----------------------------------------------------------------------------//
// RAJA::seq_exec policy enforces strictly sequential execution.
//----------------------------------------------------------------------------//

  std::memset(hist, 0, M * sizeof(int));

  using EXEC_POL1 = RAJA::seq_exec;
  using ATOMIC_POL1 = RAJA::atomic::seq_atomic;

  std::cout << "\n Running RAJA sequential atomic histogram...\n";

  RAJA::forall<EXEC_POL1>(RAJA::RangeSegment(0, N), [=](int i) {
    RAJA::atomic::atomicAdd<ATOMIC_POL1>(&hist[array[i]], 1);
  });

  checkResult(hist, hist_ref, M);
//printArray(hist, M);


//----------------------------------------------------------------------------//
// RAJA omp_atomic policy is used with the RAJA OpenMP execution policy.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(hist, 0, M * sizeof(int));

  using EXEC_POL2 = RAJA::omp_parallel_for_exec;
  using ATOMIC_POL2 = RAJA::atomic::omp_atomic;

  std::cout << "\n Running RAJA OpenMP atomic histogram...\n";

  RAJA::forall<EXEC_POL2>(RAJA::RangeSegment(0, N), [=](int i) {
    RAJA::atomic::atomicAdd<ATOMIC_POL2>(&hist[array[i]], 1);
  });

  checkResult(hist, hist_ref, M);
//printArray(hist, M);

#endif


//----------------------------------------------------------------------------//
// RAJA auto_atomic policy can also be used with the RAJA OpenMP 
// execution policy. 
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(hist, 0, M * sizeof(int));

  using EXEC_POL3 = RAJA::omp_parallel_for_exec;
  using ATOMIC_POL3 = RAJA::atomic::auto_atomic;

  std::cout << "\n Running RAJA OpenMP histogram with auto atomic policy...\n";
  
  RAJA::forall<EXEC_POL3>(RAJA::RangeSegment(0, N), [=](int i) {
    RAJA::atomic::atomicAdd<ATOMIC_POL3>(&hist[array[i]], 1);
  });
    
  checkResult(hist, hist_ref, M);
//printArray(hist, M);

#endif


//----------------------------------------------------------------------------//
// RAJA cuda_atomic policy is used with the RAJA CUDA execution policy.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::memset(hist, 0, M * sizeof(int));

  std::cout << "\n Running RAJA CUDA atomic histogram...\n";

  using EXEC_POL4 = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using ATOMIC_POL4 = RAJA::atomic::cuda_atomic;

  RAJA::forall<EXEC_POL4>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (int i) {
    RAJA::atomic::atomicAdd<ATOMIC_POL4>(&hist[array[i]], 1);
  });

  checkResult(hist, hist_ref, M);
//printArray(hist, M);

#endif


//----------------------------------------------------------------------------//
// RAJA auto_atomic policy can also be used with the RAJA CUDA 
// execution policy.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::memset(hist, 0, M * sizeof(int));

  using EXEC_POL5 = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using ATOMIC_POL5 = RAJA::atomic::auto_atomic;

  std::cout << "\n Running RAJA CUDA histogram with auto atomic policy...\n";
 
  RAJA::forall<EXEC_POL5>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (int i) {
    RAJA::atomic::atomicAdd<ATOMIC_POL5>(&hist[array[i]], 1);
  });
   
  checkResult(hist, hist_ref, M);
//printArray(hist, M);

#endif

  //
  // Clean up.
  //

  memoryManager::deallocate(array);
  memoryManager::deallocate(hist);
  memoryManager::deallocate(hist_ref);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* hist, int* hist_ref, int len)
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( correct && hist[i] != hist_ref[i] ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print array.
//
void printArray(int* v, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "v[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
}
