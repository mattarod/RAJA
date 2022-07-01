//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for gpu_hashmap class
///

#include <RAJA/RAJA.hpp>

#include "RAJA/util/gpu_hashmap.hpp"
#include "RAJA_gtest.hpp"
#include "RAJA_test-base.hpp"

constexpr size_t EMPTY(size_t(-1));
constexpr size_t DELETED(size_t(-2));

void *allocate(RAJA::Index_type size)
{
  int8_t *ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged((void **)&ptr,
                               sizeof(int8_t) * size,
                               cudaMemAttachGlobal));
#else
  ptr = new int8_t[size];
#endif
  return (void *)ptr;
}

void deallocate(void *&ptr)
{
  if (ptr) {
    int8_t *ptri = (int8_t *)ptr;
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(ptri));
#else
    delete[] ptri;
#endif
    ptr = nullptr;
  }
}

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
int8_t *allocate_gpu(RAJA::Index_type size)
{
  int8_t *ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMalloc((void **)&ptr, sizeof(int8_t) * size));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipMalloc((void **)&ptr, sizeof(int8_t) * size));
#endif
  return ptr;
}

void deallocate_gpu(void *&ptr)
{
  if (ptr) {
    int8_t *ptri = (int8_t *)ptr;
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(ptri));
#elif defined(RAJA_ENABLE_HIP)
    hipErrchk(hipFree(ptri));
#endif
    ptr = nullptr;
  }
}
#endif

typedef RAJA::gpu_hashmap<size_t, size_t, std::hash<size_t>, EMPTY, DELETED>
    test_hashmap_t;

// A trivial test that simply constructs and deconstructs a hash map.
TEST(GPUHashmapUnitTest, ConstructionTest)
{
  constexpr size_t CHUNK_SIZE = 1000;
  void *chunk = allocate_gpu(CHUNK_SIZE);
  test_hashmap_t map(reinterpret_cast<void *>(chunk),
                     CHUNK_SIZE * sizeof(int8_t));
  deallocate_gpu(chunk);
}
