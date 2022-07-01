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

constexpr size_t EMPTY(size_t(-1));
constexpr size_t DELETED(size_t(-2));
typedef size_t K;
typedef size_t V;
typedef RAJA::gpu_hashmap<K, V, std::hash<K>, EMPTY, DELETED> test_hashmap_t;

// A trivial test that simply constructs and deconstructs a hash map.
TEST(GPUHashmapUnitTest, ConstructionTest)
{
  constexpr size_t CHUNK_SIZE = 1000;
  void *chunk = allocate(CHUNK_SIZE);
  test_hashmap_t map(chunk, CHUNK_SIZE);
  deallocate(chunk);
}

// A trivial test that simply constructs and deconstructs a hash map.
TEST(GPUHashmapUnitTest, OneElementTest)
{
  constexpr size_t CHUNK_SIZE = 1000;
  void *chunk = allocate(CHUNK_SIZE);
  test_hashmap_t map(chunk, CHUNK_SIZE);

  // Insertion of a new key should succeed
  ASSERT_TRUE(map.insert(1, 2));

  // Reinsertion of same key should fail
  ASSERT_FALSE(map.insert(1, 3));

  // Map should contain key and have the correct associated value
  V v = 0;
  ASSERT_TRUE(map.contains(1, v));
  ASSERT_EQ(v, 2);

  // Map should not contain a non-inserted key
  ASSERT_FALSE(map.contains(2, v));

  // Removing the key should succeed
  v = 0;
  ASSERT_TRUE(map.remove(1, v));
  ASSERT_EQ(v, 2);

  // Lookup of removed key should fail
  ASSERT_FALSE(map.remove(1, v));

  // Reinsertion of removed key should succeed
  ASSERT_TRUE(map.insert(1, 3));

  deallocate(chunk);
}