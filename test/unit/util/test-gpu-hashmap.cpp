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

// Allocate a fixed number of bytes.
void *allocate(const size_t size)
{
  char *ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged((void **)&ptr, size, cudaMemAttachGlobal));
#else
  ptr = new char[size];
#endif
  return (void *)ptr;
}

// Free a pointer allocated by the allocate() method above.
void deallocate(void *&ptr)
{
  char *cptr = (char *)ptr;
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(cptr));
#else
    delete[] cptr;
#endif
    ptr = nullptr;
  }
}

constexpr size_t EMPTY(size_t(-1));
constexpr size_t DELETED(size_t(-2));
typedef size_t K;
typedef size_t V;
typedef RAJA::gpu_hashmap<K, V, std::hash<K>, EMPTY, DELETED> test_hashmap_t;

RAJA_HOST_DEVICE void initialize(test_hashmap_t *map)
{
  size_t capacity = map->get_capacity();
  constexpr int CUDA_BLOCK_SIZE = 256;
  using policy = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  auto range = RAJA::RangeSegment(0, capacity);

  RAJA::forall<policy>(range,
                       [=] RAJA_HOST_DEVICE(int i) { map->initialize(i); });
}

// A trivial test that simply constructs and deconstructs a hash map.
TEST(GPUHashmapUnitTest, ConstructionTest)
{
  constexpr size_t CHUNK_SIZE = 1000;
  void *chunk = allocate(CHUNK_SIZE);
  auto map = new test_hashmap_t(chunk, CHUNK_SIZE);
  initialize(map);
  deallocate(chunk);
  delete map;
}

RAJA_HOST_DEVICE bool contains(test_hashmap_t *map, const K &k, V &v)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> result_reduction(true);
  RAJA::ReduceBitOr<RAJA::cuda_reduce, V> v_reduction((V)0x0);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int) {
    V temp = v;
    bool result = map->contains(k, temp);
    result_reduction &= result;
    v_reduction |= v;
  });

  v = v_reduction.get();
  return result_reduction.get();
}

RAJA_HOST_DEVICE bool insert(test_hashmap_t *map, const K &k, const V &v)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> result_reduction(true);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int) {
    bool result = map->insert(k, v);
    result_reduction &= result;
  });

  return result_reduction.get();
}

RAJA_HOST_DEVICE bool remove(test_hashmap_t *map, const K &k, V &v)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  RAJA::ReduceBitAnd<RAJA::cuda_reduce, bool> result_reduction(true);
  RAJA::ReduceBitOr<RAJA::cuda_reduce, V> v_reduction((V)0x0);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int) {
    V temp = v;
    bool result = map->remove(k, temp);
    result_reduction &= result;
    v_reduction |= v;
  });

  v = v_reduction.get();
  return result_reduction.get();
}

// A trivial test that simply constructs and deconstructs a hash map.
TEST(GPUHashmapUnitTest, OneElementTest)
{
  constexpr size_t CHUNK_SIZE = 1000;
  void *chunk = allocate(CHUNK_SIZE);
  auto map = new test_hashmap_t(chunk, CHUNK_SIZE);

  // Insertion of a new key should succeed
  ASSERT_TRUE(insert(map, 1, 2));

  // Reinsertion of same key should fail
  ASSERT_FALSE(insert(map, 1, 3));

  // Map should contain key and have the correct associated value
  V v = 0;
  bool result = contains(map, 1, v);
  ASSERT_TRUE(result);
  ASSERT_EQ(v, 2);

  // Map should not contain a non-inserted key
  ASSERT_FALSE(contains(map, 2, v));

  // Removing the key should succeed
  v = 0;
  ASSERT_TRUE(remove(map, 1, v));
  ASSERT_EQ(v, 2);

  // Lookup of removed key should fail
  ASSERT_FALSE(remove(map, 1, v));

  // Reinsertion of removed key should succeed
  ASSERT_TRUE(insert(map, 1, 3));

  deallocate(chunk);
  delete map;
}