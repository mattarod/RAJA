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

struct gpu_hasher {
  RAJA_DEVICE
  size_t operator()(size_t const &s) const noexcept
  {
    constexpr size_t LARGE_PRIME = 17;  // FIXME
    return s * LARGE_PRIME;
  }
};

constexpr size_t EMPTY(size_t(-1));
constexpr size_t DELETED(size_t(-2));
typedef size_t K;
typedef size_t V;
typedef RAJA::gpu_hashmap<K, V, gpu_hasher, EMPTY, DELETED> test_hashmap_t;

// Allocate [size] objects of type T in GPU memory and return a pointer to it.
template <typename T>
T *allocate(size_t size)
{
  T *ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(
      cudaMallocManaged((void **)&ptr, sizeof(T) * size, cudaMemAttachGlobal));
#elif defined(RAJA_ENABLE_HIP)
  hipErrchk(hipMalloc((void **)&ptr, sizeof(T) * size));
#elif defined(RAJA_ENABLE_SYCL)
  ptr = sycl_res->allocate<T>(size, camp::resources::MemoryAccess::Managed);
#else
  ptr = new T[size];
#endif
  return ptr;
}

// Deallocate a pointer allocated by allocate().
template <typename T>
void deallocate(T *&ptr)
{
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(ptr));
#elif defined(RAJA_ENABLE_HIP)
    hipErrchk(hipFree(ptr));
#elif defined(RAJA_ENABLE_SYCL)
    sycl_res->deallocate(ptr);
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}

// Allocate a chunk of memory for [count] buckets.
void *allocate_table(size_t count)
{
  char *chunk = allocate<char>(count * test_hashmap_t::BUCKET_SIZE);
  return reinterpret_cast<void *>(chunk);
}

// Deallocate a chunk of memory allocated by allocate_table().
void deallocate_table(void *&ptr)
{
  char *ptr_c = reinterpret_cast<char *>(ptr);
  deallocate(ptr_c);
  ptr = nullptr;
}

// Helper function that initializes the gpu_hashmap.
void initialize(test_hashmap_t *map, void *chunk, const size_t bucket_count)
{
  constexpr int CUDA_BLOCK_SIZE = 256;
  using policy = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  auto range = RAJA::RangeSegment(0, 1);

  // Initialize the hashmap object.
  bool *result_gpu = allocate<bool>(1);
  RAJA::forall<policy>(range, [=] RAJA_DEVICE(int i) {
    *result_gpu =
        map->initialize(chunk, bucket_count * test_hashmap_t::BUCKET_SIZE);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, 1, cudaMemcpyDeviceToHost);
  ASSERT_TRUE(result);
  deallocate(result_gpu);

  // Initialize the buckets in the hashmap.
  range = RAJA::RangeSegment(0, bucket_count);
  RAJA::forall<policy>(range,
                       [=] RAJA_DEVICE(int i) { map->initialize_table(i); });
}

// Helper function. A function called from host that evokes a map's contains
// function in device.
bool contains(test_hashmap_t *map, const K &k, V *v)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  bool *result_gpu = allocate<bool>(1);
  V *v_gpu = allocate<V>(1);

  RAJA::forall<policy>(range, [=] RAJA_DEVICE(int) {
    *result_gpu = map->contains(k, v_gpu);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, 1, cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_gpu, 1, cudaMemcpyDeviceToHost);
  deallocate(result_gpu);
  deallocate(v_gpu);
  return result;
}

// Helper function. A function called from host that evokes a map's insert
// function in device.
bool insert(test_hashmap_t *map, const K &k, const V &v)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  bool *result_gpu = allocate<bool>(1);

  RAJA::forall<policy>(range, [=] RAJA_DEVICE(int) {
    *result_gpu = map->insert(k, v);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, 1, cudaMemcpyDeviceToHost);
  deallocate(result_gpu);
  return result;
}

// Helper function. A function called from host that evokes a map's remove
// function in device.
bool remove(test_hashmap_t *map, const K &k, V *v)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  bool *result_gpu = allocate<bool>(1);
  V *v_gpu = allocate<V>(1);

  RAJA::forall<policy>(range, [=] RAJA_DEVICE(int) {
    *result_gpu = map->remove(k, v_gpu);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, 1, cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_gpu, 1, cudaMemcpyDeviceToHost);
  deallocate(result_gpu);
  deallocate(v_gpu);
  return result;
}

//------------------//
// TESTS BEGIN HERE //
//------------------//

// A trivial test that simply constructs and deconstructs a hash map.
TEST(GPUHashmapUnitTest, ConstructionTest)
{
  test_hashmap_t *map = allocate<test_hashmap_t>(1);
  constexpr size_t BUCKET_COUNT = 1000;
  void *chunk = allocate_table(1000);
  initialize(map, chunk, BUCKET_COUNT);
  deallocate_table(chunk);
  deallocate(map);
}

// A small test that repeatedly inserts, removes, and tests a single element.
TEST(GPUHashmapUnitTest, OneElementTest)
{
  test_hashmap_t *map = allocate<test_hashmap_t>(1);
  constexpr size_t BUCKET_COUNT = 1000;
  void *chunk = allocate_table(1000);
  initialize(map, chunk, BUCKET_COUNT);

  // Insertion of a new key should succeed
  ASSERT_TRUE(insert(map, 1, 2));

  // Reinsertion of same key should fail
  ASSERT_FALSE(insert(map, 1, 3));

  // Map should contain key and have the correct associated value
  V v = 0;
  bool result = contains(map, 1, &v);
  ASSERT_TRUE(result);
  ASSERT_EQ(v, 2);

  // Map should not contain a non-inserted key
  ASSERT_FALSE(contains(map, 2, &v));

  // Removing the key should succeed
  v = 0;
  ASSERT_TRUE(remove(map, 1, &v));
  ASSERT_EQ(v, 2);

  // Lookup of removed key should fail
  ASSERT_FALSE(remove(map, 1, &v));

  // Reinsertion of removed key should succeed
  ASSERT_TRUE(insert(map, 1, 3));

  deallocate_table(chunk);
  deallocate(map);
}
