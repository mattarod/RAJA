//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for gpu_hashmap class
///

#include <random>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/gpu_hashmap.hpp"
#include "RAJA/util/lock/lock_manager.hpp"
#include "RAJA/util/rng/lehmer64.hpp"
#include "RAJA_gtest.hpp"
#include "RAJA_test-base.hpp"

constexpr size_t LARGE_PRIME = 2654435761;

struct gpu_hasher {
  RAJA_HOST_DEVICE
  size_t operator()(size_t const &s) const noexcept { return s * LARGE_PRIME; }
};


constexpr size_t KEY_OFFSET = 0x0CE1000000000000;
constexpr size_t VAL_OFFSET = 0x07A1000000000000;

typedef size_t K;
typedef size_t V;

constexpr size_t PROBE_LIMIT = 64;
constexpr size_t EMPTY = size_t(-1);
constexpr size_t DELETED = size_t(-2);

typedef RAJA::gpu_hashmap<K,
                          V,
                          gpu_hasher,
                          RAJA::lock_manager,
                          PROBE_LIMIT,
                          EMPTY,
                          DELETED>
    test_hashmap_t;

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

// Allocate a chunk of memory [size] bytes big and cast it to void.
void *allocate_table(size_t size)
{
  char *chunk = allocate<char>(size);
  return reinterpret_cast<void *>(chunk);
}

// Helper function that initializes the gpu_hashmap.
void initialize(test_hashmap_t *map,
                void *table_chunk,
                const size_t bucket_count,
                void *lock_chunk,
                const size_t lock_count)
{
  constexpr int CUDA_BLOCK_SIZE = 256;
  using policy = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  auto range = RAJA::RangeSegment(0, 1);

  // Initialize the hashmap object.
  bool *result_gpu = allocate<bool>(1);
  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int i) {
    new (map) test_hashmap_t();
    *result_gpu = map->initialize(table_chunk,
                                  bucket_count * test_hashmap_t::BUCKET_SIZE,
                                  lock_chunk,
                                  lock_count * RAJA::lock_manager::LOCK_SIZE);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, 1, cudaMemcpyDeviceToHost);
  ASSERT_TRUE(result);
  deallocate(result_gpu);

  // Initialize the buckets in the hashmap.
  range = RAJA::RangeSegment(0, bucket_count);
  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int i) {
    map->initialize_table(i);
  });
}

// Attempts to rezise the hashmap. If successful, returns the old chunk so that
// the caller can free it. If failed, return nullptr.
void *resize(test_hashmap_t *map, void *new_chunk, const size_t bucket_count)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  void **result_gpu = allocate<void *>(1);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int) {
    *result_gpu = map->resize(new_chunk, bucket_count);
  });

  void *result = nullptr;
  cudaMemcpy(&result, result_gpu, sizeof(void *), cudaMemcpyDeviceToHost);
  deallocate(result_gpu);
  return result;
}

// Helper function. A function called from host that evokes a map's contains
// function in device.
bool contains(test_hashmap_t *map, const K &k, V *v, size_t *probe_count)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  bool *result_gpu = allocate<bool>(1);
  V *v_gpu = allocate<V>(1);
  size_t *probe_count_gpu = allocate<size_t>(1);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int) {
    *result_gpu = map->contains(k, v_gpu, probe_count_gpu);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_gpu, sizeof(V), cudaMemcpyDeviceToHost);
  cudaMemcpy(probe_count,
             probe_count_gpu,
             sizeof(size_t),
             cudaMemcpyDeviceToHost);
  deallocate(result_gpu);
  deallocate(v_gpu);
  deallocate(probe_count_gpu);
  return result;
}

bool contains(test_hashmap_t *map, const K &k, V *v)
{
  // Delegate with a dummy variable
  size_t _;
  return contains(map, k, v, &_);
}

// Helper function. A function called from host that evokes a map's insert
// function in device.
bool insert(test_hashmap_t *map, const K &k, const V &v, size_t *probe_count)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  bool *result_gpu = allocate<bool>(1);
  size_t *probe_count_gpu = allocate<size_t>(1);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int) {
    *result_gpu = map->insert(k, v, probe_count_gpu);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(probe_count,
             probe_count_gpu,
             sizeof(size_t),
             cudaMemcpyDeviceToHost);
  deallocate(result_gpu);
  deallocate(probe_count_gpu);
  return result;
}

bool insert(test_hashmap_t *map, const K &k, const V &v)
{
  // Delegate with a dummy variable
  size_t _;
  return insert(map, k, v, &_);
}

// Helper function. A function called from host that evokes a map's remove
// function in device.
bool remove(test_hashmap_t *map, const K &k, V *v, size_t *probe_count)
{
  using policy = RAJA::cuda_exec<1>;
  auto range = RAJA::RangeSegment(0, 1);
  bool *result_gpu = allocate<bool>(1);
  V *v_gpu = allocate<V>(1);
  size_t *probe_count_gpu = allocate<size_t>(1);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int) {
    *result_gpu = map->remove(k, v_gpu, probe_count_gpu);
  });

  bool result = false;
  cudaMemcpy(&result, result_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_gpu, sizeof(V), cudaMemcpyDeviceToHost);
  cudaMemcpy(probe_count,
             probe_count_gpu,
             sizeof(size_t),
             cudaMemcpyDeviceToHost);
  deallocate(result_gpu);
  deallocate(v_gpu);
  deallocate(probe_count_gpu);
  return result;
}

bool remove(test_hashmap_t *map, const K &k, V *v)
{
  // Delegate with a dummy variable
  size_t _;
  return remove(map, k, v, &_);
}

//------------------//
// TESTS BEGIN HERE //
//------------------//

// A trivial test that simply constructs and deconstructs a hash map.
TEST(GPUHashmapUnitTest, ConstructionTest)
{
  // The number of buckets to use in the hash table for this test.
  constexpr size_t BUCKET_COUNT = 64;
  constexpr size_t LOCK_COUNT = 1;

  // Initialize map
  test_hashmap_t *map = allocate<test_hashmap_t>(1);
  void *chunk = allocate_table(BUCKET_COUNT * test_hashmap_t::BUCKET_SIZE);
  void *lock_chunk = allocate_table(LOCK_COUNT * RAJA::lock_manager::LOCK_SIZE);
  initialize(map, chunk, BUCKET_COUNT, lock_chunk, LOCK_COUNT);

  // Clean up.
  deallocate(chunk);
  deallocate(map);
}

// A small test that repeatedly inserts, removes, and tests a single element.
TEST(GPUHashmapUnitTest, OneElementTest)
{
  // The number of buckets to use in the hash table for this test.
  constexpr size_t BUCKET_COUNT = 64;
  constexpr size_t LOCK_COUNT = 1;

  // Initialize map
  test_hashmap_t *map = allocate<test_hashmap_t>(1);
  void *chunk = allocate_table(BUCKET_COUNT * test_hashmap_t::BUCKET_SIZE);
  void *lock_chunk = allocate_table(LOCK_COUNT * RAJA::lock_manager::LOCK_SIZE);
  initialize(map, chunk, BUCKET_COUNT, lock_chunk, LOCK_COUNT);

  // Insertion of a new key should succeed
  ASSERT_TRUE(insert(map, KEY_OFFSET, VAL_OFFSET));

  // Reinsertion of same key should fail
  ASSERT_FALSE(insert(map, KEY_OFFSET, VAL_OFFSET + 1));

  // Map should contain key and have the correct associated value
  V v = 0;
  ASSERT_TRUE(contains(map, KEY_OFFSET, &v));
  ASSERT_EQ(v, VAL_OFFSET);

  // Map should not contain a non-inserted key
  ASSERT_FALSE(contains(map, KEY_OFFSET + 1, &v));

  // Removing the key should succeed
  v = 0;
  ASSERT_TRUE(remove(map, KEY_OFFSET, &v));
  ASSERT_EQ(v, VAL_OFFSET);

  // Lookup of removed key should fail
  ASSERT_FALSE(remove(map, KEY_OFFSET, &v));

  // Reinsertion of removed key with a different value should succeed
  ASSERT_TRUE(insert(map, KEY_OFFSET, VAL_OFFSET + 2));

  // The value should indeed be different
  ASSERT_TRUE(contains(map, KEY_OFFSET, &v));
  ASSERT_EQ(v, VAL_OFFSET + 2);

  // Clean up.
  deallocate(chunk);
  deallocate(map);
}

// A test that inserts and removes a moderate number of elements.
TEST(GPUHashmapUnitTest, ModerateElementsTest)
{
  // The number of buckets to use in the hash table for this test.
  constexpr size_t BUCKET_COUNT = 16384;
  constexpr size_t LOCK_COUNT = 256;

  // Initialize map
  test_hashmap_t *map = allocate<test_hashmap_t>(1);
  void *chunk = allocate_table(BUCKET_COUNT * test_hashmap_t::BUCKET_SIZE);
  void *lock_chunk = allocate_table(LOCK_COUNT * RAJA::lock_manager::LOCK_SIZE);
  initialize(map, chunk, BUCKET_COUNT, lock_chunk, LOCK_COUNT);

  // General hashmap guidance is that they should be about 70% full.
  constexpr size_t ELEMENT_COUNT = (BUCKET_COUNT * 7) / 10;

  // Insert the elements.
  // TODO: It would probably be better to do this in a more idiomatically
  // RAJA-like way.
  for (size_t i = 0; i < ELEMENT_COUNT; ++i) {
    ASSERT_TRUE(insert(map, KEY_OFFSET + i, VAL_OFFSET + i));
  }

  // Test for existence of the keys.
  V v = 0;
  for (size_t i = 0; i < ELEMENT_COUNT; ++i) {
    ASSERT_TRUE(contains(map, KEY_OFFSET + i, &v));
    ASSERT_EQ(v, VAL_OFFSET + i);
  }

  // Remove the keys.
  for (size_t i = 0; i < ELEMENT_COUNT; ++i) {
    ASSERT_TRUE(remove(map, KEY_OFFSET + i, &v));
    ASSERT_EQ(v, VAL_OFFSET + i);
  }

  // Clean up.
  deallocate(chunk);
  deallocate(map);
}

// Comparison against a reference implementation for correctness and
// consistency.
TEST(GPUHashmapUnitTest, ConsistencyTest)
{
  using std::make_pair;

  // The number of operations to perform.
  constexpr size_t OPS = 16384;

  // The number of buckets to use in the hash table for this test.
  constexpr size_t START_BUCKET_COUNT = 16384;
  constexpr size_t LOCK_COUNT = 256;

  // General hashmap guidance is that they should be about 70% full.
  constexpr size_t ELEMENT_COUNT = (START_BUCKET_COUNT * 7) / 10;

  // For this test, we want about 50% of the keys to be present in the map.
  constexpr size_t KEY_RANGE = ELEMENT_COUNT * 2;

  // Initialize map
  test_hashmap_t *map = allocate<test_hashmap_t>(1);
  void *chunk = allocate_table(START_BUCKET_COUNT);
  void *lock_chunk = allocate_table(LOCK_COUNT * RAJA::lock_manager::LOCK_SIZE);
  initialize(map, chunk, START_BUCKET_COUNT, lock_chunk, LOCK_COUNT);
  size_t bucket_count = START_BUCKET_COUNT;

  // A reference implementation to test against.
  std::map<K, V> ref;

  // Initialize RNG.
  std::mt19937 mt;
  std::uniform_int_distribution<int> action_dist(0, 2);
  std::uniform_int_distribution<K> key_dist(0, KEY_RANGE);
  std::uniform_int_distribution<V> val_dist(0, size_t(-1));

  // Initialize with half of the elements.
  for (size_t k = 0; k < KEY_RANGE; k += 2) {
    V v = val_dist(mt);
    ASSERT_TRUE(insert(map, k, v));
    ref.insert(make_pair(k, v));
  }

  for (size_t i = 0; i < OPS; ++i) {
    if (i == 2015) {
      std::cout << "This is the bad one" << std::endl;
    }

    size_t probe_count = 0;
    int action = action_dist(mt);
    K k = key_dist(mt);

    if (action == 0) {
      // Contains
      V actual_v;
      bool actual_result = contains(map, k, &actual_v, &probe_count);
      auto it = ref.find(k);
      bool expected_result = (it != ref.end());
      ASSERT_EQ(actual_result, expected_result);
      if (expected_result) {
        V expected_v = it->second;
        ASSERT_EQ(actual_v, expected_v);
      }
    } else if (action == 1) {
      // Insert
      V v = val_dist(mt);
      bool actual_result = insert(map, k, v, &probe_count);

      while (!actual_result && probe_count > PROBE_LIMIT) {
        std::cout << "Resizing table from " << bucket_count << " to "
                  << bucket_count * 2 << std::endl;

        // If insert failed and every single bucket was checked, resize.
        bucket_count *= 2;
        size_t new_table_size = bucket_count * test_hashmap_t::BUCKET_SIZE;
        void *new_chunk = allocate_table(new_table_size);
        void *old_chunk = resize(map, chunk, new_table_size);

        // If resize succeeded, free old_chunk and attempt reinsertion.
        if (old_chunk != nullptr) {
          chunk = new_chunk;
          deallocate(old_chunk);
          actual_result = insert(map, k, v, &probe_count);
          // It is unlikely but possible for the insert
          // to fail again even after the resize. So, we loop.
        } else {
          // Otherwise, free new_chunk and try again (with an even bigger
          // resize).
          deallocate(new_chunk);
        }
      }

      auto ret = ref.insert(make_pair(k, v));
      auto it = ret.first;
      bool expected_result = ret.second;

      std::cout << "op " << i << ": "
                << "insert (" << k << ", " << v
                << "), result: " << actual_result
                << ", probe_count: " << probe_count << std::endl;

      ASSERT_EQ(actual_result, expected_result)
          << " insert result mismatch occurred with key " << k << ", value "
          << v << ", operation " << i
          << ". Element preventing insertion is: " << (it->first) << ", "
          << (it->second) << " with probe_count: " << probe_count;

    } else {
      // Remove
      V actual_v;
      bool actual_result = remove(map, k, &actual_v, &probe_count);
      auto it = ref.find(k);
      bool expected_result = (it != ref.end());

      std::cout << "op " << i << ": "
                << "remove (" << k << "), result: " << actual_result
                << ", probe_count: " << probe_count << std::endl;

      ASSERT_EQ(actual_result, expected_result);
      if (expected_result) {
        V expected_v = it->second;
        ref.erase(it);
        ASSERT_EQ(actual_v, expected_v);
      }
    }
  }

  // Clean up.
  deallocate(chunk);
  deallocate(map);
}

RAJA_HOST_DEVICE size_t work(test_hashmap_t *map,
                             const size_t tid,
                             const size_t num_ops,
                             const size_t key_range)
{
  // Use the lehmer RNG, as C++'s standard RNG library is unusable from GPU.
  __uint128_t g_lehmer64_state = lehmer64_seed(tid * LARGE_PRIME);

  size_t i;

  for (i = 0; i < num_ops; ++i) {
    size_t probe_count = 0;

    int action = lehmer64(g_lehmer64_state) % 3;
    K k = lehmer64(g_lehmer64_state) % key_range;

    if (action == 0) {
      // Contains
      V _;
      map->contains(k, &_, &probe_count);
    } else if (action == 1) {
      // Insert
      V v = k;

      // Resizing in a multithreaded context is not gonna be straightforward...
      // for now, just don't resize.

      /* bool result = */ map->insert(k, v, &probe_count);

      // while (!actual_result && probe_count > PROBE_LIMIT) {
      //   std::cout << "Resizing table from " << bucket_count << " to "
      //             << bucket_count * 2 << std::endl;

      //   // If insert failed and every single bucket was checked, resize.
      //   bucket_count *= 2;
      //   size_t new_table_size = bucket_count * test_hashmap_t::BUCKET_SIZE;
      //   void *new_chunk = allocate_table(new_table_size);
      //   void *old_chunk = resize(map, chunk, new_table_size);

      //   // If resize succeeded, free old_chunk and attempt reinsertion.
      //   if (old_chunk != nullptr) {
      //     chunk = new_chunk;
      //     deallocate(old_chunk);
      //     actual_result = insert(map, k, v, &probe_count);
      //     // It is unlikely but possible for the insert
      //     // to fail again even after the resize. So, we loop.
      //   } else {
      //     // Otherwise, free new_chunk and try again (with an even bigger
      //     resize). deallocate(new_chunk);
      //   }
      // }

    } else {
      // Remove
      V _;
      map->remove(k, &_, &probe_count);
    }
  }

  return i;
}

// Launch workers for a multithreaded test.
template <size_t NUM_WORKERS>
bool launch_multithread_test(const size_t num_ops)
{
  using policy = RAJA::cuda_exec<NUM_WORKERS>;

  // The number of buckets to use in the hash table for this test.
  constexpr size_t START_BUCKET_COUNT = 16384;
  constexpr size_t LOCK_COUNT = 256;

  // General hashmap guidance is that they should be about 70% full.
  constexpr size_t ELEMENT_COUNT = (START_BUCKET_COUNT * 7) / 10;

  // For this test, we want about 50% of the keys to be present in the map.
  constexpr size_t KEY_RANGE = ELEMENT_COUNT * 2;

  // Initialize map
  test_hashmap_t *map = allocate<test_hashmap_t>(1);
  void *chunk = allocate_table(START_BUCKET_COUNT);
  void *lock_chunk = allocate_table(LOCK_COUNT * RAJA::lock_manager::LOCK_SIZE);
  initialize(map, chunk, START_BUCKET_COUNT, lock_chunk, LOCK_COUNT);

  // Initialize with half of the elements.
  for (size_t k = 0; k < KEY_RANGE; k += 2) {
    bool result = insert(map, k, reinterpret_cast<V>(k));
    if (!result) return false;
  }

  auto range = RAJA::RangeSegment(0, NUM_WORKERS);
  size_t *results_gpu = allocate<size_t>(NUM_WORKERS);

  RAJA::forall<policy>(range, [=] RAJA_HOST_DEVICE(int id) {
    results_gpu[id] = work(map, id, num_ops, KEY_RANGE);
  });

  size_t results[NUM_WORKERS];
  cudaMemcpy(&results,
             results_gpu,
             sizeof(size_t) * NUM_WORKERS,
             cudaMemcpyDeviceToHost);
  deallocate(results_gpu);

  bool success = true;
  for (int i = 0; i < NUM_WORKERS; ++i) {
    // A thread is successful only if it has completed num_ops operations.
    success = success && (results[i] == num_ops);
  }

  // Clean up.
  deallocate(chunk);
  deallocate(map);
  return success;
}

// Test of basic concurrency correctness.
TEST(GPUHashmapUnitTest, TwoThreadTest)
{
  // The number of operations to perform per thread.
  constexpr size_t OPS = 256;

  // Run the test.
  ASSERT_TRUE(launch_multithread_test<2>(OPS));
}
