/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_GPU_HASHMAP_HPP
#define RAJA_GPU_HASHMAP_HPP

#include <utility>  // for std::pair

namespace RAJA
{

/// An unordered map implemented with a hash table optimized for use on GPU.
/// Uses open addressing.
///
/// Template Parameters:
/// @param K        - The type of the key for k/v pairs.
/// @param V        - The type of the value for k/v pairs.
/// @param HASHER   - A function that hashes K.
/// @param EMPTY    - A value for K that is reserved by this class to represent
///                   empty buckets. May not be inserted.
/// @param DELETED  - A value for K that is reserved by this class to represent
///                   deleted buckets. May not be inserted.
template <typename K, typename V, typename HASHER, K EMPTY, K DELETED>
class gpu_hashmap
{
  // A bucket consisting of a key/value pair.
  typedef std::pair<K, V> bucket_t;
  static constexpr size_t BUCKET_SIZE = sizeof(bucket_t);

  // An array of buckets representing the hash table itself.
  bucket_t *table;

  // The number of buckets that the table can accommodate.
  size_t capacity;

public:
  /// Constructor for the hashmap. Requires the user to pass in a chunk of
  /// allocated memory, along with its size in bytes.
  gpu_hashmap(void *chunk, size_t size)
      : table(reinterpret_cast<bucket_t *>(chunk)), capacity(size / BUCKET_SIZE)
  {
    if (size < BUCKET_SIZE) {
      throw std::invalid_argument(
          "This class requires " + std::to_string(BUCKET_SIZE) +
          " bytes per bucket, so a chunk of size " + std::to_string(size) +
          " bytes is insufficient to accommodate even a single bucket.");
    }

    if (chunk == nullptr) {
      throw std::invalid_argument("Parameter [chunk] must not be null.");
    }
  }

  /// Get the capacity of the table, in buckets.
  size_t get_capacity() const { return capacity; }

  /// Initialize the bucket at the given index.
  /// This must be done for ALL i in [0, capacity) before use.
  /// This is implemented this way so this operation can be parallelized
  /// through RAJA.
  RAJA_DEVICE void initialize(int i)
  {
    // Set all bucket's keys to EMPTY.
    if (i < capacity) table[i].first = EMPTY;
  }

  /// Searches for key K. If found, return true and set v to its value.
  /// Otherwise, return false.
  RAJA_DEVICE bool contains(const K &k, V *v)
  {
    HASHER hasher;
    size_t hash_code = hasher(k);

    for (size_t i = 0; i < capacity; ++i) {
      size_t index = (hash_code + i) % capacity;
      bucket_t &bucket = table[index];
      if (bucket.first == EMPTY) {
        // Found EMPTY--therefore, the key cannot exist anywhere in the table.
        return false;
      }
      if (bucket.first == k) {
        // Key found!
        *v = bucket.second;
        return true;
      }
    }

    // Fallback (pathological): checked every single bucket and not one was
    // EMPTY so we couldn't terminate early
    return false;
  }

  /// Inserts a key/value pair. Returns true if successful; false if failed.
  /// Failure may occur due to finding that the key is already inserted,
  /// or due to the entire table being full (pathologically bad, but possible.)
  RAJA_DEVICE bool insert(const K &k, const V &v)
  {
    HASHER hasher;
    size_t hash_code = hasher(k);

    for (size_t i = 0; i < capacity; ++i) {
      size_t index = (hash_code + i) % capacity;
      bucket_t &bucket = table[index];
      if (bucket.first == EMPTY || bucket.first == DELETED) {
        bucket.first = k;
        bucket.second = v;
        return true;
      }
      if (bucket.first == k) {
        // Key is already present
        return false;
      }
    }

    // Fallback (pathological): insert failed because the entire table is full
    return false;
  }

  /// Removes a key/value pair. If found and removed,
  /// return true and set v to its value. Otherwise, return false.
  RAJA_DEVICE bool remove(const K &k, V *v)
  {
    HASHER hasher;
    size_t hash_code = hasher(k);

    for (size_t i = 0; i < capacity; ++i) {
      size_t index = (hash_code + i) % capacity;
      bucket_t &bucket = table[index];
      if (bucket.first == EMPTY) {
        // Found EMPTY--therefore, the key cannot exist anywhere in the table.
        return false;
      }
      if (bucket.first == k) {
        // Key found!
        *v = bucket.second;
        bucket.first = DELETED;
        return true;
      }
    }

    // Fallback (pathological): checked every single bucket and not one was
    // EMPTY so we couldn't terminate early
    return false;
  }
};

}  // namespace RAJA

#endif
