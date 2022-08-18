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

#include "../policy/desul/atomic.hpp"
#include "lock/lock_manager.hpp"

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

  // An array of buckets representing the hash table itself.
  bucket_t *table;

  // The number of buckets that the table can accommodate.
  size_t capacity;

  // Class to handle bucket access.
  lock_manager lock_mgr;

  enum probe_result_t {
    PRESENT,           // found key
    ABSENT_AVAILABLE,  // determined key was absent, and found room for it
    ABSENT_FULL        // determined key was absent, but found no room for it
  };

  // Delegate method for insert, remove, and contains.
  // Probes the hash table for key k.
  // If k is present, return FOUND_K, and set location to its index.
  // If k is absent, but an EMPTY bucket was found, return ABSENT_AVAILABLE, and
  // set location to the first instance of EMPTY seen.
  // If k is absent, and there is not a single EMPTY bucket in the entire hash
  // table, return ABSENT_FULL.
  // This method will also lock the bucket containing the element, and keep it
  // locked for the caller's sake, except in the case of exhaustive search.
  RAJA_HOST_DEVICE probe_result_t probe(const K &k,
                                   size_t &location,
                                   size_t *probe_count)
  {
    HASHER hasher;
    size_t hash_code = hasher(k);
    size_t &i = *probe_count;
    location = capacity;  // Initialize location to invalid index

    size_t last_locked = hash_code % capacity;
    lock_mgr.acquire(hash_code % capacity);

    // We stay in this first loop body until we find k or EMPTY. DELETED buckets
    // are considered burned, so they are treated the same as any non-matching
    // key.
    i = 0;
    while (i < capacity) {
      size_t index = (hash_code + i) % capacity;
      lock_mgr.exchange(last_locked, index);
      last_locked = index;
      ++i;  // this sets probe_count to the correct value before the return.
      bucket_t &bucket = table[index];

      if (bucket.first == k) {
        // Found the key.
        location = index;
        return PRESENT;

      } else if (bucket.first == EMPTY) {
        // Found EMPTY--therefore, the key cannot exist anywhere in the table.
        location = index;
        return ABSENT_AVAILABLE;
      }
    }

    // Fallback (pathological): Checked entire table without finding k or EMPTY.
    // Key is absent, and there is no room to insert it.
    lock_mgr.release(last_locked);
    return ABSENT_FULL;
  }

public:
  static constexpr size_t BUCKET_SIZE = sizeof(bucket_t);

  /// Initializer for the hashmap. Requires the user to pass in a chunk of
  /// allocated memory, along with its size in bytes.
  RAJA_HOST_DEVICE bool initialize(void *chunk, const size_t size)
  {
    if (size < BUCKET_SIZE || chunk == nullptr) {
      return false;
    }

    table = reinterpret_cast<bucket_t *>(chunk);
    capacity = size / BUCKET_SIZE;
    return true;
  }

  /// Get the capacity of the table, in buckets.
  RAJA_HOST_DEVICE size_t get_capacity() const { return capacity; }

  /// Initialize the bucket at the given index.
  /// This must be done for ALL i in [0, capacity) before use.
  /// This is implemented this way so this operation can be parallelized
  /// through RAJA.
  RAJA_HOST_DEVICE void initialize_table(const int i)
  {
    // Set all bucket's keys to EMPTY.
    if (i < capacity) {
      table[i].first = EMPTY;
    }
  }

  /// Searches for key K. If found, return true and set v to its value.
  /// Otherwise, return false.
  RAJA_HOST_DEVICE bool contains(const K &k, V *v, size_t *probe_count)
  {
    size_t index;
    probe_result_t result = probe(k, index, probe_count);
    if (result == PRESENT) {
      *v = table[index].second;
    }
    if (result != ABSENT_FULL) lock_mgr.release(index);
    return result;
  }

  RAJA_HOST_DEVICE bool contains(const K &k, V *v)
  {
    // Delegate with a dummy variable
    size_t _;
    return contains(k, v, &_);
  }


  /// Inserts a key/value pair. Returns true if successful; false if failed.
  /// Failure may occur due to finding that the key is already inserted,
  /// or due to the entire table being full (pathologically bad, but possible.)
  RAJA_HOST_DEVICE bool insert(const K &k, const V &v, size_t *probe_count)
  {
    size_t index;
    probe_result_t result = probe(k, index, probe_count);
    if (result == ABSENT_AVAILABLE) {
      table[index].first = k;
      table[index].second = v;
    }
    if (result != ABSENT_FULL) lock_mgr.release(index);
    return result;
  }

  RAJA_HOST_DEVICE bool insert(const K &k, const V &v)
  {
    // Delegate with a dummy variable
    size_t _;
    return insert(k, v, &_);
  }

  /// Removes a key/value pair. If found and removed,
  /// return true and set v to its value. Otherwise, return false.
  RAJA_HOST_DEVICE bool remove(const K &k, V *v, size_t *probe_count)
  {
    size_t index;
    probe_result_t result = probe(k, index, probe_count);
    if (result == PRESENT) {
      *v = table[index].second;
      table[index].first = DELETED;
    }
    if (result != ABSENT_FULL) lock_mgr.release(index);
    return result;
  }

  RAJA_HOST_DEVICE bool remove(const K &k, V *v)
  {
    // Delegate with a dummy variable
    size_t _;
    return remove(k, v, &_);
  }

  /// Resizes the hashmap.
  /// If successful, returns old chunk, so that caller can free it. If failed
  /// due to new chunk not being big enough, returns nullptr.
  /// TODO: This can be made faster by parallelizing it.
  RAJA_HOST_DEVICE void *resize(void *new_chunk, const size_t new_capacity)
  {
    lock_mgr.acquire_all();

    size_t old_capacity = capacity;
    bucket_t *old_table = table;

    capacity = new_capacity;
    table = reinterpret_cast<bucket_t *>(new_chunk);
    bool any_failed = false;

    // Initialize the new table to EMPTY
    // FIXME: Speed this up with RAJA parallelism.
    for (size_t i = 0; i < capacity; ++i) {
      table[i].first = EMPTY;
    }

    // Copy elements from old table to new.
    for (size_t i = 0; i < old_capacity; ++i) {
      bucket_t &bucket = old_table[i];
      K &k = bucket.first;

      if (k != EMPTY && k != DELETED) {
        if (!insert(k, bucket.second)) {
          any_failed = true;
          break;
        }
      }
    }

    if (any_failed) {
      // Restore the old table
      capacity = old_capacity;
      table = old_table;
      lock_mgr.release_all();
      return nullptr;
    }

    lock_mgr.release_all();
    return reinterpret_cast<void *>(old_table);
  }
};

}  // namespace RAJA

#endif
