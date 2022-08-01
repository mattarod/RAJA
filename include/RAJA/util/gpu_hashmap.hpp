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

  enum probe_result_t {
    PRESENT,           // found key
    ABSENT_AVAILABLE,  // determined key was absent, and found room for it
    ABSENT_FULL        // determined key was absent, but found no room for it
  };

  // Delegate method for insert, remove, and contains.
  // Probes the hash table for key k.
  // If k is present, return FOUND_K, and set location to its index.
  // If k is absent, return FOUND_, and set location to the first instance of
  // DELETED or EMPTY seen.
  RAJA_DEVICE probe_result_t probe(const K &k,
                                   size_t &location,
                                   size_t *probe_count)
  {
    HASHER hasher;
    size_t hash_code = hasher(k);
    size_t &i = *probe_count;
    location = capacity;  // Initialize location to invalid index

    // We stay in this first loop body until we find k, EMPTY, or DELETED.
    // In the first two cases, we return; in the last case, we break and proceed
    // to the next loop down.
    i = 0;
    while (i < capacity) {
      size_t index = (hash_code + i) % capacity;
      ++i;  // this sets probe_count to the correct value before the return.
      bucket_t &bucket = table[index];

      if (bucket.first == k) {
        // Found the key.
        location = index;
        return PRESENT;

      } else if (bucket.first == EMPTY) {
        // Found EMPTY--therefore, the key cannot exist anywhere in the table.
        // We haven't yet found any instance of DELETED,
        // so this EMPTY is the first available slot.
        location = index;
        return ABSENT_AVAILABLE;

      } else if (bucket.first == DELETED) {
        // If this is the first instance of DELETED, record it.
        // However, keep probing--the key could be further down the line.
        // We break out of this loop and proceed to the next loop.
        location = index;
        break;
      }
    }

    // If we have found an instance of DELETED, we enter this second, simpler
    // loop body, and stay in it until we find k or EMPTY.
    while (i < capacity) {
      size_t index = (hash_code + i) % capacity;
      ++i;
      bucket_t &bucket = table[index];

      if (bucket.first == k) {
        // Found the key. Since it was actually found, we want location to point
        // to it, not the first instance of DELETED.
        location = index;
        return PRESENT;

      } else if (bucket.first == EMPTY) {
        // Found EMPTY--therefore, the key cannot exist anywhere in the table.
        // location was already set to the first instance of DELETED by the loop
        // above.
        return ABSENT_AVAILABLE;
      }
    }

    // Fallback (pathological): checked every single bucket without finding K or
    // EMPTY, so unable to terminate early. If location is set to a valid index,
    // we found a DELETED bucket that can accommodate k; otherwise, the whole
    // hash table is completely full!
    return location == capacity ? ABSENT_FULL : ABSENT_AVAILABLE;
  }

public:
  static constexpr size_t BUCKET_SIZE = sizeof(bucket_t);

  /// Initializer for the hashmap. Requires the user to pass in a chunk of
  /// allocated memory, along with its size in bytes.
  RAJA_DEVICE bool initialize(void *chunk, size_t size)
  {
    if (size < BUCKET_SIZE || chunk == nullptr) {
      return false;
    }

    table = reinterpret_cast<bucket_t *>(chunk);
    capacity = size / BUCKET_SIZE;
    return true;
  }

  /// Get the capacity of the table, in buckets.
  RAJA_DEVICE size_t get_capacity() const { return capacity; }

  /// Initialize the bucket at the given index.
  /// This must be done for ALL i in [0, capacity) before use.
  /// This is implemented this way so this operation can be parallelized
  /// through RAJA.
  RAJA_DEVICE void initialize_table(int i)
  {
    // Set all bucket's keys to EMPTY.
    if (i < capacity) {
      table[i].first = EMPTY;
    }
  }

  /// Searches for key K. If found, return true and set v to its value.
  /// Otherwise, return false.
  RAJA_DEVICE bool contains(const K &k, V *v, size_t *probe_count)
  {
    size_t index;
    bool result = probe(k, index, probe_count) == PRESENT;
    if (result) {
      *v = table[index].second;
    }
    return result;
  }

  RAJA_DEVICE bool contains(const K &k, V *v)
  {
    // Delegate with a dummy variable
    size_t _;
    return contains(k, v, &_);
  }


  /// Inserts a key/value pair. Returns true if successful; false if failed.
  /// Failure may occur due to finding that the key is already inserted,
  /// or due to the entire table being full (pathologically bad, but possible.)
  RAJA_DEVICE bool insert(const K &k, const V &v, size_t *probe_count)
  {
    size_t index;
    bool result = probe(k, index, probe_count) == ABSENT_AVAILABLE;
    if (result) {
      table[index].first = k;
      table[index].second = v;
    }
    return result;
  }

  RAJA_DEVICE bool insert(const K &k, const V &v)
  {
    // Delegate with a dummy variable
    size_t _;
    return insert(k, v, &_);
  }

  /// Removes a key/value pair. If found and removed,
  /// return true and set v to its value. Otherwise, return false.
  RAJA_DEVICE bool remove(const K &k, V *v, size_t *probe_count)
  {
    size_t index;
    bool result = probe(k, index, probe_count) == PRESENT;
    if (result) {
      *v = table[index].second;
      table[index].first = DELETED;
    }
    return result;
  }

  RAJA_DEVICE bool remove(const K &k, V *v)
  {
    // Delegate with a dummy variable
    size_t _;
    return remove(k, v, &_);
  }
};

}  // namespace RAJA

#endif
