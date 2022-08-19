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

#ifndef RAJA_LOCK_MANAGER_HPP
#define RAJA_LOCK_MANAGER_HPP

#include "desul_mutex.hpp"

namespace RAJA
{

/// A class to map locks to indices and manage access to them.
class lock_manager
{
  typedef desul_mutex lock_t;

  // The table of locks.
  lock_t *lock_table;

  // The number of locks.
  size_t lock_count;

public:
  static constexpr size_t LOCK_SIZE = sizeof(lock_t);

  RAJA_HOST_DEVICE lock_manager() {}

  /// Initializer for the lock table. Requires the user to pass in a chunk of
  /// allocated memory, along with its size in bytes.
  RAJA_HOST_DEVICE bool initialize(void *chunk, const size_t size)
  {
    if (size < LOCK_SIZE || chunk == nullptr) {
      return false;
    }

    lock_table = reinterpret_cast<lock_t *>(chunk);
    lock_count = size / LOCK_SIZE;
    return true;
  }

  /// Get the number of locks.
  RAJA_HOST_DEVICE size_t get_lock_count() const { return lock_count; }

  /// Initialize the lock at the given index.
  /// This must be done for ALL i in [0, lock_count) before use.
  /// This is implemented this way so this operation can be parallelized
  /// through RAJA.
  RAJA_HOST_DEVICE void initialize_locks(const int i)
  {
    // Initialize all locks to UNLOCKED.
    if (i < lock_count) {
      new (&lock_table[i]) lock_t();
    }
  }

  /// Acquire the lock for index i.
  RAJA_HOST_DEVICE void acquire(size_t)
  {
    // FIXME: For now, just use the first lock.
    lock_table[0].acquire();
  }

  /// Release the lock for index i.
  RAJA_HOST_DEVICE void release(size_t)
  {
    // FIXME: For now, just use the first lock.
    lock_table[0].release();
  }

  /// Acquire all locks in order to "stop the world."
  RAJA_HOST_DEVICE void acquire_all()
  {
    // FIXME: For now, just use the first lock.
    lock_table[0].acquire();
  }

  /// Release all locks after call to acquire_all().
  RAJA_HOST_DEVICE void release_all()
  {
    // FIXME: For now, just use the first lock.
    lock_table[0].release();
  }

  /// Drop the lock on index i and take it on index j.
  /// If i and j are protected by the same lock, no-op.
  RAJA_HOST_DEVICE void exchange(size_t, size_t)
  {
    /// FIXME: Implement.
  }
};

}  // namespace RAJA

#endif
