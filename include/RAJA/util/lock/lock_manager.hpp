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


#include "desul_mutex.hpp"

namespace RAJA
{

/// A class to map locks to indices and manage access to them.
class lock_manager
{
  // For now, just one mutex.
  desul_mutex lock;

public:
  RAJA_HOST_DEVICE lock_manager() {}

  /// Acquire the lock for index i.
  RAJA_HOST_DEVICE void acquire(size_t) { lock.acquire(); }

  /// Release the lock for index i.
  RAJA_HOST_DEVICE void release(size_t) { lock.release(); }

  /// Acquire all locks in order to "stop the world."
  RAJA_HOST_DEVICE void acquire_all() { lock.acquire(); }

  /// Release all locks after call to acquire_all().
  RAJA_HOST_DEVICE void release_all() { lock.release(); }

  /// Drop the lock on index i and take it on index j.
  /// If i and j are protected by the same lock, no-op.
  RAJA_HOST_DEVICE void exchange(size_t, size_t) {}
};

}  // namespace RAJA
