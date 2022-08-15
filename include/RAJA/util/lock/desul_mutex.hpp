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


#include "desul/atomics.hpp"

namespace RAJA
{

/// A mutex lock implemented using DESUL atomics.
class desul_mutex
{

  static constexpr desul::MemoryOrderAcquire acquireOrder =
      desul::MemoryOrderAcquire();
  static constexpr desul::MemoryOrderRelease releaseOrder =
      desul::MemoryOrderRelease();
  static constexpr desul::MemoryOrderRelaxed relaxedOrder =
      desul::MemoryOrderRelaxed();
  static constexpr desul::MemoryScopeNode nodeScope = desul::MemoryScopeNode();

  // The bool representing the mutex.
  // True means locked, false means unlocked.
  bool lock;

public:
  RAJA_DEVICE desul_mutex() : lock(false) {}

  /// Acquire the mutex. If locked, wait until unlocked.
  RAJA_DEVICE void acquire()
  {
    bool succeeded = false;
    while (!succeeded) {
      // Try in a loop to CAS it from false to true until successful.
      // TODO: is relaxed correct for failure order?
      bool _ = 0;
      succeeded = atomic_compare_exchange_weak(
          &lock, _, true, acquireOrder, relaxedOrder, nodeScope);
    }
  }

  /// Release the mutex.
  RAJA_DEVICE void release()
  {
    // Just write false.
    atomic_store(&lock, false, releaseOrder, nodeScope);
  }
};

}  // namespace RAJA
