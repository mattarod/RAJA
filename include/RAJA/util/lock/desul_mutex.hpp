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
  // The bool representing the mutex.
  // True means locked, false means unlocked.
  bool lock;

public:
  RAJA_HOST_DEVICE desul_mutex() : lock(false) {}

  /// Acquire the mutex. If locked, wait until unlocked.
  RAJA_HOST_DEVICE void acquire()
  {
    bool succeeded = false;
    while (!succeeded) {
      // Try in a loop to CAS it from false to true until successful.
      // TODO: is relaxed correct for failure order?
      succeeded = !atomic_exchange(&lock,
                                   true,
                                   desul::MemoryOrderAcquire(),
                                   desul::MemoryScopeDevice());
    }
  }

  /// Release the mutex.
  RAJA_HOST_DEVICE void release()
  {
    // Just write false.
    atomic_store(&lock,
                 false,
                 desul::MemoryOrderRelease(),
                 desul::MemoryScopeDevice());
  }
};

}  // namespace RAJA
