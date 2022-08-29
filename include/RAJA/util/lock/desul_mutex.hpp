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

#ifndef RAJA_DESUL_MUTEX_HPP
#define RAJA_DESUL_MUTEX_HPP

#include "desul/atomics.hpp"

namespace RAJA
{

/// A simple mutex lock implemented using DESUL atomics.
/// There are two possible states, locked and unlocked.
class desul_mutex
{
  // The type used to implement the lock. bool doesn't work as of this writing,
  // so a 32-bit integer is used instead.
  typedef int32_t lock_t;

  // The value representing an unheld mutex.
  static constexpr lock_t UNLOCKED = 0;

  // The value representing a held mutex.
  static constexpr lock_t LOCKED = 1;

  // The value representing the mutex.
  lock_t lock;

public:
  RAJA_HOST_DEVICE desul_mutex() : lock(UNLOCKED) {}

  /// Atomically acquire the mutex. If locked, busywait until it is available.
  RAJA_HOST_DEVICE void acquire()
  {
    bool exchanged_value = LOCKED;
    do {
      // Try in a loop to exchange it from UNLOCKED to LOCKED until successful.
      // TODO: may be cheaper to use a regular read until unlocked?
      exchanged_value = !atomic_exchange(&lock,
                                         LOCKED,
                                         desul::MemoryOrderAcquire(),
                                         desul::MemoryScopeDevice());
    } while (exchanged_value == LOCKED);
  }

  /// Atomically release the mutex.
  RAJA_HOST_DEVICE void release()
  {
    // Just write UNLOCKED.
    atomic_store(&lock,
                 UNLOCKED,
                 desul::MemoryOrderRelease(),
                 desul::MemoryScopeDevice());
  }
};

}  // namespace RAJA

#endif
