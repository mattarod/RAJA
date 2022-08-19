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

/// A mutex lock implemented using DESUL atomics.
class desul_mutex
{
  typedef int32_t lock_t;

  static constexpr lock_t UNLOCKED = 0;

  static constexpr lock_t LOCKED = 1;

  // The bool representing the mutex.
  // 1 means locked, 0 means unlocked.
  int32_t lock;

public:
  RAJA_HOST_DEVICE desul_mutex() : lock(UNLOCKED) {}

  /// Acquire the mutex. If locked, wait until unlocked.
  RAJA_HOST_DEVICE void acquire()
  {
    bool exchanged_value = LOCKED;
    do {
      // Try in a loop to exchange it from UNLOCKED to LOCKED until successful.
      exchanged_value = !atomic_exchange(&lock,
                                         LOCKED,
                                         desul::MemoryOrderAcquire(),
                                         desul::MemoryScopeDevice());
    } while (exchanged_value == LOCKED);
  }

  /// Release the mutex.
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
