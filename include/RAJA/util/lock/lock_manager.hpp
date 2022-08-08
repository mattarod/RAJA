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
  lock_manager() {}

  /// Acquire the lock for index i.
  void acquire(size_t) { lock.acquire(); }

  /// Release the lock for index i.
  void release(size_t) { lock.release(); }

  /// Drop the lock on index i and take it on index j.
  /// If i and j are protected by the same lock, no-op.
  void exchange(size_t, size_t) {}
};

}  // namespace RAJA
