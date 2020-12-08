/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel conditional templates
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_Resource_HPP
#define RAJA_pattern_kernel_Resource_HPP


#include "RAJA/config.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace internal
{

struct ResourceBase {
};

}// end namespace internal

namespace statement
{

/*!
 * An expression that returns the value of the specified RAJA::kernel
 * parameter.
 *
 * This allows run-time values to affect the control logic within
 * RAJA::kernel execution policies.
 */
template <camp::idx_t ResourceId>
struct Resource : public internal::ResourceBase {

  constexpr static camp::idx_t param_idx = ResourceId;

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static auto eval(Data const &data)
      -> decltype(camp::get<ResourceId>(data.param_tuple))
  {
    return camp::get<ResourceId>(data.param_tuple);
  }
};

}  // end namespace statement
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
