/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkGroup templates for
 *          hip execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hip_WorkGroup_HPP
#define RAJA_hip_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"

#include "RAJA/pattern/detail/WorkGroup.hpp"
#include "RAJA/pattern/WorkGroup.hpp"

#include "RAJA/policy/hip/policy.hpp"

#if defined(RAJA_ENABLE_HIP)

namespace RAJA
{

namespace detail
{

template < typename T, typename ... CallArgs >
__device__ void Vtable_hip_device_call(void* obj, CallArgs... args)
{
  T* obj_as_T = static_cast<T*>(obj);
  (*obj_as_T)(std::forward<CallArgs>(args)...);
}

template < typename T, typename ... CallArgs >
__global__ void get_device_Vtable_hip_device_call(void(**ptr)(void*, CallArgs...))
{
  *ptr = &Vtable_hip_device_call<T, CallArgs>;
}

inline void* get_Vtable_hip_device_call_ptrptr()
{
  void* ptrptr = nullptr;
  hipErrchk(hipMallocHost(&ptrptr, sizeof(void(*)())));

  return ptrptr;
}

inline void* get_cached_Vtable_hip_device_call_ptrptr()
{
  static void* ptrptr = get_Vtable_hip_device_call_ptrptr();
  return ptrptr;
}

// TODO: make thread safe
template < typename T, typename ... CallArgs >
inline void(*)(void*, CallArgs...) get_Vtable_hip_device_call()
{
  void(**ptrptr)(void*, CallArgs...) =
      static_cast<void(**)(void*, CallArgs...)>(
        get_cached_Vtable_hip_device_call_ptrptr());
  get_device_Vtable_hip_device_call<T, CallArgs...><<<1,1>>>(ptrptr);
  hipErrchk(hipGetLastError());
  hipErrchk(hipDeviceSynchronize());

  void(*ptr)(void*, CallArgs...) = *ptrptr;
  return ptr;
}

template < typename T, typename ... CallArgs >
inline void(*)(void*, CallArgs...) get_cached_Vtable_hip_device_call()
{
  static void(*ptr)(void*, CallArgs...) = get_Vtable_hip_device_call();
  return ptr;
}

/*!
* Populate and return a Vtable object where the
* call operator is a device function
*/
template < typename T, typename ... CallArgs >
inline Vtable<CallArgs...> get_Vtable(hip_work const&)
{
return Vtable<CallArgs...>{
      &Vtable_move_construct<T, CallArgs...>,
      get_cached_Vtable_hip_device_call(),
      &Vtable_destroy<T, CallArgs...>
    };
}

}  // namespace detail

}  // namespace RAJA

#endif

#endif  // closing endif for header file include guard
