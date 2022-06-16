/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for basic RAJA configuration options.
 *
 *          This file contains platform-specific parameters that control
 *          aspects of compilation of application code using RAJA. These
 *          parameters specify: SIMD unit width, data alignment information,
 *          inline directives, etc.
 *
 *          IMPORTANT: These options are set by CMake and depend on the options
 *          passed to it.
 *
 *          IMPORTANT: Exactly one e RAJA_COMPILER_* option must be defined to
 *          ensure correct behavior.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_config_HPP
#define RAJA_config_HPP

#include <utility>
#include <type_traits>

#if defined(_MSVC_LANG)
#define RAJA_CXX_VER _MSVC_LANG
#else
#define RAJA_CXX_VER __cplusplus
#endif

#if RAJA_CXX_VER >= 201402L
#define RAJA_HAS_SOME_CXX14 1
#elif __cpp_generic_lambdas >= 201304 && \
      __cpp_constexpr >=  201304 && \
      __cpp_decltype_auto >= 201304 && \
      __cpp_return_type_deduction >= 201304 && \
      __cpp_aggregate_nsdmi >= 201304 && \
      __cpp_variable_templates >= 201304
#define RAJA_HAS_SOME_CXX14 1
#else
#define RAJA_HAS_SOME_CXX14 0

static_assert(__cpp_generic_lambdas >= 201304,
              "RAJA requires C++14 __cpp_generic_lambdas to operate.");

static_assert(__cpp_constexpr >=  201304,
              "RAJA requires C++14 __cpp_constexpr to operate.");

static_assert(__cpp_decltype_auto >= 201304 ,
              "RAJA requires C++14 __cpp_decltype_auto to operate");

static_assert(__cpp_return_type_deduction >= 201304,
              "RAJA requires C++14 __cpp_return_type_deduction to operate");

static_assert(__cpp_aggregate_nsdmi >= 201304,
              "RAJA requires C++14 __cpp_aggregate_nsdmi to operate");

#endif
/* NOTE: we want this one __cpp_init_captures >= 201304 */
/* NOTE: we want this too __cpp_lib_transformation_trait_aliases >= 201304 */
// __has_cpp_attribute(deprecated) >= 201309
// __cpp_lib_exchange_function >= 201304
// __cpp_lib_make_unique >= 201304
// __cpp_lib_integer_sequence >= 201304
// __cpp_lib_tuples_by_type >= 201304
// __cpp_lib_tuple_element_t >= 201402
// __cpp_lib_transparent_operators >= 201210
// __cpp_lib_integral_constant_callable >= 201304
// __cpp_lib_result_of_sfinae >= 201210
// __cpp_binary_literals 201304
// __cpp_sized_deallocation  201309
// __cpp_lib_is_final  201402
// __cpp_lib_is_null_pointer 201309
// __cpp_lib_chrono_udls 201304
// __cpp_lib_string_udls 201304
// __cpp_lib_generic_associative_lookup  201304
// __cpp_lib_null_iterators  201304
// __cpp_lib_make_reverse_iterator 201402
// __cpp_lib_robust_nonmodifying_seq_ops 201304
// __cpp_lib_complex_udls  201309
// __cpp_lib_quoted_string_io  201304
// __has_include(<shared_mutex>) 1
// __cpp_lib_shared_timed_mutex  201402


static_assert(RAJA_HAS_SOME_CXX14,
              "RAJA requires at least basic C++14 to operate correctly, your "
              "compiler and/or standard library does not claim support for "
              "C++14 features we need");

/*!
 ******************************************************************************
 *
 * \brief Enable/disable exploratory fault tolerance mechanism.
 *
 ******************************************************************************
 */
/* #undef RAJA_ENABLE_FT */
/* #undef RAJA_ENABLE_ITERATOR_OVERFLOW_DEBUG */
/*!
 ******************************************************************************
 *
 * \brief Default RAJA floating point scalar and pointer options.
 *
 ******************************************************************************
 */
#define RAJA_USE_DOUBLE
#define RAJA_USE_RESTRICT_PTR

/*!
 ******************************************************************************
 *
 * \brief Deprecated tests
 *
 ******************************************************************************
 */
/* #undef RAJA_DEPRECATED_TESTS */

/*!
 ******************************************************************************
 *
 * \brief Add forceinline recursive directive to Kernel and Forall (Intel only)
 *
 ******************************************************************************
 */
#define RAJA_ENABLE_FORCEINLINE_RECURSIVE

/*!
 ******************************************************************************
 *
 * \brief Add bounds check to views and layouts
 *
 ******************************************************************************
 */
/* #undef RAJA_ENABLE_BOUNDS_CHECK */

/*
 ******************************************************************************
 *
 * \brief Exhaustive index types for tests
 *
 ******************************************************************************
 */
/* #undef RAJA_TEST_EXHAUSTIVE */

/*!
 ******************************************************************************
 *
 * \brief Programming model back-ends.
 *
 ******************************************************************************
 */
/* #undef RAJA_ENABLE_OPENMP */
/* #undef RAJA_ENABLE_TARGET_OPENMP */
/* #undef RAJA_ENABLE_TBB */
/* #undef RAJA_ENABLE_CUDA */
/* #undef RAJA_ENABLE_CLANG_CUDA */
/* #undef RAJA_ENABLE_HIP */
/* #undef RAJA_ENABLE_SYCL */

/* #undef RAJA_ENABLE_NV_TOOLS_EXT */
/* #undef RAJA_ENABLE_ROCTX */

/*!
 ******************************************************************************
 *
 * \brief Timer options.
 *
 ******************************************************************************
 */
#define RAJA_USE_CHRONO
/* #undef RAJA_USE_GETTIME */
/* #undef RAJA_USE_CLOCK */
/* #undef RAJA_USE_CYCLE */

/*!
 ******************************************************************************
 *
 * \brief Runtime plugins.
 *
 ******************************************************************************
 */
/* #undef RAJA_ENABLE_RUNTIME_PLUGINS */

/*!
 ******************************************************************************
 *
 * \brief Desul atomics.
 *
 ******************************************************************************
 */
/* #undef RAJA_ENABLE_DESUL_ATOMICS */

/*!
 ******************************************************************************
 *
 * \brief Detect the host C++ compiler we are using.
 *
 ******************************************************************************
 */
#if defined(__INTEL_COMPILER)
#define RAJA_COMPILER_INTEL
#elif defined(__ibmxl__)
#define RAJA_COMPILER_XLC
#elif defined(__clang__)
#define RAJA_COMPILER_CLANG
#elif defined(__PGI)
#define RAJA_COMPILER_PGI
#elif defined(_WIN32)
#define RAJA_COMPILER_MSVC
#elif defined(__GNUC__)
#define RAJA_COMPILER_GNU
#endif

#define RAJA_STRINGIFY(x) RAJA_DO_STRINGIFY(x)
#define RAJA_DO_STRINGIFY(x) #x
#ifdef _WIN32
#define RAJA_PRAGMA(x) __pragma(x)
#else
#define RAJA_PRAGMA(x) _Pragma(RAJA_STRINGIFY(x))
#endif

namespace RAJA {

#if defined(RAJA_ENABLE_OPENMP)
#if defined(_OPENMP)
#if _OPENMP >= 200805
#define RAJA_ENABLE_OPENMP_TASK
#endif
#else
#error RAJA configured with RAJA_ENABLE_OPENMP, but OpenMP not supported by current compiler
#endif // _OPENMP
#endif // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA) && defined(__CUDACC__)
#define RAJA_CUDA_ACTIVE
#endif // RAJA_ENABLE_CUDA && __CUDACC__

#if defined(RAJA_ENABLE_HIP) && defined(__HIPCC__)
#define RAJA_HIP_ACTIVE

#include <hip/hip_version.h>
#if (HIP_VERSION_MAJOR > 4) || \
    (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 3)
// enable device function pointers with rocm version >= 4.3
#define RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL
#endif
#if (HIP_VERSION_MAJOR > 4) || \
    (HIP_VERSION_MAJOR == 4 && HIP_VERSION_MINOR >= 2)
// enable occupancy calculator with rocm version >= 4.2
// can't test older versions thought they may work
#define RAJA_ENABLE_HIP_OCCUPANCY_CALCULATOR
#endif
#endif // RAJA_ENABLE_HIP && __HIPCC__

#if defined(RAJA_CUDA_ACTIVE) || \
  defined(RAJA_HIP_ACTIVE)
#define RAJA_DEVICE_ACTIVE
#endif

/*!
 ******************************************************************************
 *
 * \brief RAJA software version number.
 *
 ******************************************************************************
 */
#define RAJA_VERSION_MAJOR 2022
#define RAJA_VERSION_MINOR 3
#define RAJA_VERSION_PATCHLEVEL 0


/*!
 ******************************************************************************
 *
 * \brief Useful macros.
 *
 ******************************************************************************
 */

//
//  Platform-specific constants for data alignment:
//
//     DATA_ALIGN - used in compiler-specific intrinsics and type aliases
//                  to specify alignment of data, loop bounds, etc.;
//                  units of "bytes"
const int DATA_ALIGN = 64;

#if defined (_WIN32)
#define RAJA_RESTRICT __restrict
#else
#define RAJA_RESTRICT __restrict__
#endif

#if !defined(RAJA_COMPILER_MSVC)
#define RAJA_COLLAPSE(X) collapse(X)
#else
#define RAJA_COLLAPSE(X)
#endif

//
// Runtime bounds checking for Views
//
#if defined(RAJA_ENABLE_BOUNDS_CHECK)
#define RAJA_BOUNDS_CHECK_INTERNAL
#define RAJA_BOUNDS_CHECK_constexpr
#else
#define RAJA_BOUNDS_CHECK_constexpr constexpr
#endif

//
//  Compiler-specific definitions for inline directives, data alignment
//  intrinsics, and SIMD vector pragmas
//
//  Variables for compiler instrinsics, directives, type aliases
//
//     RAJA_INLINE - macro to enforce method inlining
//
//     RAJA_ALIGN_DATA(<variable>) - macro to express alignment of data,
//                              loop bounds, etc.
//
//     RAJA_SIMD - macro to express SIMD vectorization pragma to force
//                 loop vectorization
//
//     RAJA_ALIGNED_ATTR(<alignment>) - macro to express type or variable alignments
//

#if (defined(_WIN32) || defined(_WIN64)) && !defined(RAJA_WIN_STATIC_BUILD)
#ifdef RAJASHAREDDLL_EXPORTS
#define RAJASHAREDDLL_API __declspec(dllexport)
#else
#define RAJASHAREDDLL_API __declspec(dllimport)
#endif
#else
#define RAJASHAREDDLL_API
#endif

#if defined(RAJA_COMPILER_GNU)
#define RAJA_ALIGNED_ATTR(N) __attribute__((aligned(N)))
#else
#define RAJA_ALIGNED_ATTR(N) alignas(N)
#endif


#if defined(RAJA_COMPILER_INTEL)
//
// Configuration options for Intel compilers
//

#if defined (RAJA_ENABLE_FORCEINLINE_RECURSIVE)
#define RAJA_FORCEINLINE_RECURSIVE  RAJA_PRAGMA(forceinline recursive)
#else
#define RAJA_FORCEINLINE_RECURSIVE
#endif

#if defined (_WIN32)
#define RAJA_INLINE inline
#else
#define RAJA_INLINE inline  __attribute__((always_inline))
#endif

#define RAJA_UNROLL RAJA_PRAGMA(unroll)

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#define RAJA_ALIGN_DATA(d) d
#else
#define RAJA_ALIGN_DATA(d) __assume_aligned(d, RAJA::DATA_ALIGN)
#endif

#if defined(_OPENMP) && (_OPENMP >= 201307) && (__INTEL_COMPILER >= 1700)
#define RAJA_SIMD  RAJA_PRAGMA(omp simd)
#define RAJA_NO_SIMD RAJA_PRAGMA(novector)
#elif defined(_OPENMP) && (_OPENMP >= 201307) && (__INTEL_COMPILER < 1700)
#define RAJA_SIMD
#define RAJA_NO_SIMD RAJA_PRAGMA(novector)
#else
#define RAJA_SIMD RAJA_PRAGMA(simd)
#define RAJA_NO_SIMD RAJA_PRAGMA(novector)
#endif


#elif defined(RAJA_COMPILER_GNU)
//
// Configuration options for GNU compilers
//
#define RAJA_FORCEINLINE_RECURSIVE
#define RAJA_INLINE inline  __attribute__((always_inline))

#if !defined(__NVCC__)
#define RAJA_UNROLL RAJA_PRAGMA(GCC unroll 10000)
#else
#define RAJA_UNROLL RAJA_PRAGMA(unroll)
#endif

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#define RAJA_ALIGN_DATA(d) d
#else
#define RAJA_ALIGN_DATA(d) __builtin_assume_aligned(d, RAJA::DATA_ALIGN)
#endif

#if defined(_OPENMP) && (_OPENMP >= 201307)
#define RAJA_SIMD  RAJA_PRAGMA(omp simd)
#define RAJA_NO_SIMD
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && \
      ( ( (__GNUC__ == 4) && (__GNUC_MINOR__ == 9) ) || (__GNUC__ >= 5) )
#define RAJA_SIMD    RAJA_PRAGMA(GCC ivdep)
#define RAJA_NO_SIMD
#else
#define RAJA_SIMD
#define RAJA_NO_SIMD
#endif


#elif defined(RAJA_COMPILER_XLC)
//
// Configuration options for xlc compiler (i.e., bgq/sequoia).
//
#define RAJA_FORCEINLINE_RECURSIVE
#define RAJA_INLINE inline  __attribute__((always_inline))
#define RAJA_UNROLL 
// FIXME: alignx is breaking CUDA+xlc
#if defined(RAJA_ENABLE_CUDA)
#define RAJA_ALIGN_DATA(d) d
#else
#define RAJA_ALIGN_DATA(d) __alignx(RAJA::DATA_ALIGN, d)
#endif

#if defined(_OPENMP) && (_OPENMP >= 201307)
#define RAJA_SIMD  RAJA_PRAGMA(omp simd)
#define RAJA_NO_SIMD RAJA_PRAGMA(simd_level(0))
#else
#define RAJA_SIMD  RAJA_PRAGMA(simd_level(10))
#define RAJA_NO_SIMD RAJA_PRAGMA(simd_level(0))
#endif

// Detect altivec, but disable if NVCC is being used due to some bad interactions
#if defined(__ALTIVEC__) && (__ALTIVEC__ == 1) && !defined(__NVCC__)
#define RAJA_ALTIVEC
#endif


#elif defined(RAJA_COMPILER_CLANG)
//
// Configuration options for clang compilers
//
#define RAJA_FORCEINLINE_RECURSIVE
#define RAJA_INLINE inline  __attribute__((always_inline))
#define RAJA_UNROLL RAJA_PRAGMA(clang loop unroll(enable))


// note that neither nvcc nor Apple Clang compiler currently doesn't support
// the __builtin_assume_aligned attribute
#if defined(RAJA_ENABLE_CUDA) || defined(__APPLE__)
#define RAJA_ALIGN_DATA(d) d
#else
#define RAJA_ALIGN_DATA(d) __builtin_assume_aligned(d, RAJA::DATA_ALIGN)
#endif

#if defined(_OPENMP) && (_OPENMP >= 201307) && (__clang_major__ >= 4 )
#define RAJA_SIMD  RAJA_PRAGMA(omp simd)
#define RAJA_NO_SIMD RAJA_PRAGMA(clang loop vectorize(disable))
#else

// Clang 3.7 and later changed the "pragma clang loop vectorize" options
// Apple Clang compiler supports older options
#if ( ( (__clang_major__ >= 4 ) ||  (__clang_major__ >= 3 && __clang_minor__ > 7) ) && !defined(__APPLE__) )
#define RAJA_SIMD    RAJA_PRAGMA(clang loop vectorize(assume_safety))
#else
#define RAJA_SIMD    RAJA_PRAGMA(clang loop vectorize(enable))
#endif

#define RAJA_NO_SIMD  RAJA_PRAGMA(clang loop vectorize(disable))
#endif

// Detect altivec, but only seems to work since Clang 9
#if defined(__ALTIVEC__) && (__clang_major__ >= 9 ) && (__ALTIVEC__ == 1)
#define RAJA_ALTIVEC
#endif


// This is the same as undefined compiler, but squelches the warning message
#elif defined(RAJA_COMPILER_MSVC)

#define RAJA_FORCEINLINE_RECURSIVE
#define RAJA_INLINE inline
#define RAJA_ALIGN_DATA(d) d
#define RAJA_SIMD
#define RAJA_NO_SIMD
#define RAJA_UNROLL

#else

#pragma message("RAJA_COMPILER unknown, using default empty macros.")
#define RAJA_FORCEINLINE_RECURSIVE
#define RAJA_INLINE inline
#define RAJA_ALIGN_DATA(d) d
#define RAJA_SIMD
#define RAJA_NO_SIMD
#define RAJA_UNROLL

#endif

#define RAJA_HAVE_POSIX_MEMALIGN
/* #undef RAJA_HAVE_ALIGNED_ALLOC */
/* #undef RAJA_HAVE_MM_MALLOC */

//
//Creates a general framework for compiler alignment hints
//
// Example usage:
// double *a = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,10*sizeof(double));
// double *y = RAJA::align_hint

template<typename T>
RAJA_INLINE
T * align_hint(T * x)
{

#if defined(RAJA_COMPILER_XLC) && defined(RAJA_ENABLE_CUDA)
  return x;
#elif defined(RAJA_COMPILER_INTEL) || defined(RAJA_COMPILER_XLC)
  RAJA_ALIGN_DATA(x);
  return x;
#else
  return static_cast<T *>(RAJA_ALIGN_DATA(x));
#endif
}


}  // closing brace for RAJA namespace


#ifndef RAJA_UNROLL
#define RAJA_UNROLL
#endif

// If we're in CUDA device code, we can use the nvcc unroll pragma
#if defined(__CUDA_ARCH__) && defined(RAJA_CUDA_ACTIVE)
#undef RAJA_UNROLL
#define RAJA_UNROLL RAJA_PRAGMA(unroll)
#endif

#endif // closing endif for header file include guard
