/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Main RAJA header file.
 *
 *          This is the main header file to include in code that uses RAJA.
 *          It provides a single access point to all RAJA features by
 *          including other RAJA headers.
 *
 *          IMPORTANT: If changes are made to this file, note that contents
 *                     of some header files require that they are included
 *                     in the order found here.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_HPP
#define RAJA_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/Registry.hpp"


//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/teams.hpp"

//
// Generic templates to describe SIMD/SIMT registers and vectors
//
#include "RAJA/pattern/tensor.hpp"

//
// All platforms must support sequential execution.
//
#include "RAJA/policy/sequential.hpp"

//
// All platforms must support loop execution.
//
#include "RAJA/policy/loop.hpp"

//
// All platforms should support simd and vector execution.
//
#include "RAJA/policy/simd.hpp"
#include "RAJA/policy/tensor.hpp"

#if defined(RAJA_ENABLE_TBB)
#include "RAJA/policy/tbb.hpp"
#endif

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip.hpp"
#endif

#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)
#include "RAJA/policy/openmp_target.hpp"
#endif
#endif

#if defined(RAJA_ENABLE_DESUL_ATOMICS)
    #include "RAJA/policy/desul.hpp"
#endif

#include "RAJA/index/IndexSet.hpp"

//
// Strongly typed index class
//
#include "RAJA/index/IndexValue.hpp"


//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"

#include "RAJA/policy/MultiPolicy.hpp"


//
// Multidimensional layouts and views
//
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"
#include "RAJA/util/PermutedLayout.hpp"
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/View.hpp"


//
// View for sequences of objects
//
#include "RAJA/util/Span.hpp"

//
// zip iterator to iterator over sequences simultaneously
//
#include "RAJA/util/zip.hpp"

//
// Atomic operations support
//
#include "RAJA/pattern/atomic.hpp"

//
// Shared memory view patterns
//
#include "RAJA/util/LocalArray.hpp"

//
// Bit masking operators
//
#include "RAJA/util/BitMask.hpp"

//
// sort algorithms
//
#include "RAJA/util/sort.hpp"

//
// WorkPool, WorkGroup, WorkSite objects
//
#include "RAJA/policy/WorkGroup.hpp"
#include "RAJA/pattern/WorkGroup.hpp"

//
// Reduction objects
//
#include "RAJA/pattern/reduce.hpp"


//
// Synchronization
//
#include "RAJA/pattern/synchronize.hpp"

//
//////////////////////////////////////////////////////////////////////
//
// These contents of the header files included here define index set
// and segment execution methods whose implementations depend on
// programming model choice.
//
// The ordering of these file inclusions must be preserved since there
// are dependencies among them.
//
//////////////////////////////////////////////////////////////////////
//

#include "RAJA/index/IndexSetUtils.hpp"
#include "RAJA/index/IndexSetBuilders.hpp"

#include "RAJA/pattern/scan.hpp"

#if defined(RAJA_ENABLE_RUNTIME_PLUGINS)
#include "RAJA/util/PluginLinker.hpp"
#endif

#include "RAJA/pattern/sort.hpp"

namespace RAJA {
namespace expt{}
  // provide a RAJA::expt namespace for experimental work, but bring alias
  // it into RAJA so it doesn't affect user code
  using namespace expt;
}


#if 1


int test_main (int argc, char* argv[])
{
  double * A, * B, * C;

  if( argc != 0 ){
    A = *((double**)argv[0]);
    B = *((double**)argv[1]);
    C = *((double**)argv[2]);
  } else {
    double * volatile ptr;
    A = ptr;
    B = ptr;
    C = ptr;
  }

#if 0
  using vec_t = RAJA::expt::VectorRegister<double>;
  using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

  volatile int dummy = 0;

  auto aa   = RAJA::make_view<int, double>( A );

  auto bV   = RAJA::make_view<int, double>( B );

  auto cc   = RAJA::make_view<int, double>( C );

  auto vall = idx_t::all();

  constexpr int vsize = 4;

  // Vector Register
  cc    (idx_t::range(0,vsize))             = bV(0)   + aa  (idx_t::range(0,vsize))          ;
  //cc    (idx_t::range(vsize*1,vsize*2))             = bV(1)   + aa  (idx_t::range(vsize*1,vsize*2))          ;

#else


    using mat_t = RAJA::expt::RectMatrixRegister<double, RAJA::expt::RowMajorLayout, 16, 4, RAJA::avx2_register>;
    using row_t = RAJA::expt::RowIndex<int, mat_t>;
    using col_t = RAJA::expt::ColIndex<int, mat_t>;

   #if 1 
    auto aa   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_IJ,16,4>>( A );
    auto bV   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_I,16>>( B );
    auto cc   = RAJA::View<double, RAJA::StaticLayout<RAJA::PERM_IJ,16,4>>( C );
   #else
    auto aa   = RAJA::View<double, RAJA::Layout<2>>( A, 16, 4 );
    auto bV   = RAJA::View<double, RAJA::Layout<1>>( B, 16 );
    auto cc   = RAJA::View<double, RAJA::Layout<2>>( C, 16, 4 );
   #endif

    #if 0

    using vec_t = RAJA::expt::VectorRegister<double>;
    using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

    auto vall = idx_t::all();
    auto rall = row_t::all();


    constexpr int vsize = 4;

    using cpu_launch = RAJA::expt::seq_launch_t;
    //using gpu_launch = RAJA::expt::cuda_launch_t<false>;
    using launch_pol = RAJA::expt::LoopPolicy<cpu_launch>;  //, gpu_launch>;

    RAJA::expt::ExecPlace select_cpu_or_gpu = RAJA::expt::HOST;

    #endif

    auto call = col_t::all();


      #if 1
      // Slightly better matrix form
      cc( row_t::range(0,1)  , call )     =   bV(0)     +     aa( row_t::range(0,1)  , call );
      cc( row_t::range(1,2)  , call )     =   bV(1)     +     aa( row_t::range(1,2)  , call );
      cc( row_t::range(2,3)  , call )     =   bV(2)     +     aa( row_t::range(2,3)  , call );
      cc( row_t::range(3,4)  , call )     =   bV(3)     +     aa( row_t::range(3,4)  , call );
      cc( row_t::range(4,5)  , call )     =   bV(4)     +     aa( row_t::range(4,5)  , call );
      cc( row_t::range(5,6)  , call )     =   bV(5)     +     aa( row_t::range(5,6)  , call );
      cc( row_t::range(6,7)  , call )     =   bV(6)     +     aa( row_t::range(6,7)  , call );
      cc( row_t::range(7,8)  , call )     =   bV(7)     +     aa( row_t::range(7,8)  , call );
      cc( row_t::range(8,9)  , call )     =   bV(8)     +     aa( row_t::range(8,9)  , call );
      cc( row_t::range(9,10) , call )     =   bV(9)     +     aa( row_t::range(9,10) , call );
      cc( row_t::range(10,11), call )     =   bV(10)    +     aa( row_t::range(10,11), call );
      cc( row_t::range(11,12), call )     =   bV(11)    +     aa( row_t::range(11,12), call );
      cc( row_t::range(12,13), call )     =   bV(12)    +     aa( row_t::range(12,13), call );
      cc( row_t::range(13,14), call )     =   bV(13)    +     aa( row_t::range(13,14), call );
      cc( row_t::range(14,15), call )     =   bV(14)    +     aa( row_t::range(14,15), call );
      cc( row_t::range(15,16), call )     =   bV(15)    +     aa( row_t::range(15,16), call );
      #else
      // Vector Register
      cc    (idx_t::range(0,vsize))             = bV(0)   + aa  (idx_t::range(0,vsize))          ;
      cc    (idx_t::range(vsize*1,vsize*2))     = bV(1)   + aa  (idx_t::range(vsize*1,vsize*2))  ;
      cc    (idx_t::range(vsize*2,vsize*3))     = bV(2)   + aa  (idx_t::range(vsize*2,vsize*3))  ;
      cc    (idx_t::range(vsize*3,vsize*4))     = bV(3)   + aa  (idx_t::range(vsize*3,vsize*4))  ;
      cc    (idx_t::range(vsize*4,vsize*5))     = bV(4)   + aa  (idx_t::range(vsize*4,vsize*5))  ;
      cc    (idx_t::range(vsize*5,vsize*6))     = bV(5)   + aa  (idx_t::range(vsize*5,vsize*6))  ;
      cc    (idx_t::range(vsize*6,vsize*7))     = bV(6)   + aa  (idx_t::range(vsize*6,vsize*7))  ;
      cc    (idx_t::range(vsize*7,vsize*8))     = bV(7)   + aa  (idx_t::range(vsize*7,vsize*8))  ;
      cc    (idx_t::range(vsize*8,vsize*9))     = bV(8)   + aa  (idx_t::range(vsize*8,vsize*9))  ;
      cc    (idx_t::range(vsize*9,vsize*10))    = bV(9)   + aa  (idx_t::range(vsize*9,vsize*10)) ;
      cc    (idx_t::range(vsize*10,vsize*11))   = bV(10)  + aa  (idx_t::range(vsize*10,vsize*11));
      cc    (idx_t::range(vsize*11,vsize*12))   = bV(11)  + aa  (idx_t::range(vsize*11,vsize*12));
      cc    (idx_t::range(vsize*12,vsize*13))   = bV(12)  + aa  (idx_t::range(vsize*12,vsize*13));
      cc    (idx_t::range(vsize*13,vsize*14))   = bV(13)  + aa  (idx_t::range(vsize*13,vsize*14));
      cc    (idx_t::range(vsize*14,vsize*15))   = bV(14)  + aa  (idx_t::range(vsize*14,vsize*15));
      cc    (idx_t::range(vsize*15,vsize*16))   = bV(15)  + aa  (idx_t::range(vsize*15,vsize*16));
      #endif    


#endif


  return 0;
}

#endif

#endif  // closing endif for header file include guard
