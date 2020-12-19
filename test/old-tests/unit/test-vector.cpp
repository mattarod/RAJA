//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic simd/simt vector operations
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"
#include "RAJA_gtest.hpp"

using VectorTestTypes = ::testing::Types<

  // Test automatically wrapped types, since the specific register
  // implementations are tested elsewhere
  //RAJA::VectorRegister<double>,
  RAJA::VectorRegister<double>
  >;


template <typename Policy>
class VectorTest : public ::testing::Test
{
protected:

  VectorTest() = default;
  virtual ~VectorTest() = default;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }

};
TYPED_TEST_SUITE_P(VectorTest);


TYPED_TEST_P(VectorTest, GetSet)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;

  element_t A[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)(i*2);
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load array A as vector
    vector_t vec;
    vec.load_packed_n(&A[0], N);

    // check get operations
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(vec.get(i), (element_t)(i*2));
    }

    // check set operations
    for(camp::idx_t i = 0;i < N;++ i){
      vec.set(i, (element_t)(i+1));
    }
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(vec.get(i), (element_t)(i+1));
    }

  }
}

TYPED_TEST_P(VectorTest, MinMaxSumDot)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;

  element_t A[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load array A as vector
    vector_t vec;
    vec.load_packed_n(&A[0], N);

    // check min
    ASSERT_SCALAR_EQ(vec.min(N), (element_t)0);

    // check max
    ASSERT_SCALAR_EQ(vec.max(N), (element_t)(N-1));

    // compute expected values
    element_t ex_sum(0);
    element_t ex_dot(0);
    for(camp::idx_t i = 0;i < N;++ i){
      ex_sum += A[i];
      ex_dot += A[i]*A[i];
    }

    // check sum
    ASSERT_SCALAR_EQ(vec.sum(), ex_sum);

    // check dot
    ASSERT_SCALAR_EQ(vec.dot(vec), ex_dot);

  }
}


TYPED_TEST_P(VectorTest, FmaFms)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;

  element_t A[vector_t::s_num_elem];
  element_t B[vector_t::s_num_elem];
  element_t C[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
    B[i] = (element_t)i*2;
    C[i] = (element_t)i*3;
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load arrays as vectors
    vector_t vec_A;
    vec_A.load_packed_n(&A[0], N);

    vector_t vec_B;
    vec_B.load_packed_n(&B[0], N);

    vector_t vec_C;
    vec_C.load_packed_n(&C[0], N);


    // check FMA (A*B+C)

    vector_t fma = vec_A.fused_multiply_add(vec_B, vec_C);
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fma.get(i), A[i]*B[i]+C[i]);
    }

    // check FMS (A*B-C)
    vector_t fms = vec_A.fused_multiply_subtract(vec_B, vec_C);
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fms.get(i), A[i]*B[i]-C[i]);
    }

  }
}


TYPED_TEST_P(VectorTest, ForallVectorRef1d)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;


  size_t N = 10*vector_t::s_num_elem;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
    N += (size_t)(100*NO_OPT_RAND);


  element_t *A = new element_t[N];
  element_t *B = new element_t[N];
  element_t *C = new element_t[N];
  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<double, RAJA::Layout<1>> X(A, N);
  RAJA::View<double, RAJA::Layout<1>> Y(B, N);
  RAJA::View<double, RAJA::Layout<1>> Z(C, N);


  auto all = RAJA::VectorIndex<int, vector_t>::all();

  Z[all] = 3 + (X[all]*(5/Y[all])) + 9;
  /*
  RAJA::forall<RAJA::vector_exec<vector_t>>(RAJA::TypedRangeSegment<int>(0, N),
      [=](RAJA::VectorIndex<int, vector_t> i)
  {
    Z[i] = 3+(X[i]*(5/Y[i]))+9;
  });*/


  for(size_t i = 0;i < N;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }

  delete[] A;
  delete[] B;
  delete[] C;
}


TYPED_TEST_P(VectorTest, ForallVectorRef2d)
{
  using vector_t = TypeParam;
  using index_t = ptrdiff_t;

  using element_t = typename vector_t::element_type;


  index_t N = 3*vector_t::s_num_elem;
  index_t M = 4*vector_t::s_num_elem;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
  if(!vector_t::s_is_fixed){
    N += (size_t)(10*NO_OPT_RAND);
    M += (size_t)(10*NO_OPT_RAND);
  }

  element_t *A = new element_t[N*M];
  element_t *B = new element_t[N*M];
  element_t *C = new element_t[N*M];
  for(index_t i = 0;i < N*M; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<double, RAJA::Layout<2>> X(A, N, M);
  RAJA::View<double, RAJA::Layout<2>> Y(B, N, M);
  RAJA::View<double, RAJA::Layout<2>> Z(C, N, M);

  //
  // Make sure the indexing is working as expected, and that the
  // View returns a vector object
  //

  ASSERT_EQ(A, X(0, RAJA::VectorIndex<index_t, vector_t>(0, 1)).get_pointer());
  ASSERT_EQ(A+M, X(1, RAJA::VectorIndex<index_t, vector_t>(0, 1)).get_pointer());
  ASSERT_EQ(A+1, X(0, RAJA::VectorIndex<index_t, vector_t>(1, 1)).get_pointer());

  using policy_t =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0>
        >
      >;


  auto all = RAJA::VectorIndex<index_t, vector_t>::all();

  RAJA::kernel<policy_t>(
      RAJA::make_tuple(RAJA::TypedRangeSegment<index_t>(0, N)),

      [=](index_t i)
  {
    Z(i,all) = 3+(X(i,all)*(5/Y(i,all)))+9;
  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }


  //
  // Test with forall
  //

  RAJA::forall<RAJA::loop_exec>(RAJA::TypedRangeSegment<index_t>(0, N),
      [=](index_t i){


    Z(i,all) = 3+(X(i,all)*(5/Y(i,all)))+9;


  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }





  delete[] A;
  delete[] B;
  delete[] C;
}


REGISTER_TYPED_TEST_SUITE_P(VectorTest, GetSet, MinMaxSumDot, FmaFms, ForallVectorRef1d, ForallVectorRef2d);

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, VectorTest, VectorTestTypes);
