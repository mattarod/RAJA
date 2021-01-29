//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <math.h> 
#include <typeinfo>
#include <cxxabi.h>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
#include "array-reductions2.hpp"

//#ifdef HAVE_CXA_DEMANGLE
#if 1
const char* demangle(const char* name)
{
     char buf[1024];
     size_t size=1024;
     int status;
     char* res = abi::__cxa_demangle (name,
                                      buf,
                                      &size,
                                      &status);
     return res;
}
#else
const char* demangle(const char* name)
{
    return name;
}
#endif

int main(int argc, char* argv[])
{
  // Test Parameters
  constexpr int NUM_NODES = 30;
  int NUM_NODE_LISTS = 2;
  int NUM_PAIRS = 50000;
  if (argc > 1)
    NUM_PAIRS = std::atoi(argv[1]);

  RAJA::RangeSegment np_range(0, NUM_PAIRS);

  // Initialize NodePair list
  auto pairlist = generatePairList(NUM_NODES, NUM_PAIRS);
  auto pairlist2d = generate2DPairList(NUM_NODES, NUM_NODE_LISTS, NUM_PAIRS);

  // Calculate Solution
  auto nodeDataSolution = generateSolution(NUM_NODES, pairlist);

  //using EXEC_POL = RAJA::seq_exec;
  //using REDUCE_POL = RAJA::seq_reduce;
  using EXEC_POL = RAJA::omp_parallel_for_exec;
  using REDUCE_POL = RAJA::omp_reduce;

  using BASE_T = double;
  using REDUCESUM_T = RAJA::ReduceSum<REDUCE_POL, BASE_T>;

  data_dim<double, 3> data3(0, 3, 2, 1);
  data_dim<double, 2> data2(1, 7, 3);

  data3[1][1][0] = 15;
  data2[3][0] = 12;

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 2; j++){
      for(int k = 0; k < 1; k++){
        std::cout << data3[i][j][k] << " ";
      }
    }
  }
  std::cout << "\n";

  for(int j = 0; j < 7; j++){
    for(int k = 0; k < 3; k++){
      std::cout << data2[j][k] << " ";
    }
  }
  std::cout << "\n";
#if 0
  // --------------------------------------------------------------------------------
  // Run Vector-of-Type Reduction Example
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning 1D Vector-of-Type (VoT_t) Reducer example ...\n";

  VoT_t<1, BASE_T> VoT_nodeData(NUM_NODES);

  RAJA::ChronoTimer type_timer;
  type_timer.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist[ i ].first;
    int j_idx = pairlist[ i ].second;

    BASE_T& i_data = VoT_nodeData.at( i_idx );
    BASE_T& j_data = VoT_nodeData.at( j_idx );
    i_data += j_idx;
    j_data += i_idx;
  });
  type_timer.stop();
  checkResults(nodeDataSolution, VoT_nodeData, type_timer);
#endif

}




// --------------------------------------------------------------------------------
// Helper Function Definitions
// --------------------------------------------------------------------------------

pairlist_t generatePairList(const int n_nodes, const int n_pairs){
  srand(0);
  pairlist_t pl;
  for (auto i = 0; i < n_pairs; i++)
    pl.push_back(std::make_pair(rand() % n_nodes, rand() % n_nodes));
  return pl;
}

pairlist_t generate2DPairList(const int n_nodes, const int n_node_lists, const int n_pairs){
  srand(0);
  pairlist_t pl;
  for (auto i = 0; i < n_pairs; i++)
    pl.push_back(std::make_pair(rand() % n_node_lists, rand() % n_nodes));
  return pl;
}

std::vector<double> generateSolution(const int n_nodes, const pairlist_t pl){
  std::vector<double> solution(n_nodes);
  for (size_t i = 0; i < pl.size(); i++){
    int i_idx = pl[ i ].first;
    int j_idx = pl[ i ].second;
    double& i_data = solution[ i_idx ];
    double& j_data = solution[ j_idx ];
    i_data += j_idx;
    j_data += i_idx;
  }
  return solution;
}

std::vector<std::vector<double>> generate2DSolution(const int n_node_lists, const int n_nodes, const pairlist_t pl){
  std::vector<std::vector<double>> solution;
  for (int i = 0; i < n_node_lists; i++) {
    solution.push_back(std::vector<double>(n_nodes));
  }

  for (size_t i = 0; i < pl.size(); i++){
    int i_idx = pl[ i ].first;
    int j_idx = pl[ i ].second;
    double& _data = solution[i_idx][j_idx];
    _data += i_idx + j_idx;
  }
  return solution;
}

template<typename T1, typename T2>
void checkResults(const  T1& solution, const T2& test, const RAJA::ChronoTimer& timer){

  std::cout << "\tTime : " << timer.elapsed() << "\n";

  //if (correctness)
  if (test == solution)
    std::cout<< "\tPASSED !\n";
  else
    std::cout<< "\tFAILED !\n";
}
