#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -ne 2 ]]; then
  echo
  echo "You must pass to the script a compiler version number for nvcc followed"
  echo "by a version number for xl. For example,"
  echo "    blueos_nvcc_xl.sh 11.1.1 2020.03.31"
  exit
fi

COMP_NVCC_VER=$1
COMP_XL_VER=$2

BUILD_SUFFIX=lc_blueos-nvcc${COMP_NVCC_VER}-xl${COMP_XL_VER}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/xl/xl-${COMP_XL_VER}/bin/xlc++_r \
  -DBLT_CXX_STD=c++11 \
  -C ../host-configs/lc-builds/blueos/nvcc_xl_X.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER} \
  -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER}/bin/nvcc \
  -DCUDA_ARCH=sm_70 \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
