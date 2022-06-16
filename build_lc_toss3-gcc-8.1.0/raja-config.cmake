#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#
#=== Usage ===================================================================
# This file allows RAJA to be automatically detected by other libraries
# using CMake.  To build with RAJA, you can do one of two things:
#
#   1. Set the RAJA_DIR environment variable to the root directory of the RAJA
#      installation.  If you loaded RAJA through a dotkit, this may already
#      be set, and RAJA will be autodetected by CMake.
#
#   2. Configure your project with this option:
#      -DRAJA_DIR=<RAJA install prefix>/share/
#
# If you have done either of these things, then CMake should automatically find
# and include this file when you call find_package(RAJA) from your
# CMakeLists.txt file.
#
#=== Components ==============================================================
#
# To link against these, just do, for example:
#
#   find_package(RAJA REQUIRED)
#   add_executable(foo foo.c)
#   target_link_libraries(foo RAJA)
#
# That's all!
#

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was RAJA-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if (NOT TARGET camp)
  include(CMakeFindDependencyMacro)
  find_dependency(camp REQUIRED PATHS "${PACKAGE_PREFIX_DIR}")
endif ()

include("${CMAKE_CURRENT_LIST_DIR}/RAJA.cmake")

check_required_components("RAJA")
