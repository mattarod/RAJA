# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake

# The command to remove a file.
RM = /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /g/g20/cuneo3/testraja/RAJA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0

# Include any dependencies generated for this target.
include examples/CMakeFiles/tut_reductions.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/tut_reductions.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/tut_reductions.dir/flags.make

examples/CMakeFiles/tut_reductions.dir/tut_reductions.cpp.o: examples/CMakeFiles/tut_reductions.dir/flags.make
examples/CMakeFiles/tut_reductions.dir/tut_reductions.cpp.o: ../examples/tut_reductions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/tut_reductions.dir/tut_reductions.cpp.o"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tut_reductions.dir/tut_reductions.cpp.o -c /g/g20/cuneo3/testraja/RAJA/examples/tut_reductions.cpp

examples/CMakeFiles/tut_reductions.dir/tut_reductions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tut_reductions.dir/tut_reductions.cpp.i"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g20/cuneo3/testraja/RAJA/examples/tut_reductions.cpp > CMakeFiles/tut_reductions.dir/tut_reductions.cpp.i

examples/CMakeFiles/tut_reductions.dir/tut_reductions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tut_reductions.dir/tut_reductions.cpp.s"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g20/cuneo3/testraja/RAJA/examples/tut_reductions.cpp -o CMakeFiles/tut_reductions.dir/tut_reductions.cpp.s

# Object files for target tut_reductions
tut_reductions_OBJECTS = \
"CMakeFiles/tut_reductions.dir/tut_reductions.cpp.o"

# External object files for target tut_reductions
tut_reductions_EXTERNAL_OBJECTS =

bin/tut_reductions: examples/CMakeFiles/tut_reductions.dir/tut_reductions.cpp.o
bin/tut_reductions: examples/CMakeFiles/tut_reductions.dir/build.make
bin/tut_reductions: lib/libRAJA.a
bin/tut_reductions: lib/libcamp.a
bin/tut_reductions: examples/CMakeFiles/tut_reductions.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/tut_reductions"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tut_reductions.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/tut_reductions.dir/build: bin/tut_reductions

.PHONY : examples/CMakeFiles/tut_reductions.dir/build

examples/CMakeFiles/tut_reductions.dir/clean:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && $(CMAKE_COMMAND) -P CMakeFiles/tut_reductions.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/tut_reductions.dir/clean

examples/CMakeFiles/tut_reductions.dir/depend:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /g/g20/cuneo3/testraja/RAJA /g/g20/cuneo3/testraja/RAJA/examples /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/CMakeFiles/tut_reductions.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/tut_reductions.dir/depend

