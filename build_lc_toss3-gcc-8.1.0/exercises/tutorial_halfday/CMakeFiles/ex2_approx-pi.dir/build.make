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
include exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/depend.make

# Include the progress variables for this target.
include exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/progress.make

# Include the compile flags for this target's objects.
include exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/flags.make

exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.o: exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/flags.make
exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.o: ../exercises/tutorial_halfday/ex2_approx-pi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.o"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/exercises/tutorial_halfday && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.o -c /g/g20/cuneo3/testraja/RAJA/exercises/tutorial_halfday/ex2_approx-pi.cpp

exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.i"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/exercises/tutorial_halfday && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g20/cuneo3/testraja/RAJA/exercises/tutorial_halfday/ex2_approx-pi.cpp > CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.i

exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.s"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/exercises/tutorial_halfday && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g20/cuneo3/testraja/RAJA/exercises/tutorial_halfday/ex2_approx-pi.cpp -o CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.s

# Object files for target ex2_approx-pi
ex2_approx__pi_OBJECTS = \
"CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.o"

# External object files for target ex2_approx-pi
ex2_approx__pi_EXTERNAL_OBJECTS =

bin/ex2_approx-pi: exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/ex2_approx-pi.cpp.o
bin/ex2_approx-pi: exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/build.make
bin/ex2_approx-pi: lib/libRAJA.a
bin/ex2_approx-pi: lib/libcamp.a
bin/ex2_approx-pi: exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/ex2_approx-pi"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/exercises/tutorial_halfday && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ex2_approx-pi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/build: bin/ex2_approx-pi

.PHONY : exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/build

exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/clean:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/exercises/tutorial_halfday && $(CMAKE_COMMAND) -P CMakeFiles/ex2_approx-pi.dir/cmake_clean.cmake
.PHONY : exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/clean

exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/depend:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /g/g20/cuneo3/testraja/RAJA /g/g20/cuneo3/testraja/RAJA/exercises/tutorial_halfday /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/exercises/tutorial_halfday /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : exercises/tutorial_halfday/CMakeFiles/ex2_approx-pi.dir/depend

