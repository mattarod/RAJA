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
include examples/CMakeFiles/resource-runtime-teams.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/resource-runtime-teams.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/resource-runtime-teams.dir/flags.make

examples/CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.o: examples/CMakeFiles/resource-runtime-teams.dir/flags.make
examples/CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.o: ../examples/resource-runtime-teams.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.o"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.o -c /g/g20/cuneo3/testraja/RAJA/examples/resource-runtime-teams.cpp

examples/CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.i"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g20/cuneo3/testraja/RAJA/examples/resource-runtime-teams.cpp > CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.i

examples/CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.s"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g20/cuneo3/testraja/RAJA/examples/resource-runtime-teams.cpp -o CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.s

# Object files for target resource-runtime-teams
resource__runtime__teams_OBJECTS = \
"CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.o"

# External object files for target resource-runtime-teams
resource__runtime__teams_EXTERNAL_OBJECTS =

bin/resource-runtime-teams: examples/CMakeFiles/resource-runtime-teams.dir/resource-runtime-teams.cpp.o
bin/resource-runtime-teams: examples/CMakeFiles/resource-runtime-teams.dir/build.make
bin/resource-runtime-teams: lib/libRAJA.a
bin/resource-runtime-teams: lib/libcamp.a
bin/resource-runtime-teams: examples/CMakeFiles/resource-runtime-teams.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/resource-runtime-teams"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/resource-runtime-teams.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/resource-runtime-teams.dir/build: bin/resource-runtime-teams

.PHONY : examples/CMakeFiles/resource-runtime-teams.dir/build

examples/CMakeFiles/resource-runtime-teams.dir/clean:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples && $(CMAKE_COMMAND) -P CMakeFiles/resource-runtime-teams.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/resource-runtime-teams.dir/clean

examples/CMakeFiles/resource-runtime-teams.dir/depend:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /g/g20/cuneo3/testraja/RAJA /g/g20/cuneo3/testraja/RAJA/examples /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/CMakeFiles/resource-runtime-teams.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/resource-runtime-teams.dir/depend

