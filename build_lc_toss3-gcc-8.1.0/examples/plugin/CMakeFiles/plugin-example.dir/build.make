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
include examples/plugin/CMakeFiles/plugin-example.dir/depend.make

# Include the progress variables for this target.
include examples/plugin/CMakeFiles/plugin-example.dir/progress.make

# Include the compile flags for this target's objects.
include examples/plugin/CMakeFiles/plugin-example.dir/flags.make

examples/plugin/CMakeFiles/plugin-example.dir/test-plugin.cpp.o: examples/plugin/CMakeFiles/plugin-example.dir/flags.make
examples/plugin/CMakeFiles/plugin-example.dir/test-plugin.cpp.o: ../examples/plugin/test-plugin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/plugin/CMakeFiles/plugin-example.dir/test-plugin.cpp.o"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plugin-example.dir/test-plugin.cpp.o -c /g/g20/cuneo3/testraja/RAJA/examples/plugin/test-plugin.cpp

examples/plugin/CMakeFiles/plugin-example.dir/test-plugin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plugin-example.dir/test-plugin.cpp.i"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g20/cuneo3/testraja/RAJA/examples/plugin/test-plugin.cpp > CMakeFiles/plugin-example.dir/test-plugin.cpp.i

examples/plugin/CMakeFiles/plugin-example.dir/test-plugin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plugin-example.dir/test-plugin.cpp.s"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g20/cuneo3/testraja/RAJA/examples/plugin/test-plugin.cpp -o CMakeFiles/plugin-example.dir/test-plugin.cpp.s

examples/plugin/CMakeFiles/plugin-example.dir/counter-plugin.cpp.o: examples/plugin/CMakeFiles/plugin-example.dir/flags.make
examples/plugin/CMakeFiles/plugin-example.dir/counter-plugin.cpp.o: ../examples/plugin/counter-plugin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object examples/plugin/CMakeFiles/plugin-example.dir/counter-plugin.cpp.o"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plugin-example.dir/counter-plugin.cpp.o -c /g/g20/cuneo3/testraja/RAJA/examples/plugin/counter-plugin.cpp

examples/plugin/CMakeFiles/plugin-example.dir/counter-plugin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plugin-example.dir/counter-plugin.cpp.i"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /g/g20/cuneo3/testraja/RAJA/examples/plugin/counter-plugin.cpp > CMakeFiles/plugin-example.dir/counter-plugin.cpp.i

examples/plugin/CMakeFiles/plugin-example.dir/counter-plugin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plugin-example.dir/counter-plugin.cpp.s"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && /usr/tce/packages/gcc/gcc-8.1.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /g/g20/cuneo3/testraja/RAJA/examples/plugin/counter-plugin.cpp -o CMakeFiles/plugin-example.dir/counter-plugin.cpp.s

# Object files for target plugin-example
plugin__example_OBJECTS = \
"CMakeFiles/plugin-example.dir/test-plugin.cpp.o" \
"CMakeFiles/plugin-example.dir/counter-plugin.cpp.o"

# External object files for target plugin-example
plugin__example_EXTERNAL_OBJECTS =

bin/plugin-example: examples/plugin/CMakeFiles/plugin-example.dir/test-plugin.cpp.o
bin/plugin-example: examples/plugin/CMakeFiles/plugin-example.dir/counter-plugin.cpp.o
bin/plugin-example: examples/plugin/CMakeFiles/plugin-example.dir/build.make
bin/plugin-example: lib/libRAJA.a
bin/plugin-example: lib/libcamp.a
bin/plugin-example: examples/plugin/CMakeFiles/plugin-example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../bin/plugin-example"
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/plugin-example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/plugin/CMakeFiles/plugin-example.dir/build: bin/plugin-example

.PHONY : examples/plugin/CMakeFiles/plugin-example.dir/build

examples/plugin/CMakeFiles/plugin-example.dir/clean:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin && $(CMAKE_COMMAND) -P CMakeFiles/plugin-example.dir/cmake_clean.cmake
.PHONY : examples/plugin/CMakeFiles/plugin-example.dir/clean

examples/plugin/CMakeFiles/plugin-example.dir/depend:
	cd /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /g/g20/cuneo3/testraja/RAJA /g/g20/cuneo3/testraja/RAJA/examples/plugin /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0 /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin /g/g20/cuneo3/testraja/RAJA/build_lc_toss3-gcc-8.1.0/examples/plugin/CMakeFiles/plugin-example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/plugin/CMakeFiles/plugin-example.dir/depend

