# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 2.8.11)
project (OpenCLTest)

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.

set(SRC_FILES
  core/buffer.cc
  core/context.cc
  core/error.cc
  core/kernel.cc
  core/platform.cc
  core/util.cc
)

set(CMAKE_CXX_FLAGS "-Wall -framework OpenCL -g -O3")

#add_executable(example ${SRC_FILES} example.cc)
add_executable(memcpy ${SRC_FILES} copy_benchmark.cc)
