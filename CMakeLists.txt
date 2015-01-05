# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 2.8.11)
project (OpenCLTest)

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.

set(CMAKE_CXX_FLAGS "-Wall -framework OpenCL -g -I.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(EXECUTABLE_OUTPUT_PATH "bin")
set(LIBRARY_OUTPUT_PATH "bin")

add_library(Core STATIC
  core/buffer.cc
  core/context.cc
  core/error.cc
  core/kernel.cc
  core/platform.cc
  core/util.cc
)

add_executable(example examples/example.cc)
target_link_libraries(example Core)

add_executable(device_info examples/device_info.cc)
target_link_libraries(device_info Core)

add_executable(copy_benchmark examples/copy_benchmark.cc)
target_link_libraries(copy_benchmark Core)

add_executable(ambient_occlusion examples/ambient_occlusion.cc)
target_link_libraries(ambient_occlusion Core)
