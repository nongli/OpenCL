cmake_minimum_required (VERSION 2.8.7)
project (OpenCLTest)

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.

set(CMAKE_CXX_FLAGS "-Wall -g -I.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework OpenCL")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -l OpenCL")
endif()

set(EXECUTABLE_OUTPUT_PATH "bin")
set(LIBRARY_OUTPUT_PATH "bin")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

add_library(Core STATIC
  core/buffer.cc
  core/context.cc
  core/error.cc
  core/kernel.cc
  core/platform.cc
  core/util.cc
)

add_executable(example examples/example.cc)
target_link_libraries(example Core ${OPENCL_LIBRARY})

add_executable(device_info examples/device_info.cc)
target_link_libraries(device_info Core ${OPENCL_LIBRARY})

add_executable(copy_benchmark examples/copy_benchmark.cc)
target_link_libraries(copy_benchmark Core ${OPENCL_LIBRARY})

add_executable(ambient_occlusion examples/ambient_occlusion.cc)
target_link_libraries(ambient_occlusion Core ${OPENCL_LIBRARY})
