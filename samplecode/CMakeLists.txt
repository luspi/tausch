cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_COMPILER "mpic++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC")

project(sample)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(sample "sample.cpp" "main.cpp")

target_link_libraries(sample ${MPI_LIBRARIES})
target_link_libraries(sample ${OpenCL_LIBRARIES})
target_link_libraries(sample "tausch")
set_target_properties(sample PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
set_target_properties(sample PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")

