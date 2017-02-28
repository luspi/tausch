cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_COMPILER "mpicc")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

project(csample)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

add_executable(csample "samplecpu.h" "samplegpu.h" "paramstruct.h" "main.c")

set(CMAKE_BUILD_TYPE "Debug")

target_link_libraries(csample ${MPI_LIBRARIES})
target_link_libraries(csample ${OpenCL_LIBRARIES})
target_link_libraries(csample "m")
target_link_libraries(csample "pthread")
target_link_libraries(csample "tausch")
target_link_libraries(csample "stdc++")
set_target_properties(csample PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
set_target_properties(csample PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")

