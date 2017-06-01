cmake_minimum_required(VERSION 3.1)

SET(CMAKE_CXX_COMPILER "mpic++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

project(sample1d)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_BUILD_TYPE "Debug")

if(OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OPENCL_INCLUDE_DIRS})
endif(OPENCL)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

option(OPENCL "Enable OpenCL support" ON)

add_executable(sample1d "sample.cpp" "main.cpp")

# find OpenCL
if(OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OPENCL_INCLUDE_DIRS})
    target_link_libraries(sample1d "OpenCL")
    ADD_DEFINITIONS(-DOPENCL)
endif(OPENCL)

target_link_libraries(sample1d ${MPI_LIBRARIES})
target_link_libraries(sample1d "tausch")
set_target_properties(sample1d PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
set_target_properties(sample1d PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")

