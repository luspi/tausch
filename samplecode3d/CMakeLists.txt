cmake_minimum_required(VERSION 3.1)

SET(CMAKE_CXX_COMPILER "mpic++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g")

project(sample3d)

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

add_executable(sample3d "sample.cpp" "main.cpp")

# find OpenCL
if(OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OPENCL_INCLUDE_DIRS})
    target_link_libraries(sample3d "OpenCL")
    ADD_DEFINITIONS(-DOPENCL)
endif(OPENCL)

target_link_libraries(sample3d ${MPI_LIBRARIES})
target_link_libraries(sample3d "tausch")
target_link_libraries(sample3d "atomic")
set_target_properties(sample3d PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
set_target_properties(sample3d PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")

