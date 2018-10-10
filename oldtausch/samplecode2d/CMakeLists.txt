cmake_minimum_required(VERSION 3.1)

SET(CMAKE_CXX_COMPILER "mpic++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -L/home/luspi/Studium/UIUC_Research_Assistant/LOCAL/lib")

project(sample2d)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_BUILD_TYPE "Debug")

option(OPENCL "Enable OpenCL support" ON)

option(DEBUGMSG "Display debug messages" OFF)

# find OpenCL
#if(OPENCL)
#    find_package(OpenCL REQUIRED)
#    include_directories(${OPENCL_INCLUDE_DIRS})
#    ADD_DEFINITIONS(-DOPENCL)
#endif(OPENCL)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(sample2d "sample.cpp" "main.cpp")

if(OPENCL)
    target_link_libraries(sample2d "OpenCL")
endif(OPENCL)

target_link_libraries(sample2d ${MPI_LIBRARIES})
target_link_libraries(sample2d "tausch")
target_link_libraries(sample2d "atomic")
set_target_properties(sample2d PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
set_target_properties(sample2d PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")

