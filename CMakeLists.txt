cmake_minimum_required(VERSION 2.8)

set(CXX "mpic++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread")

project(tausch)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE "Debug")

include_directories("include")

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(tausch "main.cpp" "tausch.cpp" "sample.cpp" "kernels.cl")

target_link_libraries(tausch ${MPI_LIBRARIES})
target_link_libraries(tausch ${OPENCL_LIBRARIES})
set_target_properties(tausch PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
set_target_properties(tausch PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} -lOpenCL")

add_definitions(-DSOURCEDIR="${CMAKE_SOURCE_DIR}")


