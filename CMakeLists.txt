cmake_minimum_required(VERSION 3.1)

project(tausch)

# Default MPI compiler
SET(CMAKE_CXX_COMPILER "mpic++")

# Enable C++11 and optimisations
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# option to toggle OpenCL support
option(OPENCL "Enable OpenCL support" ON)

# find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# find OpenCL
if(OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OPENCL_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DTAUSCH_OPENCL)
endif(OPENCL)

# fine OpenMP
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Create library
add_library(tausch SHARED "tausch.h" "modules/tausch2d.cpp" "modules/tausch3d.cpp" "modules/ctausch2d.cpp")
add_library(tausch_static STATIC "tausch.h" "modules/tausch2d.cpp" "modules/tausch3d.cpp" "modules/ctausch2d.cpp")

# Install library
install(TARGETS tausch DESTINATION "lib")
install(TARGETS tausch_static DESTINATION "lib")

# Install library headers
install(FILES "tausch.h" DESTINATION "include/tausch")
install(FILES "modules/tausch2d.h" "modules/ctausch2d.h" "modules/tausch3d.h" DESTINATION "include/tausch/modules")

add_definitions(-DSOURCEDIR="${CMAKE_SOURCE_DIR}")
