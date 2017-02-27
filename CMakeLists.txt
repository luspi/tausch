cmake_minimum_required(VERSION 3.1)

project(tausch)

# Default MPI compiler
SET(CMAKE_CXX_COMPILER "mpic++")

# Enable C++11 and optimisations
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# find OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

# fine OpenMP
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Create library
add_library(tausch SHARED "tausch.cpp")
add_library(tausch_static STATIC "tausch.cpp")
add_library(ctausch SHARED "ctausch.c")
add_library(ctausch_static STATIC "ctausch.c")
set_source_files_properties("ctausch.c" PROPERTIES LANGUAGE CXX)

# Install library
install(TARGETS tausch DESTINATION "lib")
install(TARGETS tausch_static DESTINATION "lib")
install(TARGETS ctausch DESTINATION "lib")
install(TARGETS ctausch_static DESTINATION "lib")

# Install library headers
install(FILES "tausch.h" DESTINATION "include")
install(FILES "ctausch.h" DESTINATION "include")
