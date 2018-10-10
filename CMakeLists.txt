cmake_minimum_required(VERSION 3.1)

project(tausch)

set(CMAKE_PROJECT_NAME "Tausch - A general halo exchange library")

# Default MPI compiler
SET(CMAKE_CXX_COMPILER "mpic++")

# option to toggle OpenCL support
option(OPENCL "Enable OpenCL support" ON)

# find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# find OpenCL
#if(OPENCL)

#    # Enable C++11 and optimisations
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")

#    find_package(OpenCL REQUIRED)
#    include_directories(${OPENCL_INCLUDE_DIRS})
##    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl_yes.h" "${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl.h" COPYONLY)

#else(not OPENCL)

    # Enable C++03 and optimisations
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++03 -O3 -g -Wall")

#    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl_no.h" "${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl.h" COPYONLY)

#endif(OPENCL)

# All headers and source files
set(tausch_headers "tausch.h" "modules/tauschdefs.h" "modules/tausch.h" "modules/tausch_cwc.h")
set(tausch_source "tausch.h" "modules/tauschdefs.h" "modules/tausch.h" "modules/tausch_cwc.h")

# Create library
add_library(tausch SHARED ${tausch_source})
add_library(tausch_static STATIC ${tausch_source})

set_target_properties(tausch PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(tausch_static PROPERTIES LINKER_LANGUAGE CXX)

# Install library
install(TARGETS tausch DESTINATION "lib")
install(TARGETS tausch_static DESTINATION "lib")

# Install library headers
install(FILES "tausch.h" DESTINATION "include/tausch")
install(FILES "modules/tausch.h" "modules/tausch_cwc.h" "modules/tauschdefs.h"
        DESTINATION "include/tausch/modules")


