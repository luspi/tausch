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

option(USEMEMCPY "Use memcpy" ON)
option(USESTLCPY "Use std::copy" OFF)
if(USEMEMCPY)
    remove_definitions(-DUSESTLCPY)
    add_definitions(-DUSEMEMCPY)
else()
    remove_definitions(-DUSEMEMCPY)
endif()
if(USESTLCPY)
    remove_definitions(-DUSEMEMCPY)
    add_definitions(-DUSESTLCPY)
else()
    remove_definitions(-DUSESTLCPY)
endif()

# find OpenCL
if(OPENCL)

    # Enable C++11 and optimisations
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")

    find_package(OpenCL REQUIRED)
    include_directories(${OPENCL_INCLUDE_DIRS})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl_yes.h" "${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl.h" COPYONLY)

else(not OPENCL)

    # Enable C++03 and optimisations
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl_no.h" "${CMAKE_CURRENT_SOURCE_DIR}/modules/tausch_opencl.h" COPYONLY)

endif(OPENCL)

# All headers and source files
set(tausch_headers "tausch.h" "modules/tausch1d.h" "modules/tausch2d.h" "modules/tausch3d.h"
    "modules/ctausch/ctauschdouble.h" "modules/ctausch/ctauschfloat.h" "modules/ctausch/ctauschint.h" "modules/ctausch/ctauschunsignedint.h"
    "modules/tausch.h" "modules/tauschdefs.h" "modules/tausch_opencl_no.h" "modules/tausch_opencl_yes.h")
set(tausch_source "tausch.h" "modules/tausch1d.cpp" "modules/tausch2d.cpp" "modules/tausch3d.cpp"
    "modules/ctausch/ctauschdouble.cpp" "modules/ctausch/ctauschfloat.cpp" "modules/ctausch/ctauschint.cpp" "modules/ctausch/ctauschunsignedint.cpp"
    "modules/tausch.h" "modules/tauschdefs.h" "modules/tausch_opencl_no.h" "modules/tausch_opencl_yes.h")

# Set the Doxygen list
set(list_for_doxygen_headers "tausch.h" "modules/tausch1d.h" "modules/tausch2d.h" "modules/tausch3d.h" "modules/tausch.h" "modules/tauschdefs.h")
foreach(arg ${list_for_doxygen_headers})
    set(doxygen_headers "${doxygen_headers} ${CMAKE_CURRENT_SOURCE_DIR}/${arg}")
endforeach(arg ${list_for_doxygen_headers})

# Create library
add_library(tausch SHARED ${tausch_source})
add_library(tausch_static STATIC ${tausch_source})

# Install library
install(TARGETS tausch DESTINATION "lib")
install(TARGETS tausch_static DESTINATION "lib")

# Install library headers
install(FILES "tausch.h" DESTINATION "include/tausch")
install(FILES "modules/tausch1d.h" "modules/tausch2d.h" "modules/tausch3d.h" "modules/tausch.h" "modules/tausch_opencl.h" "modules/tauschdefs.h"
        DESTINATION "include/tausch/modules")
install(FILES "modules/ctausch/ctauschdouble.h" "modules/ctausch/ctauschfloat.h" "modules/ctausch/ctauschint.h" "modules/ctausch/ctauschunsignedint.h"
        DESTINATION "include/tausch/modules/ctausch")

# build documentation (if enabled)
option(DOC "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})
if(DOC)

    find_package(Doxygen)

    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.bin)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html DESTINATION share/doc/tausch OPTIONAL)
endif()
