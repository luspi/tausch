cmake_minimum_required(VERSION 3.1)

project(tausch)

# tests is enabled by default
option(TESTING "Enable Unit Tests" ON)
option(TEST_CUDA_AWARE "Enable tests for CUDA-aware MPI" OFF)

# documentation is enabled by default if doxygen found
find_package(Doxygen)
option(DOC "Build documentation" ${DOXYGEN_FOUND})

# install header file into include/tausch/ folder
install(FILES "tausch.h" DESTINATION "include/tausch")

# if doc is enabled, build doc
if(DOC)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

endif()

# build tests if enabled
if(TESTING)

    enable_testing()

    set(CMAKE_BUILD_TYPE "Debug")

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DOMPI_SKIP_MPICXX")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -DOMPI_SKIP_MPICXX")

    # custom function to add mpi test
    function(add_mpi_test name files ocl cuda)

        separate_arguments(files_list UNIX_COMMAND ${files})

        # each test is run with 1, 2, and 4 mpi ranks
        set(numprocs 1 2 4)

        add_executable(${name} ${files_list})
        target_link_libraries(${name} ${MPI_C_LIBRARIES})
        target_link_libraries(${name} ${MPI_CXX_LIBRARIES})
        if(ocl)
            target_include_directories(${name} PUBLIC "${OpenCL_INCLUDE_DIRS}")
            target_link_libraries(${name} "${OpenCL_LIBRARIES}")
        endif()
        if(cuda)
            target_include_directories(${name} PUBLIC "${CUDA_INCLUDE_DIRS}")
            target_link_libraries(${name} "${CUDA_LIBRARIES}")
        endif()
        foreach(n IN LISTS numprocs)
            add_test(NAME ${name}_${n} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 "./${name}")
        endforeach()

    endfunction(add_mpi_test)

    find_package(MPI COMPONENTS C REQUIRED)
    find_package(CUDA REQUIRED)
    find_package(OpenCL REQUIRED)

    # cpu -> *

    add_mpi_test(cpu2cpu "testing/cpu2cpu/main.cpp testing/cpu2cpu/derived.cpp testing/cpu2cpu/packunpack.cpp testing/cpu2cpu/empty.cpp testing/cpu2cpu/randomaccess.cpp" false false)
    add_mpi_test(cpu2ocl "testing/cpu2ocl/main.cpp testing/cpu2ocl/packunpack.cpp testing/cpu2ocl/empty.cpp testing/cpu2ocl/randomaccess.cpp" true false)
    add_mpi_test(cpu2cuda "testing/cpu2cuda/main.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" false true)

    # cuda -> *

    add_mpi_test(cuda2cpu "testing/cuda2cpu/main.cpp testing/cuda2cpu/packunpack.cpp testing/cuda2cpu/empty.cpp testing/cuda2cpu/randomaccess.cpp" false true)
    add_mpi_test(cuda2ocl "testing/cuda2ocl/main.cpp testing/cuda2ocl/packunpack.cpp testing/cuda2ocl/empty.cpp testing/cuda2ocl/randomaccess.cpp" true true)
    add_mpi_test(cuda2cuda "testing/cuda2cuda/main.cpp testing/cuda2cuda/packunpack.cpp testing/cuda2cuda/empty.cpp testing/cuda2cuda/randomaccess.cpp" false true)

    # ocl -> *

    add_mpi_test(ocl2cpu "testing/ocl2cpu/main.cpp testing/ocl2cpu/packunpack.cpp testing/ocl2cpu/empty.cpp testing/ocl2cpu/randomaccess.cpp" true false)
    add_mpi_test(ocl2ocl "testing/ocl2ocl/main.cpp testing/ocl2ocl/packunpack.cpp testing/ocl2ocl/empty.cpp testing/ocl2ocl/randomaccess.cpp" true false)
    add_mpi_test(ocl2cuda "testing/ocl2cuda/main.cpp testing/ocl2cuda/packunpack.cpp testing/ocl2cuda/empty.cpp testing/ocl2cuda/randomaccess.cpp" true true)

if(TEST_CUDA_AWARE)
    add_mpi_test(cudaaware "testing/cudaaware/main.cpp testing/cudaaware/cuda2cuda.cpp testing/cudaaware/cuda2cpu.cpp testing/cudaaware/cpu2cuda.cpp" false true)
endif()

endif()
