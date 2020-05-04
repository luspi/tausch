cmake_minimum_required(VERSION 3.1)

project(tausch)

install(FILES "tausch.h" DESTINATION "include")

option(TESTING "Enable Unit Tests" ON)

find_package(Doxygen)
option(DOC "Build documentation" ${DOXYGEN_FOUND})

if(DOC)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()

if(TESTING)

    enable_testing()

    set(CMAKE_BUILD_TYPE "Debug")

    function(add_mpi_test name files num_mpi_proc ocl cuda)
        separate_arguments(files_list UNIX_COMMAND ${files})
        add_executable(${name} ${files_list})
        target_link_libraries(${name} ${MPI_C_LIBRARIES})
        target_link_libraries(${name} ${MPI_CXX_LIBRARIES})
        add_test(NAME ${name} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${num_mpi_proc} "./${name}")
        if(ocl)
            target_include_directories(${name} PUBLIC "${OpenCL_INCLUDE_DIRS}")
            target_link_libraries(${name} "${OpenCL_LIBRARIES}")
        endif()
        if(cuda)
            target_include_directories(${name} PUBLIC "${CUDA_INCLUDE_DIRS}")
            target_link_libraries(${name} "${CUDA_LIBRARIES}")
        endif()
    endfunction(add_mpi_test)

    find_package(MPI REQUIRED)
    find_package(CUDA REQUIRED)
    find_package(OpenCL REQUIRED)

    # cpu -> *

    add_mpi_test(cpu2cpu_1 "testing/cpu2cpu/main.cpp testing/cpu2cpu/derived.cpp testing/cpu2cpu/packunpack.cpp testing/cpu2cpu/empty.cpp testing/cpu2cpu/randomaccess.cpp" 1 false false)
    add_mpi_test(cpu2cpu_2 "testing/cpu2cpu/main.cpp testing/cpu2cpu/derived.cpp testing/cpu2cpu/packunpack.cpp testing/cpu2cpu/empty.cpp testing/cpu2cpu/randomaccess.cpp" 2 false false)
    add_mpi_test(cpu2cpu_4 "testing/cpu2cpu/main.cpp testing/cpu2cpu/derived.cpp testing/cpu2cpu/packunpack.cpp testing/cpu2cpu/empty.cpp testing/cpu2cpu/randomaccess.cpp" 4 false false)

    add_mpi_test(cpu2cuda_1 "testing/cpu2cuda/main.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 1 false true)
    add_mpi_test(cpu2cuda_2 "testing/cpu2cuda/main.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 2 false true)
    add_mpi_test(cpu2cuda_4 "testing/cpu2cuda/main.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 4 false true)

#    # cuda -> *

    add_mpi_test(cuda2cpu_1 "testing/cuda2cpu/main.cpp testing/cuda2cpu/packunpack.cpp testing/cuda2cpu/empty.cpp testing/cuda2cpu/randomaccess.cpp" 1 false true)
    add_mpi_test(cuda2cpu_2 "testing/cuda2cpu/main.cpp testing/cuda2cpu/packunpack.cpp testing/cuda2cpu/empty.cpp testing/cuda2cpu/randomaccess.cpp" 2 false true)
    add_mpi_test(cuda2cpu_4 "testing/cuda2cpu/main.cpp testing/cuda2cpu/packunpack.cpp testing/cuda2cpu/empty.cpp testing/cuda2cpu/randomaccess.cpp" 4 false true)

    add_mpi_test(cuda2cuda_1 "testing/cuda2cuda/main.cpp testing/cuda2cuda/packunpack.cpp testing/cuda2cuda/empty.cpp testing/cuda2cuda/randomaccess.cpp" 1 false true)
    add_mpi_test(cuda2cuda_2 "testing/cuda2cuda/main.cpp testing/cuda2cuda/packunpack.cpp testing/cuda2cuda/empty.cpp testing/cuda2cuda/randomaccess.cpp" 2 false true)
    add_mpi_test(cuda2cuda_4 "testing/cuda2cuda/main.cpp testing/cuda2cuda/packunpack.cpp testing/cuda2cuda/empty.cpp testing/cuda2cuda/randomaccess.cpp" 4 false true)

endif()
