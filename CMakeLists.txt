cmake_minimum_required(VERSION 3.1)

project(tausch)

install(FILES "tausch.h" DESTINATION "include")

option(TESTING "Enable Unit Tests" ON)

find_package(Doxygen)
option(DOC "Build documentation (make doc)" ${DOXYGEN_FOUND})

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

    function(add_mpi_test name files ocl cuda)

        separate_arguments(files_list UNIX_COMMAND ${files})

        set(numprocs 1 2 4)

        foreach(n IN LISTS numprocs)
            message(STATUS, "n=${n}")
            add_executable(${name}_${n} ${files_list})
            target_link_libraries(${name}_${n} ${MPI_C_LIBRARIES})
            target_link_libraries(${name}_${n} ${MPI_CXX_LIBRARIES})
            add_test(NAME ${name}_${n} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 "./${name}_${n}")
            if(ocl)
                target_include_directories(${name}_${n} PUBLIC "${OpenCL_INCLUDE_DIRS}")
                target_link_libraries(${name}_${n} "${OpenCL_LIBRARIES}")
            endif()
            if(cuda)
                target_include_directories(${name}_${n} PUBLIC "${CUDA_INCLUDE_DIRS}")
                target_link_libraries(${name}_${n} "${CUDA_LIBRARIES}")
            endif()
        endforeach()

    endfunction(add_mpi_test)

    find_package(MPI REQUIRED)
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

endif()
