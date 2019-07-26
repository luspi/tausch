cmake_minimum_required(VERSION 3.1)

project(tausch)

install(FILES "tausch.h" DESTINATION "include")

option(TESTING "Enable Unit Tests" ON)

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

    add_mpi_test(cpu2ocl_1 "testing/cpu2ocl/ocl.h testing/cpu2ocl/main.cpp testing/cpu2ocl/derived.cpp testing/cpu2ocl/packunpack.cpp testing/cpu2ocl/empty.cpp testing/cpu2ocl/randomaccess.cpp" 1 true false)
    add_mpi_test(cpu2ocl_2 "testing/cpu2ocl/ocl.h testing/cpu2ocl/main.cpp testing/cpu2ocl/derived.cpp testing/cpu2ocl/packunpack.cpp testing/cpu2ocl/empty.cpp testing/cpu2ocl/randomaccess.cpp" 2 true false)
    add_mpi_test(cpu2ocl_4 "testing/cpu2ocl/ocl.h testing/cpu2ocl/main.cpp testing/cpu2ocl/derived.cpp testing/cpu2ocl/packunpack.cpp testing/cpu2ocl/empty.cpp testing/cpu2ocl/randomaccess.cpp" 4 true false)

    add_mpi_test(cpu2cuda_1 "testing/cpu2cuda/main.cpp testing/cpu2cuda/derived.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 1 false true)
    add_mpi_test(cpu2cuda_2 "testing/cpu2cuda/main.cpp testing/cpu2cuda/derived.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 2 false true)
    add_mpi_test(cpu2cuda_4 "testing/cpu2cuda/main.cpp testing/cpu2cuda/derived.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 4 false true)

    # opencl -> *

    add_mpi_test(ocl2cpu_1 "testing/ocl2cpu/ocl.h testing/ocl2cpu/main.cpp testing/ocl2cpu/derived.cpp testing/ocl2cpu/packunpack.cpp testing/ocl2cpu/empty.cpp testing/ocl2cpu/randomaccess.cpp" 1 true false)
    add_mpi_test(ocl2cpu_2 "testing/ocl2cpu/ocl.h testing/ocl2cpu/main.cpp testing/ocl2cpu/derived.cpp testing/ocl2cpu/packunpack.cpp testing/ocl2cpu/empty.cpp testing/ocl2cpu/randomaccess.cpp" 2 true false)
    add_mpi_test(ocl2cpu_4 "testing/ocl2cpu/ocl.h testing/ocl2cpu/main.cpp testing/ocl2cpu/derived.cpp testing/ocl2cpu/packunpack.cpp testing/ocl2cpu/empty.cpp testing/ocl2cpu/randomaccess.cpp" 4 true false)

    add_mpi_test(ocl2ocl_1 "testing/ocl2ocl/ocl.h testing/ocl2ocl/main.cpp testing/ocl2ocl/packunpack.cpp testing/ocl2ocl/empty.cpp testing/ocl2ocl/randomaccess.cpp" 1 true false)
    add_mpi_test(ocl2ocl_2 "testing/ocl2ocl/ocl.h testing/ocl2ocl/main.cpp testing/ocl2ocl/packunpack.cpp testing/ocl2ocl/empty.cpp testing/ocl2ocl/randomaccess.cpp" 2 true false)
    add_mpi_test(ocl2ocl_4 "testing/ocl2ocl/ocl.h testing/ocl2ocl/main.cpp testing/ocl2ocl/packunpack.cpp testing/ocl2ocl/empty.cpp testing/ocl2ocl/randomaccess.cpp" 4 true false)

    add_mpi_test(ocl2cuda_1 "testing/ocl2cuda/ocl.h testing/ocl2cuda/main.cpp testing/ocl2cuda/packunpack.cpp testing/ocl2cuda/empty.cpp testing/ocl2cuda/randomaccess.cpp" 1 true true)
    add_mpi_test(ocl2cuda_2 "testing/ocl2cuda/ocl.h testing/ocl2cuda/main.cpp testing/ocl2cuda/packunpack.cpp testing/ocl2cuda/empty.cpp testing/ocl2cuda/randomaccess.cpp" 2 true true)
    add_mpi_test(ocl2cuda_4 "testing/ocl2cuda/ocl.h testing/ocl2cuda/main.cpp testing/ocl2cuda/packunpack.cpp testing/ocl2cuda/empty.cpp testing/ocl2cuda/randomaccess.cpp" 4 true true)

    # cuda -> *

    add_mpi_test(cuda2cpu_1 "testing/cuda2cpu/main.cpp testing/cuda2cpu/packunpack.cpp testing/cuda2cpu/empty.cpp testing/cuda2cpu/randomaccess.cpp" 1 false true)
    add_mpi_test(cuda2cpu_2 "testing/cuda2cpu/main.cpp testing/cuda2cpu/packunpack.cpp testing/cuda2cpu/empty.cpp testing/cuda2cpu/randomaccess.cpp" 2 false true)
    add_mpi_test(cuda2cpu_4 "testing/cuda2cpu/main.cpp testing/cuda2cpu/packunpack.cpp testing/cuda2cpu/empty.cpp testing/cuda2cpu/randomaccess.cpp" 4 false true)

    add_mpi_test(cuda2ocl_1 "testing/cuda2ocl/ocl.h testing/cuda2ocl/main.cpp testing/cuda2ocl/packunpack.cpp testing/cuda2ocl/empty.cpp testing/cuda2ocl/randomaccess.cpp" 1 true true)
    add_mpi_test(cuda2ocl_2 "testing/cuda2ocl/ocl.h testing/cuda2ocl/main.cpp testing/cuda2ocl/packunpack.cpp testing/cuda2ocl/empty.cpp testing/cuda2ocl/randomaccess.cpp" 2 true true)
    add_mpi_test(cuda2ocl_4 "testing/cuda2ocl/ocl.h testing/cuda2ocl/main.cpp testing/cuda2ocl/packunpack.cpp testing/cuda2ocl/empty.cpp testing/cuda2ocl/randomaccess.cpp" 4 true true)

    add_mpi_test(cuda2cuda_1 "testing/cuda2cuda/main.cpp testing/cuda2cuda/ondevice.cpp testing/cuda2cuda/packunpack.cpp testing/cuda2cuda/empty.cpp testing/cuda2cuda/randomaccess.cpp" 1 false true)
    add_mpi_test(cuda2cuda_2 "testing/cuda2cuda/main.cpp testing/cuda2cuda/packunpack.cpp testing/cuda2cuda/empty.cpp testing/cuda2cuda/randomaccess.cpp" 2 false true)
    add_mpi_test(cuda2cuda_4 "testing/cuda2cuda/main.cpp testing/cuda2cuda/packunpack.cpp testing/cuda2cuda/empty.cpp testing/cuda2cuda/randomaccess.cpp" 4 false true)

endif()
