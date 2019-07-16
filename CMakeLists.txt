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

    # cpu -> cpu

    add_mpi_test(cpu2cpu_1 "testing/cpu2cpu/main.cpp testing/cpu2cpu/derived.cpp testing/cpu2cpu/packunpack.cpp testing/cpu2cpu/empty.cpp testing/cpu2cpu/randomaccess.cpp" 1 false false)
    add_mpi_test(cpu2cpu_2 "testing/cpu2cpu/main.cpp testing/cpu2cpu/derived.cpp testing/cpu2cpu/packunpack.cpp testing/cpu2cpu/empty.cpp testing/cpu2cpu/randomaccess.cpp" 2 false false)
    add_mpi_test(cpu2cpu_4 "testing/cpu2cpu/main.cpp testing/cpu2cpu/derived.cpp testing/cpu2cpu/packunpack.cpp testing/cpu2cpu/empty.cpp testing/cpu2cpu/randomaccess.cpp" 4 false false)

    add_mpi_test(cpu2ocl_1 "testing/cpu2ocl/ocl.h testing/cpu2ocl/main.cpp testing/cpu2ocl/derived.cpp testing/cpu2ocl/packunpack.cpp testing/cpu2ocl/empty.cpp testing/cpu2ocl/randomaccess.cpp" 1 true false)
    add_mpi_test(cpu2ocl_2 "testing/cpu2ocl/ocl.h testing/cpu2ocl/main.cpp testing/cpu2ocl/derived.cpp testing/cpu2ocl/packunpack.cpp testing/cpu2ocl/empty.cpp testing/cpu2ocl/randomaccess.cpp" 2 true false)
    add_mpi_test(cpu2ocl_4 "testing/cpu2ocl/ocl.h testing/cpu2ocl/main.cpp testing/cpu2ocl/derived.cpp testing/cpu2ocl/packunpack.cpp testing/cpu2ocl/empty.cpp testing/cpu2ocl/randomaccess.cpp" 4 true false)

    add_mpi_test(cpu2cuda_1 "testing/cpu2cuda/main.cpp testing/cpu2cuda/derived.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 1 false true)
    add_mpi_test(cpu2cuda_2 "testing/cpu2cuda/main.cpp testing/cpu2cuda/derived.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 2 false true)
    add_mpi_test(cpu2cuda_4 "testing/cpu2cuda/main.cpp testing/cpu2cuda/derived.cpp testing/cpu2cuda/packunpack.cpp testing/cpu2cuda/empty.cpp testing/cpu2cuda/randomaccess.cpp" 4 false true)

    # opencl -> cpu
    add_executable(testing_ocl2cpu testing/ocl2cpu.cpp)
    target_include_directories(testing_ocl2cpu PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_ocl2cpu "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_ocl2cpu "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_ocl2cpu PUBLIC "${OpenCL_INCLUDE_DIR}")
    target_link_libraries(testing_ocl2cpu "${OpenCL_LIBRARY}")
    add_test(ocl2cpu testing_ocl2cpu)

    # opencl -> opencl
    add_executable(testing_ocl2ocl testing/ocl2ocl.cpp)
    target_include_directories(testing_ocl2ocl PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_ocl2ocl "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_ocl2ocl "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_ocl2ocl PUBLIC "${OpenCL_INCLUDE_DIR}")
    target_link_libraries(testing_ocl2ocl "${OpenCL_LIBRARY}")
    add_test(ocl2ocl testing_ocl2ocl)

    # opencl -> cuda
    add_executable(testing_ocl2cuda testing/ocl2cuda.cpp)
    target_include_directories(testing_ocl2cuda PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_ocl2cuda "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_ocl2cuda "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_ocl2cuda PUBLIC "${CUDA_INCLUDE_DIRS}")
    target_link_libraries(testing_ocl2cuda "${CUDA_LIBRARIES}")
    target_include_directories(testing_ocl2cuda PUBLIC "${OpenCL_INCLUDE_DIR}")
    target_link_libraries(testing_ocl2cuda "${OpenCL_LIBRARY}")
    add_test(ocl2cuda testing_ocl2cuda)

    # cuda -> cpu
    add_executable(testing_cuda2cpu testing/cuda2cpu.cpp)
    target_include_directories(testing_cuda2cpu PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_cuda2cpu "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_cuda2cpu "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_cuda2cpu PUBLIC "${CUDA_INCLUDE_DIRS}")
    target_link_libraries(testing_cuda2cpu "${CUDA_LIBRARIES}")
    add_test(cuda2cpu testing_cuda2cpu)

    # cuda -> opencl
    add_executable(testing_cuda2ocl testing/cuda2ocl.cpp)
    target_include_directories(testing_cuda2ocl PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_cuda2ocl "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_cuda2ocl "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_cuda2ocl PUBLIC "${CUDA_INCLUDE_DIRS}")
    target_link_libraries(testing_cuda2ocl "${CUDA_LIBRARIES}")
    target_include_directories(testing_cuda2ocl PUBLIC "${OpenCL_INCLUDE_DIR}")
    target_link_libraries(testing_cuda2ocl "${OpenCL_LIBRARY}")
    add_test(cuda2ocl testing_cuda2ocl)

    # cuda -> cuda
    add_executable(testing_cuda2cuda testing/cuda2cuda.cpp)
    target_include_directories(testing_cuda2cuda PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_cuda2cuda "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_cuda2cuda "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_cuda2cuda PUBLIC "${CUDA_INCLUDE_DIRS}")
    target_link_libraries(testing_cuda2cuda "${CUDA_LIBRARIES}")
    add_test(cuda2cuda testing_cuda2cuda)

endif()
