cmake_minimum_required(VERSION 3.1)

project(tausch)

install(FILES "tausch.h" DESTINATION "include")

option(TESTING "Enable Unit Tests" ON)

if(TESTING)

    enable_testing()

    set(CMAKE_BUILD_TYPE "Debug")

    find_package(MPI REQUIRED)
    find_package(CUDA REQUIRED)
    find_package(OpenCL REQUIRED)

    # cpu -> cpu
    add_executable(testing_cpu2cpu testing/cpu2cpu.cpp)
    target_include_directories(testing_cpu2cpu PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_cpu2cpu "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_cpu2cpu "${MPI_CXX_LIBRARIES}")
    add_test(cpu2cpu testing_cpu2cpu)

    # cpu -> opencl
    add_executable(testing_cpu2ocl testing/cpu2ocl.cpp)
    target_include_directories(testing_cpu2ocl PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_cpu2ocl "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_cpu2ocl "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_cpu2ocl PUBLIC "${OpenCL_INCLUDE_DIR}")
    target_link_libraries(testing_cpu2ocl "${OpenCL_LIBRARY}")
    add_test(cpu2ocl testing_cpu2ocl)

    # cpu -> cuda
    add_executable(testing_cpu2cuda testing/cpu2cuda.cpp)
    target_include_directories(testing_cpu2cuda PUBLIC "${MPI_C_INCLUDE_DIRS}")
    target_link_libraries(testing_cpu2cuda "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_cpu2cuda "${MPI_CXX_LIBRARIES}")
    target_include_directories(testing_cpu2cuda PUBLIC "${CUDA_INCLUDE_DIRS}")
    target_link_libraries(testing_cpu2cuda "${CUDA_LIBRARIES}")
    add_test(cpu2cuda testing_cpu2cuda)

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
