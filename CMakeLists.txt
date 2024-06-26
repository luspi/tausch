cmake_minimum_required(VERSION 3.19)

project(tausch)

# tests is enabled by default
option(TESTING "Enable Unit Tests" ON)
option(TEST_OPENCL "Enable OpenCL tests" OFF)
option(TEST_CUDA "Enable CUDA tests" OFF)
option(TEST_CUDA_AWARE "Enable tests for CUDA-aware MPI" OFF)
option(TEST_HIP "Enable HIP tests" OFF)
option(HIP_NVIDIA "Use NVIDIA backend" ON)
option(HIP_AMD "Use AMD backend" OFF)

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

add_library(ctausch SHARED ctausch.cpp)

if(TEST_CUDA)
    enable_language("CUDA")
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

if(TEST_OPENCL)
    find_package(OpenCL REQUIRED)
    add_library(ctauschocl SHARED ctausch.cpp)
    target_include_directories(ctauschocl PUBLIC "${OpenCL_INCLUDE_DIRS}")
    target_link_libraries(ctauschocl "${OpenCL_LIBRARIES}")
    target_compile_definitions(ctauschocl PRIVATE TAUSCH_OPENCL)
endif()

if(TEST_HIP)
    list(APPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip")
    find_package(HIP REQUIRED)
    include_directories("/opt/rocm/include")
    if(HIP_NVIDIA)
        add_definitions(-DHIP_NVIDIA)
        enable_language("CUDA")
        find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
    if(HIP_AMD)
        add_definitions(-DHIP_AMD)
    endif()
endif()

# build tests if enabled
if(TESTING)

    enable_testing()

    find_package(Catch2 3 REQUIRED)

    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DOMPI_SKIP_MPICXX -O0 -g -Wno-deprecated-declarations")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -DOMPI_SKIP_MPICXX -O0 -g -Wno-deprecated-declarations")

    # custom function to add mpi test
    function(add_mpi_test name senddevice recvdevice)

        separate_arguments(files_list UNIX_COMMAND "testing/main.cpp testing/rma.cpp testing/packunpack.cpp testing/randomaccess.cpp testing/empty.cpp")

        # each test is run with 1, 2, and 4 mpi ranks
        set(numprocs 1 2 4)

        add_executable(${name} ${files_list})
        target_link_libraries(${name} ${MPI_C_LIBRARIES})
        target_link_libraries(${name} ${MPI_CXX_LIBRARIES})

        target_link_libraries(${name} Catch2::Catch2)

        if(senddevice STREQUAL "cpu")
            target_compile_definitions(${name} PRIVATE "TEST_SEND_TAUSCH_CPU")
        endif()
        if(recvdevice STREQUAL "cpu")
            target_compile_definitions(${name} PRIVATE "TEST_RECV_TAUSCH_CPU")
        endif()

        if(senddevice STREQUAL "cuda" OR recvdevice STREQUAL "cuda")
            target_include_directories(${name} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
            target_link_libraries(${name} ${CUDART_LIBRARY})
            if(senddevice STREQUAL "cuda")
                target_compile_definitions(${name} PRIVATE "TEST_SEND_TAUSCH_CUDA")
            endif()
            if(recvdevice STREQUAL "cuda")
                target_compile_definitions(${name} PRIVATE "TEST_RECV_TAUSCH_CUDA")
            endif()
        endif()

        if(senddevice STREQUAL "opencl" OR recvdevice STREQUAL "opencl")
            target_include_directories(${name} PUBLIC "${OpenCL_INCLUDE_DIRS}")
            target_link_libraries(${name} "${OpenCL_LIBRARIES}")
            if(senddevice STREQUAL "opencl")
                target_compile_definitions(${name} PRIVATE "TEST_SEND_TAUSCH_OPENCL")
            endif()
            if(recvdevice STREQUAL "opencl")
                target_compile_definitions(${name} PRIVATE "TEST_RECV_TAUSCH_OPENCL")
            endif()
        endif()

        if(senddevice STREQUAL "hip" OR recvdevice STREQUAL "hip")
            target_include_directories(${name} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
            target_include_directories(${name} PRIVATE "${HIP_INCLUDE_DIRS}")
            target_link_libraries(${name} "${HIP_LIBRARIES}")
            target_link_libraries(${name} ${CUDART_LIBRARY})
            if(senddevice STREQUAL "hip")
                target_compile_definitions(${name} PRIVATE "TEST_SEND_TAUSCH_HIP")
            endif()
            if(recvdevice STREQUAL "hip")
                target_compile_definitions(${name} PRIVATE "TEST_RECV_TAUSCH_HIP")
            endif()
        endif()

        foreach(n IN LISTS numprocs)
            add_test(NAME ${name}_${n} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n} "./${name}")
        endforeach()

    endfunction(add_mpi_test)

    find_package(MPI COMPONENTS C REQUIRED)
    # add_mpi_test(capi_cpu "testing/ctausch/cpu.c" false false true false)
    add_mpi_test(cpu2cpu "cpu" "cpu")

    if(TEST_CUDA)
        add_mpi_test(cpu2cuda "cpu" "cuda")
        add_mpi_test(cuda2cpu "cuda" "cpu")
        add_mpi_test(cuda2cuda "cuda" "cuda")
    endif()

    if(TEST_HIP)
        add_mpi_test(cpu2hip "cpu" "hip")
        add_mpi_test(hip2cpu "hip" "cpu")
        add_mpi_test(hip2hip "hip" "hip")
    endif()

    if(TEST_OPENCL)
        add_mpi_test(cpu2ocl "cpu" "opencl")
        add_mpi_test(ocl2cpu "opencl" "cpu")
        add_mpi_test(ocl2ocl "opencl" "opencl")
    endif()

    if(TEST_CUDA AND TEST_HIP)
        add_mpi_test(cuda2hip "cuda" "hip")
        add_mpi_test(hip2cuda "hip" "cuda")
    endif()

    if(TEST_CUDA AND TEST_OPENCL)
        add_mpi_test(cuda2ocl "cuda" "opencl")
        add_mpi_test(ocl2cuda "opencl" "cuda")
    endif()

    if(TEST_HIP AND TEST_OPENCL)
        add_mpi_test(hip2ocl "hip" "opencl")
        add_mpi_test(ocl2hip "opencl" "hip")
    endif()

endif()
