cmake_minimum_required(VERSION 2.8)

project(tauschdriver)

set(CMAKE_CXX_COMPILER "mpic++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../modules/tausch_opencl_no.h" "${CMAKE_CURRENT_SOURCE_DIR}/../modules/tausch_opencl.h" COPYONLY)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(${PROJECT_NAME} "main.cpp" "tauschdriver.cpp" "../modules/tausch1d.cpp"  "../modules/tausch2d.cpp"  "../modules/tausch3d.cpp")

target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
