cmake_minimum_required(VERSION 2.8)

project(tauschdriver)

set(CMAKE_CXX_COMPILER "mpic++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g -Wall -pedantic")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#find_package(OpenMP)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(${PROJECT_NAME} "main.cpp" "tauschdriver.cpp")

target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME} "tausch")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
