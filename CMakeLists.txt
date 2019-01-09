cmake_minimum_required(VERSION 3.1)

project(tausch)

set(CMAKE_PROJECT_NAME "Tausch - A general halo exchange library")

# Dummy stuff to make them show up in my IDE -> not actually needed!
set(tausch_files "tauschdefs.h" "tausch.h" "tausch_c2c.h" "tausch_c2g.h" "tausch_g2c.h" "tausch_g2g.h")
add_library(tausch SHARED ${tausch_files})
set_target_properties(tausch PROPERTIES LINKER_LANGUAGE CXX)

# Install library headers
install(FILES "tausch.h" "tausch_c2c.h" "tausch_c2g.h" "tausch_g2c.h" "tausch_g2g.h" "tauschdefs.h"
        DESTINATION "include/tausch")

option(TESTING "Enable Unit Tests" ON)

set(CMAKE_BUILD_TYPE "Debug")

if(TESTING)
    enable_testing()
    find_package(MPI REQUIRED)

    add_executable(testing_c2c testing/c2c.cpp)
    target_link_libraries(testing_c2c "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_c2c "${MPI_CXX_LIBRARIES}")
    add_test(testing_c2c testing_c2c)

    add_executable(testing_c2g testing/c2g.cpp)
    target_link_libraries(testing_c2g "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_c2g "${MPI_CXX_LIBRARIES}")
    target_link_libraries(testing_c2g "OpenCL")
    add_test(testing_c2g testing_c2g)

    add_executable(testing_g2c testing/g2c.cpp)
    target_link_libraries(testing_g2c "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_g2c "${MPI_CXX_LIBRARIES}")
    target_link_libraries(testing_g2c "OpenCL")
    add_test(testing_g2c testing_g2c)

    add_executable(testing_g2g testing/g2g.cpp)
    target_link_libraries(testing_g2g "${MPI_C_LIBRARIES}")
    target_link_libraries(testing_g2g "${MPI_CXX_LIBRARIES}")
    target_link_libraries(testing_g2g "OpenCL")
    add_test(testing_g2g testing_g2g)

endif(TESTING)

