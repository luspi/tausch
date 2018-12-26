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


