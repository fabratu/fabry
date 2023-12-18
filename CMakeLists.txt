cmake_minimum_required(VERSION 3.12)
project(fabry CXX)

option(FABRY_USE_CACHE "Use MPI-caching with CLaMPI." OFF)

find_package(MPI REQUIRED)

add_library(fabry SHARED lib/fabry.cpp)

list(APPEND CXX_FLAGS "-fpermissive")

set_target_properties(fabry PROPERTIES PUBLIC_HEADER "include/fabry/fabry.hpp;include/fabry/internal-base.hpp;include/fabry/internal-icb.hpp")

target_include_directories(fabry PUBLIC "${PROJECT_SOURCE_DIR}/include" "${MPI_CXX_INCLUDE_DIRS}")

target_link_libraries(fabry PRIVATE MPI::MPI_CXX)

target_compile_options(fabry PRIVATE ${CXX_FLAGS})

# Find additional caching lib (non cmake, no pkg-config)
if (FABRY_USE_CACHE)
        message("CLAMPI DIR: ${CLAMPI_PREFIX}")
        find_library(CLAMPI_LIBRARY
                NAMES clampi libclampi
                HINTS "${CLAMPI_PREFIX}/lib"
                REQUIRED
        )
        find_path(CLAMPI_INCLUDE_DIRS NAMES clampi.h PATHS "${CLAMPI_PREFIX}/include" REQUIRED)

        target_include_directories(fabry PUBLIC "${CLAMPI_INCLUDE_DIRS}")

        target_link_libraries(fabry PRIVATE ${CLAMPI_LIBRARY})
endif()

install(TARGETS fabry 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/fabry)

configure_file(fabry.pc "${PROJECT_BINARY_DIR}/fabry.pc" @ONLY)
INSTALL(FILES ${PROJECT_BINARY_DIR}/fabry.pc DESTINATION "lib/pkgconfig")
