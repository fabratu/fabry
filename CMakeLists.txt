cmake_minimum_required(VERSION 3.12)
project(fabry CXX)

find_package(MPI REQUIRED)

add_library(fabry SHARED lib/fabry.cpp)

list(APPEND CXX_FLAGS "-fpermissive")

set_target_properties(fabry PROPERTIES PUBLIC_HEADER "include/fabry/fabry.hpp;include/fabry/internal-base.hpp;include/fabry/internal-icb.hpp")

target_include_directories(fabry PUBLIC "${PROJECT_SOURCE_DIR}/include" "${MPI_CXX_INCLUDE_DIRS}")

target_link_libraries(fabry PRIVATE MPI::MPI_CXX)

target_compile_options(fabry PRIVATE ${CXX_FLAGS})

install(TARGETS fabry 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/fabry)

configure_file(fabry.pc "${PROJECT_BINARY_DIR}/fabry.pc" @ONLY)
INSTALL(FILES ${PROJECT_BINARY_DIR}/fabry.pc DESTINATION "lib/pkgconfig")
