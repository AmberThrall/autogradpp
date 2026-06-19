cmake_minimum_required(VERSION 3.14)

project(autogradpp
    VERSION 0.1.0
    DESCRIPTION "A header-only C++ autograd library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(AUTOGRADPP_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(AUTOGRADPP_IS_TOP_LEVEL ON)
endif()

option(AUTOGRADPP_BUILD_EXAMPLES "Build autogradpp examples" ${AUTOGRADPP_IS_TOP_LEVEL})

add_library(autogradpp INTERFACE)
add_library(autogradpp::autogradpp ALIAS autogradpp)

target_include_directories(autogradpp INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(autogradpp INTERFACE cxx_std_17)

if(AUTOGRADPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
