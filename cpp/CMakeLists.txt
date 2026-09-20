cmake_minimum_required(VERSION 3.16)
project(football_cv_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library definition
add_library(football_cv_core STATIC
    src/geometry.cpp
    src/perspective.cpp
    src/camera_motion.cpp
)

target_include_directories(football_cv_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Compiler warnings
if(MSVC)
    target_compile_options(football_cv_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(football_cv_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Tests
option(BUILD_TESTS "Build C++ unit test suite" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
