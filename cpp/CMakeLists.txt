cmake_minimum_required(VERSION 3.16)
project(football_cv_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Suppress deprecation warnings from external subprojects
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# Library definition
add_library(football_cv_core STATIC
    src/geometry.cpp
    src/perspective.cpp
    src/camera_motion.cpp
)

set_target_properties(football_cv_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(football_cv_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ------------------------------------------------------------------------------
# ONNX Runtime Integration
# ------------------------------------------------------------------------------
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime C++ inference backend" ON)
if(ENABLE_ONNXRUNTIME)
    set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-win-x64-1.20.1")
    if(NOT EXISTS "${ORT_ROOT}")
        message(STATUS "ONNX Runtime not found in third_party; fetching...")
        include(FetchContent)
        FetchContent_Declare(
            onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-1.20.1.zip
        )
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        set(ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
    endif()

    if(EXISTS "${ORT_ROOT}/include")
        message(STATUS "Configuring ONNX Runtime from: ${ORT_ROOT}")
        target_include_directories(football_cv_core PUBLIC "${ORT_ROOT}/include")
        if(MSVC)
            target_link_libraries(football_cv_core PUBLIC "${ORT_ROOT}/lib/onnxruntime.lib")
            set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll" CACHE INTERNAL "ONNX Runtime DLL path")
        else()
            target_link_libraries(football_cv_core PUBLIC "${ORT_ROOT}/lib/libonnxruntime.so")
            set(ORT_DLL "${ORT_ROOT}/lib/libonnxruntime.so" CACHE INTERNAL "ONNX Runtime shared lib path")
        endif()

        target_sources(football_cv_core PRIVATE src/inference/onnx_detector.cpp)
        target_compile_definitions(football_cv_core PUBLIC FOOTBALL_CV_HAS_ONNX=1)
    else()
        message(WARNING "ONNX Runtime headers not found at ${ORT_ROOT}/include; building without ONNX support.")
    endif()
endif()

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

# pybind11 Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python native extension via pybind11" ON)
if(BUILD_PYTHON_BINDINGS)
    # Prefer project virtual environment Python if available
    if(NOT DEFINED Python3_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe")
        set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe" CACHE FILEPATH "Python interpreter" FORCE)
    endif()

    # Hint pybind11 cmake directory from active python environment
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND AND NOT pybind11_DIR)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _detected_pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_detected_pybind11_dir)
            list(APPEND CMAKE_PREFIX_PATH "${_detected_pybind11_dir}")
        endif()
    endif()

    # Tell pybind11 to use modern FindPython3
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found locally; fetching via FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    if(pybind11_FOUND)
        pybind11_add_module(_core MODULE
            bindings/python_module.cpp
        )

        target_link_libraries(_core PRIVATE football_cv_core)
        target_include_directories(_core PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

        if(MSVC)
            target_compile_options(_core PRIVATE /W3 /permissive-)
        else()
            target_compile_options(_core PRIVATE -Wall -Wextra)
        endif()

        # Output compiled extension directly to src/football_cv/
        set(PYTHON_EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/football_cv")
        set_target_properties(_core PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${PYTHON_EXT_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PYTHON_EXT_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PYTHON_EXT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${PYTHON_EXT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PYTHON_EXT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PYTHON_EXT_DIR}"
        )

        # Copy onnxruntime.dll alongside _core native extension
        if(ORT_DLL AND EXISTS "${ORT_DLL}")
            add_custom_command(TARGET _core POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ORT_DLL}"
                    "${PYTHON_EXT_DIR}"
                COMMENT "Copying onnxruntime.dll to ${PYTHON_EXT_DIR}"
            )
        endif()
    endif()
endif()
