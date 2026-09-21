include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(football_cv_tests
    test_geometry.cpp
    test_perspective.cpp
    test_camera_motion.cpp
    test_onnx_detector.cpp
)

target_link_libraries(football_cv_tests
    PRIVATE
        football_cv_core
        GTest::gtest_main
)

# Copy onnxruntime.dll into test output directory for CTest execution
if(ORT_DLL AND EXISTS "${ORT_DLL}")
    add_custom_command(TARGET football_cv_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DLL}"
            "$<TARGET_FILE_DIR:football_cv_tests>"
        COMMENT "Copying onnxruntime.dll for CTest runner"
    )
endif()

include(GoogleTest)
gtest_discover_tests(football_cv_tests)
