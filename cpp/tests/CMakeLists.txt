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
)

target_link_libraries(football_cv_tests
    PRIVATE
        football_cv_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(football_cv_tests)
