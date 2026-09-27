cmake_minimum_required(VERSION 3.20)
project(tinyinfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release flags: -O3 -march=native as per plan
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native)
endif()

# Engine library (exclude main.cpp)
file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(FILTER ENGINE_SOURCES EXCLUDE REGEX ".*main\\.cpp$")
add_library(tinyinfer_lib ${ENGINE_SOURCES})
target_include_directories(tinyinfer_lib PUBLIC include)

# Main CLI
add_executable(tinyinfer src/main.cpp)
target_link_libraries(tinyinfer PRIVATE tinyinfer_lib)

# GoogleTest via FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
if(TEST_SOURCES)
  add_executable(tinyinfer_tests ${TEST_SOURCES})
  target_link_libraries(tinyinfer_tests PRIVATE tinyinfer_lib gtest_main)
  include(GoogleTest)
  gtest_discover_tests(tinyinfer_tests)
endif()

# OpenMP for Phase 7
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(tinyinfer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
