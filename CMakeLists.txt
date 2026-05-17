cmake_minimum_required(VERSION 3.20)
project(axon VERSION 4.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(CUDA_ENABLED "Enable CUDA support" ON)

if(CUDA_ENABLED)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math -mavx2 -mfma)
endif()

include_directories(include)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_library(axon STATIC)
target_sources(axon PRIVATE
    src/core/storage.cpp
)

target_link_libraries(axon PUBLIC Threads::Threads)

if(CUDA_ENABLED)
    target_link_libraries(axon PUBLIC CUDA::cublas CUDA::cudart)
endif()

add_executable(ax main.cpp)
target_link_libraries(ax axon)

enable_testing()

add_subdirectory(tests)