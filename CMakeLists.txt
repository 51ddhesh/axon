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

add_executable(ax main.cpp)