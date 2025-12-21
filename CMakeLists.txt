cmake_minimum_required(VERSION 3.20)
project(axon LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math -mavx2 -mfma)
endif()

set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(include)

add_library(axon
    src/tensor.cpp
    src/kernels.cpp
    src/ops.cpp
    src/autograd.cpp
    src/serialization.cpp
)

target_link_libraries(axon PRIVATE CUDA::cublas CUDA::cudart)

add_executable(ax main.cpp)

target_link_libraries(ax axon)