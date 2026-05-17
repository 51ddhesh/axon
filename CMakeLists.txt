cmake_minimum_required(VERSION 3.20)
project(axon LANGUAGES CXX)

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

if(CUDA_ENABLED)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include_directories(include)

set(AXON_SOURCES
    src/tensor.cpp
    src/cpu_kernels.cpp
    src/ops.cpp
    src/autograd.cpp
    src/serialization.cpp
)

if(CUDA_ENABLED)
    list(APPEND AXON_SOURCES
        src/cuda_kernels.cu
        src/gpu_kernels.cu
    )
endif()

add_library(axon ${AXON_SOURCES})

if(CUDA_ENABLED)
    target_link_libraries(axon PRIVATE CUDA::cublas CUDA::cudart)
endif()

add_executable(ax main.cpp)
target_link_libraries(ax axon)