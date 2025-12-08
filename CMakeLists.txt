cmake_minimum_required(VERSION 3.20)
project(axon3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

include_directories(include)

add_library(axon3 
    src/tensor.cpp
    src/kernels.cpp
    src/ops.cpp
    src/autograd.cpp
    src/serialization.cpp
)

add_executable(ax main.cpp)

target_link_libraries(ax axon3)