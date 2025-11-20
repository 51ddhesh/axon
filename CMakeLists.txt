cmake_minimum_required(VERSION 3.20)
project(axon VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

add_library(axon
    src/core/tensor_core.cpp
    src/core/tensor_ops.cpp
    src/core/tensor_utils.cpp
    src/nn/activations.cpp
    src/nn/linear.cpp
    src/nn/sequential.cpp
    src/nn/loss.cpp
    src/math/comparisons.cpp
    src/math/unary_ops.cpp
    src/math/binary_ops.cpp
    src/nn/autograd/backward.cpp
)

target_include_directories(axon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(ax main.cpp)

target_link_libraries(ax PRIVATE axon)
