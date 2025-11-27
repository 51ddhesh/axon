cmake_minimum_required(VERSION 3.20)
project(axon VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

include_directories(include)

add_library(axon
    src/Tensor.cpp
    src/TensorMath.cpp
    src/NN.cpp
    src/Optimizer.cpp
)

add_executable(ax main.cpp)

target_link_libraries(ax PRIVATE axon)