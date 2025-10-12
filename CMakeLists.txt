cmake_minimum_required(VERSION 3.20)
project(axon VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

add_executable(ax
    ./main.cpp
    src/tensor_core.cpp
    src/tensor_ops.cpp
    src/tensor_utils.cpp
)

target_include_directories(ax PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)