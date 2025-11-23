cmake_minimum_required(VERSION 3.20)
project(axon VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-O3 -Wall -Wextra)

include_directories(include)

add_executable(ax 
    main.cpp 
    src/Tensor.cpp
)