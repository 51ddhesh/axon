#include "axon3/tensor.hpp"
#include "axon3/ops.hpp"
#include <iostream>

int main() {
    std::cout << "Strided Memory in Axon\n";

    // Create a 2x3 Matrix
    // [ 0, 1, 2 ]
    // [ 3, 4, 5 ]
    auto a = axon::Tensor::zeros({2, 3});
    float val = 0.0f;
    for(int i=0; i<2; i++) {
        for(int j=0; j<3; j++) {
            a.at({i,j}) = val++;
        }
    }
    
    std::cout << "Original Matrix A:\n";
    a.print();

    // Transpose it -> 3x2 Matrix (STRIDED VIEW, NO COPY)
    // [ 0, 3 ]
    // [ 1, 4 ]
    // [ 2, 5 ]
    auto a_T = axon::transpose(a, 0, 1);
    
    std::cout << "Transposed A_T (Strided View):\n";
    a_T.print();

    // Create a contiguous 3x2 Matrix of 10s
    auto b = axon::Tensor::ones({3, 2});
    // Set to 10.0 manually
    for(size_t i = 0; i < b.numel(); i++) b.data_ptr()[i] = 10.0f;

    // ADD THEM (Strided + Contiguous)
    // This forces the Ops.cpp Dispatcher to use the Recursive Slow Path
    auto c = axon::add(a_T, b);

    std::cout << "Result (A_T + B):\n";
    // [ 10, 13 ]
    // [ 11, 14 ]
    // [ 12, 15 ]
    c.print();

    return 0;
}