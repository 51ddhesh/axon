#include "axon3/tensor.hpp"
#include "axon3/ops.hpp"
#include <iostream>

int main() {
    std::cout << "Axon Broadcasting\n";

    // 1. Matrix (2x3)
    // [ 0, 0, 0 ]
    // [ 0, 0, 0 ]
    axon::Tensor mat = axon::Tensor::zeros({2, 3});
    std::cout << "Matrix: \n";
    mat.print();

    // 2. Vector (1x3) - Represents a Bias
    // [ 1, 2, 3 ]
    auto bias = axon::Tensor::zeros({1, 3});
    bias.at({0, 0}) = 1.0f;
    bias.at({0, 1}) = 2.0f;
    bias.at({0, 2}) = 3.0f;

    std::cout << "Bias Vector:\n";
    bias.print();

    // 3. Add (Broadcasting happens here!)
    // The bias (1,3) is implicitly expanded to (2,3)
    auto result = axon::add(mat, bias);

    std::cout << "Result (Matrix + Bias):\n";
    result.print(); 
    
    // Expected:
    // [ 1, 2, 3 ]
    // [ 1, 2, 3 ]

    return 0;
}