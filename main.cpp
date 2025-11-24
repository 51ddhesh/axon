#include "axon/Tensor.hpp"
#include <iostream>
#include <cassert>

using namespace axon;

int main() {
    std::cout << "Axon Autograd Test (Simple)" << std::endl;

    // 1. Simple Addition (Same Shapes)
    // x = [1, 2, 3]
    Tensor x = {1.0, 2.0, 3.0};
    
    // y = [4, 5, 6]
    Tensor y = {4.0, 5.0, 6.0};

    // z = x + y = [5, 7, 9]
    Tensor z = x + y;
    
    // loss = sum(z) = 5+7+9 = 21
    Tensor loss = z.sum(); 

    // Forward pass check
    std::cout << "Loss: " << loss.data_ptr()[0] << std::endl; // 21

    // Backward pass
    // dLoss/dz = 1
    // dz/dx = 1, dz/dy = 1
    // -> dLoss/dx = 1
    loss.backward();

    std::cout << "Gradients for X: [ ";
    for(int i=0; i<3; ++i) std::cout << x.grad_ptr()[i] << " ";
    std::cout << "]" << std::endl;

    if (x.grad_ptr()[0] == 1.0) {
        std::cout << "SUCCESS: Gradient flowed back correctly." << std::endl;
    } else {
        std::cout << "FAILURE: Gradient is wrong." << std::endl;
    }

    return 0;
}