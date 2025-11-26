#include "axon/Tensor.hpp"

using namespace axon;

int main() {
    std::cout << "Axon MatMul Test" << std::endl;

    // A = (2, 3)
    // [ 1 2 3 ]
    // [ 4 5 6 ]
    Tensor A = {1, 2, 3, 4, 5, 6};
    A = A.reshape({2, 3});

    // B = (3, 2)
    // [ 7  8 ]
    // [ 9  1 ]
    // [ 2  3 ]
    Tensor B = {7, 8, 9, 1, 2, 3};
    B = B.reshape({3, 2});

    std::cout << "\n1. Forward Pass (A @ B)..." << std::endl;
    // C = (2, 2)
    // Row 0: 1*7 + 2*9 + 3*2 = 31
    // Row 0: 1*8 + 2*1 + 3*3 = 19
    // Row 1: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // Row 1: 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    Tensor C = A.matmul(B);
    
    C.print();
    // Expected: [ 31 19 85 55 ]

    std::cout << "\n2. Backward Pass..." << std::endl;
    // Let's create a scalar loss to backpropagate.
    // Loss = Sum(C)
    Tensor loss = C.sum();
    
    // Check Loss value: 31+19+85+55 = 190
    std::cout << "   Loss: " << loss.data_ptr()[0] << std::endl;
    
    loss.backward();
    
    std::cout << "   Backward executed." << std::endl;
    std::cout << "   A.grad (2x3):" << std::endl;
    // We can manually verify one gradient:
    // dLoss/dA_00 = (dLoss/dC_00 * B_00) + (dLoss/dC_01 * B_01)
    // Since dLoss/dC is 1 everywhere:
    // dL/dA_00 = B_00 + B_01 = 7 + 8 = 15.
    
    // To print nicely, we need to handle the shape. 
    // For now, let's just peek at raw data.
    const double* g = A.grad_ptr();
    std::cout << "   [ ";
    for(int i=0; i<6; ++i) std::cout << g[i] << " ";
    std::cout << "]" << std::endl;
    
    if (g[0] == 15.0) {
        std::cout << "   SUCCESS: Gradient check passed (A[0,0] == 15)." << std::endl;
    } else {
        std::cout << "   FAILURE: Gradient check failed." << std::endl;
    }

    return 0;
}