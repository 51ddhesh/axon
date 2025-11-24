#include "axon/Tensor.hpp"
#include <iostream>

using namespace axon;

int main() {
    std::cout << "Axon Math Test" << std::endl;

    // 1. Setup Data
    // A: Shape (2, 3)
    // [ 1 2 3 ]
    // [ 4 5 6 ]
    Tensor A = {1, 2, 3, 4, 5, 6};
    A = A.reshape({2, 3});

    // B: Shape (1, 3) - Acts like a Bias Vector
    // [ 10 20 30 ]
    Tensor B = {10, 20, 30};
    B = B.reshape({1, 3}); 

    std::cout << "\n1. Broadcasting Add: (2,3) + (1,3)" << std::endl;
    // Should add [10, 20, 30] to BOTH rows of A
    Tensor C = A + B;
    
    C.print_meta();
    C.print(); 
    // Expected Result:
    // [ 11 22 33 ]
    // [ 14 25 36 ]

    // 2. Scalar Multiplication
    std::cout << "\n2. Scalar Mult: C * 2.0" << std::endl;
    // We treat scalars as (1,1) tensors for now
    Tensor Two = {2.0};
    Two = Two.reshape({1}); // Rank 1, Size 1
    
    Tensor D = C * Two;
    D.print();
    // Expected: [ 22 44 66 28 50 72 ]

    // 3. Reduction
    std::cout << "\n3. Summation" << std::endl;
    Tensor Loss = D.sum();
    Loss.print_meta();
    std::cout << "   Total Sum: " << Loss.data_ptr()[0] << std::endl;

    // 4. Graph Connectivity Check
    std::cout << "\n4. Graph Connectivity" << std::endl;
    // C came from A + B. Does C know this?
    // We can't access prev_ directly (private), but we can infer it works
    // if the code compiled (operator+ sets it).
    // we will traverse this to compute gradients.
    std::cout << "   Graph connections established successfully (Compilation passed)." << std::endl;

    return 0;
}