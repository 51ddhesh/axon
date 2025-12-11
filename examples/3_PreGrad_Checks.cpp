#include "axon/tensor.hpp"
#include "axon/ops.hpp"
#include <iostream>

int main() {
    std::cout << "Pre-autograd checks\n";

    // 1. Create Matrix A (2x3)
    // 1 2 3
    // 4 5 6
    auto a = axon::Tensor::zeros({2, 3});
    float val = 1.0f;
    for(int i = 0; i < 2; i++) 
        for(int j = 0; j < 3; j++) a.at({i,j}) = val++;

    // 2. Transpose A -> A_T (3x2) - STRIDED
    // 1 4
    // 2 5
    // 3 6
    auto a_t = axon::transpose(a, 0, 1);
    
    std::cout << "Transposed Input A_T:\n";
    a_t.print();

    // 3. Matrix B (2x2) - Contiguous
    // 1 0
    // 0 1
    auto b = axon::Tensor::zeros({2, 2});
    b.at({0,0}) = 1; b.at({1,1}) = 1;

    // 4. MatMul (3x2) @ (2x2) -> (3x2)
    // Should work despite a_t being strided
    std::cout << "Running MatMul(Strided, Contiguous)...\n";
    auto c = axon::matmul(a_t, b);
    
    std::cout << "Result:\n";
    c.print();

    // 5. Test Sum (Reduction) on Strided Data
    std::cout << "Summing Strided Tensor A_T...\n";
    auto s = axon::sum(a_t);
    std::cout << "Sum: " << s.at({0}) << " (Expected 21)\n";

    return 0;
}