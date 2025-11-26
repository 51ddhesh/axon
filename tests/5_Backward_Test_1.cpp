#include "axon/Tensor.hpp"

using namespace axon;

int main() {

    std::cout << "Broadcasting Backward (Bias Add)" << std::endl;
    
    // A: (2, 3)
    Tensor A = {1, 2, 3, 4, 5, 6};
    A = A.reshape({2, 3});
    
    // B: (1, 3) Bias
    Tensor B = {10, 20, 30};
    B = B.reshape({1, 3});
    
    // C = A + B
    // Rows become: [11, 22, 33] and [14, 25, 36]
    Tensor C = A + B;
    
    // Loss = Sum(C)
    // dLoss/dC = 1 everywhere.
    Tensor loss2 = C.sum();
    
    loss2.backward();
    
    // Check Gradients
    // dLoss/dA = 1 everywhere (Same shape as C)
    std::cout << "A.grad (should be all 1): " << A.grad_ptr()[0] << std::endl;
    
    // dLoss/dB:
    // C has 2 rows. Both rows depend on B.
    // So dLoss/dB = sum(dLoss/dC over rows) = 1 + 1 = 2.
    std::cout << "B.grad (should be all 2): ";
    for(int i=0; i<3; ++i) std::cout << B.grad_ptr()[i] << " ";
    std::cout << std::endl;
    
    if (B.grad_ptr()[0] == 2.0) {
        std::cout << "SUCCESS: Gradients summed over broadcasted dimension." << std::endl;
    } else {
        std::cout << "FAILURE: Broadcasting backward failed." << std::endl;
    }

    return 0;
}