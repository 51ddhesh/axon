#include "axon/Tensor.hpp"
#include <iostream>
#include <vector>

using namespace axon;

int main() {
    std::cout << "Axon N-Dim Support" << std::endl;

    // 1. Create a 3D Tensor (2, 2, 2)
    Tensor t = {
        1, 2, 3, 4, 
        5, 6, 7, 8   
    };
    t = t.reshape({2, 2, 2});
    
    std::cout << "1. Original 3D Tensor (2,2,2):" << std::endl;
    t.print_meta();
    // Value at (1, 0, 1) -> 2nd batch, 1st row, 2nd col -> Value 6
    std::cout << "   Value at (1, 0, 1): " << t({1, 0, 1}) << std::endl;

    // 2. Permute Axes (Transpose)
    std::cout << "\n2. Permuting to (Row, Batch, Col) [1, 0, 2]..." << std::endl;
    Tensor p = t.permute({1, 0, 2});
    p.print_meta();

    // Verify p(0, 1, 1).
    // In permuted view: Row 0, Batch 1, Col 1.
    // In original view: Batch 1, Row 0, Col 1 -> Value 6.
    std::cout << "   Value at p(0, 1, 1) [Should be 6]: " << p({0, 1, 1}) << std::endl;

    // 3. Verify Zero-Copy
    if (t.data_ptr() == p.data_ptr()) {
        std::cout << "   SUCCESS: Storage address is identical." << std::endl;
    }

    return 0;
}