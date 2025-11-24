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


        std::cout << "\n4. Testing Contiguity..." << std::endl;
    // p is the permuted tensor from before (2, 2, 2) but strided (2, 4, 1)
    
    if (!p.is_contiguous()) {
        std::cout << "   SUCCESS: Permuted tensor is detected as Non-Contiguous." << std::endl;
    }

    // Reshaping p directly would have failed/produced garbage before.
    // Now it should trigger an internal .contiguous() copy.
    std::cout << "   Reshaping permuted tensor to (8,)..." << std::endl;
    Tensor flat = p.reshape({8});
    
    flat.print_meta();
    flat.print(); 
    // Should print: 1 5 3 7 2 6 4 8 (The order if you read p in row-major)
    // p(0,0,0)=1, p(0,0,1)=5, p(0,1,0)=3... wait let's check values logic.
    // Original T:
    // B0: 1, 2 | 3, 4
    // B1: 5, 6 | 7, 8
    
    // P (Row, Batch, Col):
    // R0: (B0,C0)=1, (B0,C1)=2 | (B1,C0)=5, (B1,C1)=6
    // R1: (B0,C0)=3, (B0,C1)=4 | (B1,C0)=7, (B1,C1)=8
    // Flattened P should read: 1, 2, 5, 6, 3, 4, 7, 8

    return 0;
}