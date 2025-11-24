#include "axon/Tensor.hpp"
#include <cassert>

using namespace axon;

int main() {
    std::cout << "Axon Memory Architecture Test" << std::endl;

    // Create a 2D tensor A (2 x 3)
    std::cout << "1. Tensor A (2 X 3)" << std::endl;
    Tensor a = Tensor({2, 3}, 5.0);
    a.print_meta();
    a.print();

    // Initializer List
    std::cout << "2. Create a Tensor B from {1, 2, 3, 4, 5}" << std::endl;
    Tensor b = {1, 2, 3, 4, 5};
    b.print_meta();
    b.print();

    // Test Handle Copy Semantics (Shallow Copy)
    std::cout << "3. Creating Tensor C as a copy of A..." << std::endl;
    Tensor c = a; 
    
    std::cout << "   [A] "; a.print_meta();
    std::cout << "   [C] "; c.print_meta();

    // Check if they share the same storage address
    if (a.data_ptr() == c.data_ptr()) {
        std::cout << "   SUCCESS: A and C point to the same memory address." << std::endl;
    } else {
        std::cerr << "   FAILURE: A and C have different memory!" << std::endl;
        return 1;
    }

    // Test Modification
    std::cout << "4. Modifying C[0] = 99.0..." << std::endl;
    c.data_ptr()[0] = 99.0; // Direct pointer access

    std::cout << "   Checking A[0] (should be 99.0): " << a.data_ptr()[0] << std::endl;

    if (a.data_ptr()[0] == 99.0) {
        std::cout << "   SUCCESS: Modification in C reflected in A." << std::endl;
    } else {
        std::cerr << "   FAILURE: Storage was not shared." << std::endl;
        return 1;
    }

    std::cout << "All tests completed successfully" << std::endl;
    return 0;
}
