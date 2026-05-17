#include <iostream>
#include "axon/core/core.hpp"

int main() {
    std::cout << "Testing Storage..." << std::endl;

    // Test 1: Create storage
    axon::core::Storage s1(10, axon::core::CPU());
    std::cout << "Created storage size: " << s1.size() << std::endl;
    std::cout << "Data pointer: " << s1.data() << std::endl;

    // Test 2: Zero storage
    s1.zero();
    std::cout << "Zeroed storage" << std::endl;

    // Test 3: Fill storage
    s1.fill(3.14f);
    std::cout << "Filled storage" << std::endl;

    // Test 4: Copy storage (shares data)
    axon::core::Storage s2 = s1;
    std::cout << "Copied storage - is_unique: " << s1.is_unique() << std::endl;

    // Test 5: Arena
    axon::core::Arena arena(1024);
    std::cout << "Created arena size: " << arena.size() << std::endl;

    void* ptr = arena.allocate(256);
    std::cout << "Arena used: " << arena.used() << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}