#include "axon/tensor.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace axon;

int main() {
    std::cout << "[TEST] Serialization...\n";

    std::string path = "test_model.bin";

    // 1. Create a "Model" (List of Tensors)
    Tensor w = Tensor::ones({2, 2});
    Tensor b = Tensor::zeros({2});
    
    // Modify values
    w.at({0, 0}) = 0.5f;
    b.at({1}) = 9.9f;

    std::vector<Tensor> params = {w, b};

    // 2. Save
    save_model(params, path);

    // 3. Create a fresh model (initialized differently)
    Tensor w2 = Tensor::zeros({2, 2});
    Tensor b2 = Tensor::ones({2});
    std::vector<Tensor> params2 = {w2, b2};

    // 4. Load
    load_model(params2, path);

    // 5. Verify
    // w2 should match w
    assert(w2.at({0,0}) == 0.5f);
    assert(w2.at({1,1}) == 1.0f); // Was 1.0 in w

    // b2 should match b
    assert(b2.at({0}) == 0.0f);
    assert(b2.at({1}) == 9.9f);

    std::cout << "  -> Save/Load Passed.\n";
    
    // Cleanup
    return 0;
}