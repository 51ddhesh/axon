#include "axon3/tensor.hpp"
#include "axon3/nn.hpp"
#include "axon3/ops.hpp"
#include <iostream>
#include <cassert>

using namespace axon;

void test_embedding() {
    std::cout << "[TEST] Embedding...\n";
    // Vocab=10, Dim=4
    nn::Embedding emb(10, 4);
    
    // Indices: [1, 2]
    Tensor idx = Tensor::zeros({2});
    idx.data_ptr()[0] = 1.0f;
    idx.data_ptr()[1] = 2.0f;

    Tensor out = emb.forward(idx);

    // Check Shape: (2, 4)
    assert(out.get_shape()[0] == 2);
    assert(out.get_shape()[1] == 4);

    // Backward
    Tensor loss = sum(out);
    loss.backward();

    // Check Grads
    // We summed 'out'. So grad_out is all 1s.
    // The embedding layer should scatter these 1s into grad_weight.
    // row 1 of weight should have grad [1, 1, 1, 1]
    // row 2 of weight should have grad [1, 1, 1, 1]
    // row 0 should be 0.
    
    Tensor gw = *emb.weight.get_grad();
    assert(gw.at({0,0}) == 0.0f);
    assert(gw.at({1,0}) == 1.0f);
    assert(gw.at({2,0}) == 1.0f);
    
    std::cout << "  -> Embedding Passed.\n";
}

void test_layernorm() {
    std::cout << "[TEST] LayerNorm...\n";
    // Batch=2, Dim=3
    Tensor x = Tensor::zeros({2, 3});
    // Row 0: [0, 10, 20] -> Mean=10, Var ~66
    // Row 1: [3, 3, 3]   -> Mean=3, Var=0
    x.at({0,0}) = 0; x.at({0,1}) = 10; x.at({0,2}) = 20;
    x.at({1,0}) = 3; x.at({1,1}) = 3;  x.at({1,2}) = 3;
    x.set_requires_grad(true);

    nn::LayerNorm ln(3);
    Tensor out = ln.forward(x);

    // Check Logic
    // Row 1 should be [0, 0, 0] because Var=0
    // (technically approx 0 due to eps)
    assert(std::abs(out.at({1,0})) < 0.01f);
    
    // Backward
    Tensor loss = sum(out);
    loss.backward();
    
    // Just ensure it runs and produces grads
    assert(x.get_grad() != nullptr);
    assert(ln.gamma.get_grad() != nullptr);

    std::cout << "  -> LayerNorm Passed.\n";
}

int main() {
    test_embedding();
    test_layernorm();
    return 0;
}