#include "axon3/tensor.hpp"
#include "axon3/ops.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace axon;

void iota_fill(Tensor& t) {
    float* ptr = t.data_ptr();
    for (size_t i = 0; i < t.numel(); i++) {
        ptr[i] = static_cast<float>(i);
    }
}

void test_view() {
    std::cout << "[TEST] View (Reshape)...\n";

    // Create (2, 3) tensor
    Tensor t = Tensor::zeros({2, 3});
    iota_fill(t); 
    
    // View as (3, 2)
    Tensor v = axon::view(t, {3, 2});
    
    // Check Metadata
    assert(v.get_shape()[0] == 3 && v.get_shape()[1] == 2);
    assert(v.numel() == 6);

    // Check Data mapping
    assert(v.at({0, 0}) == 0.0f);
    assert(v.at({0, 1}) == 1.0f);
    assert(v.at({1, 0}) == 2.0f);
    assert(v.at({2, 1}) == 5.0f);

    std::cout << "  -> Forward Pass Passed.\n";

    // Autograd Check
    t.set_requires_grad(true);
    t.zero_grad();
    
    Tensor v2 = axon::view(t, {6}); // Flatten
    Tensor loss = axon::sum(v2);
    
    loss.backward();

    if (!t.get_grad()) {
        std::cerr << "Gradient not found!\n"; exit(1);
    }
    Tensor grad = *t.get_grad(); 
    
    assert(grad.get_shape()[0] == 2);
    assert(grad.get_shape()[1] == 3);
    
    for(size_t i=0; i<grad.numel(); i++) {
        if(grad.data_ptr()[i] != 1.0f) {
            std::cerr << "  -> Grad Error at " << i << ": " << grad.data_ptr()[i] << "\n";
            exit(1);
        }
    }

    std::cout << "  -> Backward Pass Passed.\n";
}

void test_permute() {
    std::cout << "[TEST] Permute (Transpose)...\n";

    // Create (2, 3)
    Tensor t = Tensor::zeros({2, 3});
    iota_fill(t);

    // Permute (1, 0) -> Shape (3, 2)
    Tensor p = axon::permute(t, {1, 0});

    assert(p.get_shape()[0] == 3);
    assert(p.get_shape()[1] == 2);

    // Check Data
    assert(p.at({0, 1}) == 3.0f); // Row 0, Col 1 in new tensor was Row 1, Col 0
    assert(p.at({1, 0}) == 1.0f);
    assert(p.at({2, 1}) == 5.0f);

    std::cout << "  -> Forward Pass Passed.\n";

    // Autograd Check
    t.set_requires_grad(true);
    t.zero_grad();

    Tensor p2 = axon::permute(t, {1, 0});
    Tensor loss = axon::sum(p2); 

    loss.backward();

    if (!t.get_grad()) {
        std::cerr << "Gradient not found!\n"; exit(1);
    }
    Tensor g = *t.get_grad();
    
    assert(g.get_shape()[0] == 2);
    assert(g.get_shape()[1] == 3);

    assert(g.at({0,0}) == 1.0f);

    std::cout << "  -> Backward Pass Passed.\n";
}

void test_transformer_mechanics() {
    std::cout << "[TEST] Transformer Head Split (View + Permute)...\n";

    int B=2, S=4, H=8;
    int nH=2, dH=4;

    Tensor x = Tensor::zeros({B, S, H});
    iota_fill(x);
    x.set_requires_grad(true);

    // View: Split hidden dim
    Tensor x_view = axon::view(x, {B, S, nH, dH});
    
    assert(x_view.get_shape().size() == 4);
    assert(x_view.get_shape()[2] == 2); // nH

    // Permute: Swap Seq and Heads (dim 1 and 2)
    // Target: (B, nH, S, dH)
    Tensor x_heads = axon::permute(x_view, {0, 2, 1, 3});

    assert(x_heads.get_shape()[0] == 2); // B
    assert(x_heads.get_shape()[1] == 2); // nH
    assert(x_heads.get_shape()[2] == 4); // S
    assert(x_heads.get_shape()[3] == 4); // dH

    float val = x_heads.at({0, 0, 1, 1});
    if (val != 9.0f) {
        std::cout << "Value Error! Expected 9.0, got " << val << "\n";
        exit(1);
    }
    std::cout << "  -> Forward Shapes & Strides Valid.\n";

    // Backward
    Tensor loss = axon::sum(x_heads);
    loss.backward();

    if (!x.get_grad()) {
        std::cerr << "Gradient not found!\n"; exit(1);
    }
    Tensor grad = *x.get_grad();
    
    bool ok = true;
    for(size_t i=0; i<grad.numel(); i++) {
        if(grad.data_ptr()[i] != 1.0f) ok = false;
    }

    if(ok) std::cout << "  -> Backward Pass (Complex) Passed.\n";
    else std::cout << "  -> Backward Pass (Complex) FAILED.\n";
}

int main() {
    test_view();
    test_permute();
    test_transformer_mechanics();
    std::cout << "---------------------------\n";
    std::cout << "ALL SHAPE TESTS PASSED.\n";
    return 0;
}