#include "axon/tensor.hpp"
#include "axon/ops.hpp"
#include "axon/optimizer.hpp"
#include <iostream>
#include <iomanip>

using namespace axon;

int main() {
    std::cout << "Axon: Solving XOR\n";

    // Data
    auto x = Tensor::zeros({4, 2});
    x.at({0,0}) = 0; x.at({0,1}) = 0;
    x.at({1,0}) = 0; x.at({1,1}) = 1;
    x.at({2,0}) = 1; x.at({2,1}) = 0;
    x.at({3,0}) = 1; x.at({3,1}) = 1;

    auto y = Tensor::zeros({4, 1});
    y.at({0,0}) = 0;
    y.at({1,0}) = 1;
    y.at({2,0}) = 1;
    y.at({3,0}) = 0;

    // Model: 2 -> 8 -> 1
    // use He Initialization manually here
    auto w1 = Tensor::zeros({2, 8}); 
    auto b1 = Tensor::zeros({1, 8});
    auto w2 = Tensor::zeros({8, 1});
    auto b2 = Tensor::zeros({1, 1});

    // Init weights
    for(size_t i=0; i<w1.numel(); i++) w1.data_ptr()[i] = (((float)rand()/RAND_MAX) - 0.5f) * 2.0f; // Range -1 to 1
    for(size_t i=0; i<w2.numel(); i++) w2.data_ptr()[i] = (((float)rand()/RAND_MAX) - 0.5f) * 2.0f;

    w1.set_requires_grad(true); b1.set_requires_grad(true);
    w2.set_requires_grad(true); b2.set_requires_grad(true);

    SGD optimizer({w1, b1, w2, b2}, 0.05f); 

    std::cout << std::fixed << std::setprecision(4);

    for (int epoch = 0; epoch < 2000; ++epoch) {
        
        // Forward
        auto h1 = relu(add(matmul(x, w1), b1));
        auto out = add(matmul(h1, w2), b2);
        
        // Loss: MSE
        // diff = out - y
        auto diff = sub(out, y);
        // sq = diff * diff
        auto sq = mul(diff, diff);
        auto loss = sum(sq);

        optimizer.zero_grad();
        loss.backward(); 
        optimizer.step();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss.at({0}) << "\n";
        }
    }

    // Inference
    auto h1 = relu(add(matmul(x, w1), b1));
    auto pred = add(matmul(h1, w2), b2);
    
    std::cout << "\nFinal (Target: 0, 1, 1, 0):\n";
    pred.print();

    return 0;
}