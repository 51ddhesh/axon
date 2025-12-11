#include "axon/tensor.hpp"
#include "axon/ops.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Axon Training Test\n";
    std::cout << "Goal: Learn that 2 * 3 = 6\n";

    // Data
    auto x = axon::Tensor::ones({1, 1}); 
    x.at({0,0}) = 2.0f;
    
    auto y_target = axon::Tensor::ones({1, 1});
    y_target.at({0,0}) = 6.0f;

    // Weights (Start at random or zero)
    // We start at 0.0. The model predicts 2 * 0 = 0. Error is High.
    auto w = axon::Tensor::zeros({1, 1});
    w.set_requires_grad(true);

    // Training Loop
    float learning_rate = 0.05f;

    std::cout << std::fixed << std::setprecision(4);

    for (int epoch = 0; epoch < 20; ++epoch) {
        
        // forward pass
        // pred = x * w
        auto pred = axon::matmul(x, w);
        
        // Loss = (pred - target)^2
        // We implement MSE manually using basic Ops:
        auto diff = axon::sub(pred, y_target); // Error
        auto sq_diff = axon::mul(diff, diff); // Square Error
        auto loss = axon::sum(sq_diff); // Mean/Sum (scalar)

        // backward
        // Clear previous gradients (Gradient Descent Step)
        w.zero_grad();

        // Calculate new gradients
        loss.backward();

        // optim (sgd)
        // Update weights: w = w - lr * grad
        // We access data directly (.at) to avoid tracking this update in the graph
        float grad_val = w.get_grad() -> at({0,0});
        w.at({0,0}) -= learning_rate * grad_val;

        // Log every few epochs
        if (epoch % 2 == 0) {
            std::cout << "Epoch " << epoch 
                      << " | Loss: " << loss.at({0}) 
                      << " | Grad: " << grad_val 
                      << " | W: " << w.at({0,0}) << "\n";
        }
    }

    std::cout << "--------------------------------\n";
    std::cout << "Final Prediction: 2.0 * " << w.at({0,0}) << " = " << (2.0f * w.at({0,0})) << "\n";
    std::cout << "Target was: 6.0\n";
    
    return 0;
}