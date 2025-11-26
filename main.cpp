#include "axon/Tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace axon;

int main() {
    std::cout << "Axon XOR Training Test" << std::endl;

    // 1. Data (XOR Problem)
    // Inputs: (4, 2)
    Tensor X = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };
    X = X.reshape({4, 2});

    // Targets: (4, 1)
    // 0^0=0, 0^1=1, 1^0=1, 1^1=0
    Tensor Y = {0, 1, 1, 0};
    Y = Y.reshape({4, 1});

    // 2. Weights & Biases
    // Hidden Layer (2 -> 8)
    // W1: (2, 8)
    Tensor W1 = Tensor::ones({2, 8}); 
    // Random initialization is needed! But we don't have rand yet.
    // Let's manually set some different values to break symmetry
    for(size_t i=0; i<W1.size(); ++i) W1.data_ptr()[i] = ((double)i / 10.0) * (i%2==0?1:-1); 
    
    Tensor b1 = Tensor::zeros({1, 8});

    // Output Layer (8 -> 1)
    Tensor W2 = Tensor::ones({8, 1});
    for(size_t i=0; i<W2.size(); ++i) W2.data_ptr()[i] = ((double)i / 5.0) * (i%2!=0?1:-1);

    Tensor b2 = Tensor::zeros({1, 1});

    // Hyperparameters
    double lr = 0.01;
    int epochs = 100;

    std::cout << "Training..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward Pass 
        
        // Layer 1
        // (4,2) @ (2,8) + (1,8) -> (4,8)
        Tensor h1 = X.matmul(W1) + b1; 
        Tensor a1 = h1.relu();
        
        // Layer 2
        // (4,8) @ (8,1) + (1,1) -> (4,1)
        Tensor h2 = a1.matmul(W2) + b2;
        
        // MSE Loss
        // Diff = (h2 - Y)
        Tensor diff = h2 - Y;
        // Sq = diff * diff
        Tensor sq = diff * diff;
        // Loss = Sum(Sq)
        Tensor loss = sq.sum();
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss.data_ptr()[0] << std::endl;
        }

        // Backward Pass
        
        // 1. Zero Gradients
        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        
        // 2. Backprop
        loss.backward();

        // Optimizer Step (SGD) 
        // W -= lr * grad
        // We don't have operator-= yet, and we need to be careful not to build the graph here!
        // We modify data directly.
        
        auto step = [&](Tensor& w) {
            double* d = w.data_ptr();
            double* g = w.grad_ptr();
            for(size_t i=0; i<w.size(); ++i) {
                d[i] -= lr * g[i];
            }
        };

        step(W1); step(b1);
        step(W2); step(b2);
    }
    
    // Final Prediction
    std::cout << "\nFinal Predictions:" << std::endl;
    Tensor h1 = X.matmul(W1) + b1;
    Tensor a1 = h1.relu();
    Tensor pred = a1.matmul(W2) + b2;
    pred.print(); // Should be close to 0, 1, 1, 0

    return 0;
}