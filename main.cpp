#include "axon/Tensor.hpp"  
#include "axon/NN.hpp"
#include "axon/Optimizer.hpp"
#include <iostream>

using namespace axon;

int main() {
    std::cout << "=== Axon v2.1 XOR Test (Sigmoid) ===" << std::endl;
    
    // Data
    Tensor X = {0,0, 0,1, 1,0, 1,1};
    X = X.reshape({4, 2});
    
    // Target
    Tensor Y = {0, 1, 1, 0};
    Y = Y.reshape({4, 1});

    // Model (2 -> 8 -> 1)
    nn::Linear fc1(2, 8);
    nn::Linear fc2(8, 1);

    // Optim
    std::vector<Tensor> params; 
    auto p1 = fc1.parameters();
    auto p2 = fc2.parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());

    // Learning Rate 0.5 works well for Sigmoid + MSE on XOR
    optim::SGD optimizer(params, 0.5);

    std::cout << "Training..." << std::endl;

    for (int epoch = 0; epoch < 5000; ++epoch) {
        // Forward
        Tensor h1 = fc1(X).sigmoid(); // Use Sigmoid
        Tensor out = fc2(h1).sigmoid(); // Sigmoid on output too (0-1 range)
        
        // Loss (MSE)
        Tensor diff = out - Y;
        Tensor loss = (diff * diff).sum();
        
        // Backward
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss.data_ptr()[0] << std::endl;
        }
    }

    // Predictions
    std::cout << "\nFinal Predictions (Should be 0, 1, 1, 0):" << std::endl;
    Tensor h1 = fc1(X).sigmoid();
    Tensor pred = fc2(h1).sigmoid();
    pred.print();

    return 0;
}