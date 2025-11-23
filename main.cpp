#include "include/Tensor.hpp"
#include <iostream>

using namespace axon;

int main() {

    Tensor X = Tensor::from_2d({
        {0.0, 0.0}, 
        {0.0, 1.0}, 
        {1.0, 0.0}, 
        {1.0, 1.0}
    });

    Tensor y = Tensor::from_2d({
        {0.0}, 
        {1.0}, 
        {1.0}, 
        {0.0}
    });

    // 2 Inputs -> 4 Hidden -> 1 Output
    Tensor w1({2, 4}, true);
    Tensor b1({1, 4}, true);
    Tensor w2({4, 1}, true);
    Tensor b2({1, 1}, true);

    double lr = 0.05;

    std::cout << "Start Training XOR...\n";

    for (int epoch = 0; epoch < 5000; ++epoch) {
        
        // 1. Forward
        Tensor h1_dense = matmul(X, w1); 
        Tensor h1_bias = h1_dense + b1; // Uses broadcasting operator+
        Tensor h1_relu = Tensor::relu(h1_bias);
        Tensor y_pred = matmul(h1_relu, w2) + b2; // Uses broadcasting operator+

        // 2. Loss (MSE)
        Tensor diff = y_pred - y;
        Tensor sq = Tensor::pow(diff, 2.0);
        Tensor loss = sq.sum(); 

        // 3. Zero Grad
        w1.zero_grad(); b1.zero_grad();
        w2.zero_grad(); b2.zero_grad();

        // 4. Backward
        loss.backward();

        // 5. Step (Update weights manually)
        for(size_t i=0; i<w1.size(); ++i) w1.data_ptr()[i] -= lr * w1.grad_ptr()[i];
        for(size_t i=0; i<b1.size(); ++i) b1.data_ptr()[i] -= lr * b1.grad_ptr()[i];
        for(size_t i=0; i<w2.size(); ++i) w2.data_ptr()[i] -= lr * w2.grad_ptr()[i];
        for(size_t i=0; i<b2.size(); ++i) b2.data_ptr()[i] -= lr * b2.grad_ptr()[i];

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss.item(0) << std::endl;
        }
    }
    
    std::cout << "\nPredictions:\n";
    Tensor pred = matmul(Tensor::relu(matmul(X, w1) + b1), w2) + b2;
    for(size_t i=0; i<4; ++i) {
        std::cout << "Input: (" << X.item(i*2) << "," << X.item(i*2+1) << ") Pred: " << pred.item(i) << "\n";
    }

    return 0;
}