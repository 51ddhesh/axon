// src/nn/linear.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/Linear.hpp"

Linear::Linear(size_t input_size, size_t output_size, std::function<Tensor(const Tensor&)> activation) {
    this -> _activation = activation;
    // Init a matrix of size `input_size * output_size` to maintain the correctness of matmul
    // This is done considering the input has dims: `batch_size * input_size (the number of features)`
    // Completing the forward pass would output a matrix of `batch_size * output_size`
    _weights = Tensor::randn(input_size, output_size, {-0.1, 0.1}); 
    _bias = Tensor::zeros(1, output_size);
}


// `torch` style forward pass 
// Main implementation of the forward pass
Tensor Linear::linear(const Tensor& input) {
    // Perform the matmul input @ weights
    Tensor result = matmul(input, _weights);
    result += _bias;
    return _activation(result);
}

// `tf` style forward pass
Tensor Linear::forward(const Tensor& input) {
    return this -> linear(input);
}

std::vector<Tensor*> Linear::parameters() {
    return { &_weights, &_bias };
}
