// include/Linear.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_LINEAR_HPP
#define AXON_LINEAR_HPP

#include "Tensor.hpp"   
#include "Activations.hpp"
#include <functional>

class Linear {
private:
    Tensor _weights;
    Tensor _bias;
    std::function<Tensor(const Tensor&)> _activation;

public:
    // Constructor
    // Create a linear layer with input_size=features, output_size=number of neurons and an activation function
    Linear(size_t input_size, size_t output_size, std::function<Tensor(const Tensor&)> activation);

    // torch-styled forward pass
    Tensor linear(const Tensor& input);

    // Tensorflow styled forward pass
    Tensor forward(const Tensor& input);

    // Method to get the pointers to weights and bias
    std::vector<Tensor*> parameters();
};

#endif // AXON_LINEAR_HPP
