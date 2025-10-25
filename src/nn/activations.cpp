#include "../../include/Activations.hpp"

Tensor axon_activation::relu(const Tensor& input) {
    Tensor result(input.rows(), input.cols());
    for (size_t i = 0; i < input.get_size(); ++i) {
        result(i) = std::max(0.0, input(i));
    }
    return result;
}

Tensor axon_activation::sigmoid(const Tensor& input) {
    Tensor result(input.rows(), input.cols());
    for (size_t i = 0; i < input.get_size(); i++) {
        result(i) = 1 / (1 + std::exp(-input(i)));
    }
    return result;
}

Tensor axon_activation::tanh(const Tensor& input) {
    Tensor result(input.rows(), input.cols());
    for (size_t i = 0; i < input.get_size(); i++) {
        result(i) = std::tanh(input(i));
    }
    return result;
}

// TODO:
// Tensor axon_activation::softmax(const Tensor& input, size_t dim) {
// 
// }
