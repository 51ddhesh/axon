// src/nn/activations.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Activations.hpp"

Tensor axon::activation::relu(const Tensor& input) {
    Tensor result(input.rows(), input.cols());
    for (size_t i = 0; i < input.get_size(); ++i) {
        result(i) = std::max(0.0, input(i));
    }
    return result;
}

Tensor axon::activation::sigmoid(const Tensor& input) {
    Tensor result(input.rows(), input.cols());
    for (size_t i = 0; i < input.get_size(); i++) {
        result(i) = 1 / (1 + std::exp(-input(i)));
    }
    return result;
}

Tensor axon::activation::tanh(const Tensor& input) {
    Tensor result(input.rows(), input.cols());
    for (size_t i = 0; i < input.get_size(); i++) {
        result(i) = std::tanh(input(i));
    }
    return result;
}

Tensor axon::activation::softmax(const Tensor& input) {
    Tensor result = input;
    for (size_t i = 0; i < result.rows(); i++) {
        double max_from_row = -INFINITY;
        // Find the max from each row to subtract from each element of the row
        for (size_t j = 0; j < result.cols(); j++) {
            max_from_row = std::max(max_from_row, result(i, j));
        }
        // exponentiate shifted values and find sum for the row
        double sum = 0.0;
        for (size_t j = 0; j < result.cols(); j++) {
            result(i, j) = std::exp(result(i, j) - max_from_row);
            sum += result(i, j);
        }
        // normalize to get probabilities
        if (sum != 0.0) {
            for (size_t j = 0; j < result.cols(); j++) {
                result(i, j) /= sum;
            }
        }
    }
    return result;
}
