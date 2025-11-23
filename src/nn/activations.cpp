// src/nn/activations.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Activations.hpp"
#include "../../include/TensorMath.hpp"

Tensor axon::activation::relu(const Tensor& input) {
    // TODO: Make the axon::math::maximum differentiable
    size_t tensor_size = input.get_size();
    Tensor result(input.rows(), input.cols());
    for (size_t i = 0; i < tensor_size; ++i) {
        result(i) = std::max(0.0, input(i));
    }

    result._prev = { const_cast<Tensor*>(&input) };
    
    result._backward_fn = [p_input = &input] (Tensor* self) {
        /*
            f(x) = ReLU(x) = (x > 0) ? x : 0;
            f'(x) = (x > 0) ? 1 : 0

            For the input:
            [[1.5, -2.0],  
             [-5.2, 7.0]]  

            The ReLU is:
            [[1.5, 0],
             [0, 7.0]]

            The partial derivative is:
            [[1.0, 0],
             [0, 1.0]]

        */

        // This creates a Tensor where the element is 1.0 if a > b else, 0.0 directly
        auto mask = axon::math::gt(*p_input, Tensor::zeros(p_input -> rows(), p_input -> cols()));
        *(p_input -> _grad) += *(self -> _grad) * mask;
        
    };

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
