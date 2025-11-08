// src/nn/loss.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/LossFunctions.hpp"
#include <limits>
#include <cmath>

double axon_loss::mse(const Tensor& y_pred, const Tensor& y_true) {
    if (y_pred.getShape() != y_true.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }

    double total_err = 0.0;
    double diff = 0.0;
    for (size_t i = 0; i < y_true.get_size(); i++) {
        diff = y_pred(i) - y_true(i);
        total_err += diff * diff;
    }

    return total_err / y_true.get_size();
}

// Assumes that y_true is one-hot encoded 
// for a single sample, L = -sum (true_label * ln(predicted))
// In axon_loss::cce, we return the average loss per sample
double axon_loss::cce(const Tensor& y_pred, const Tensor& y_true) {
    if (y_pred.getShape() != y_true.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }

    double total_err = 0.0;
    // Small value to avoid ln(0) -> -inf
    const double eps = 1e-9;

    for (size_t i = 0; i < y_true.get_size(); i++) {
        if (y_true(i) == 1.0) {
            total_err += std::log(y_pred(i) + eps);
        }
    }

    return -total_err / y_true.rows();
}
