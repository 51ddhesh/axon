// src/nn/loss.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/LossFunctions.hpp"
#include <limits>
#include <cmath>

Tensor axon::loss::mse(const Tensor& y_pred, const Tensor& y_true) {
    if (y_pred.getShape() != y_true.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }

    Tensor diff = y_pred - y_true;

    Tensor squared_diff = axon::math::pow(diff, 2.0);
    Tensor sum_sq = squared_diff.sum();

    return sum_sq * (1.0 / y_true.get_size());
}

// Assumes that y_true is one-hot encoded 
// for a single sample, L = -sum (true_label * ln(predicted))
// In axon::loss::cce, we return the average loss per sample
Tensor axon::loss::cce(const Tensor& y_pred, const Tensor& y_true) {
    if (y_pred.getShape() != y_true.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }

    Tensor y_pred_safe = y_pred + axon::constants::eps;
    Tensor log_preds = axon::math::ln(y_pred_safe);

    Tensor terms = y_true * log_preds;
    Tensor sum = terms.sum();

    return (sum / static_cast<axon::dtype::f64>(y_true.rows()));
}
