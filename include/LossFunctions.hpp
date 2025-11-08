// include/LossFunctions.hpp
// github.com/51ddhesh/axon
// MIT License

#ifndef AXON_LOSS_FUNCTIONS
#define AXON_LOSS_FUNCTIONS

#include "Tensor.hpp"

namespace axon_loss {
    // Loss for regression tasks
    // Mean Squared Error
    double mse(const Tensor& y_pred, const Tensor& y_true);

    // Loss for classification tasks
    // Categorical Cross-Entropy Loss
    double cce(const Tensor& y_pred, const Tensor& y_true);
} // namespace axon_loss

#endif // AXON_LOSS_FUNCTIONS
