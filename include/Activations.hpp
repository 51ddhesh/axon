// include/Activation.hpp
// github.com/51ddhesh
// MIT License


#ifndef AXON_ACTIVATION_HPP
#define AXON_ACTIVATION_HPP

#include "Tensor.hpp"
#include <algorithm>
#include <cmath>

namespace axon_activation {
    Tensor relu(const Tensor& input);
    
    Tensor sigmoid(const Tensor& input);

    Tensor tanh(const Tensor& input);

    Tensor softmax(const Tensor& input, size_t dim);
}

#endif // AXON_ACTIVATION_HPP
