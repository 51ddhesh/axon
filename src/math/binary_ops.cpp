// src/math/binary_ops.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/TensorMath.hpp"
#include "../../include/private/OperationHelpers.hpp"

Tensor axon::math::pow(const Tensor& base, const Tensor& power) {
    return axon::private_helpers::binary_op_helper::_apply_binary_op(base, power, [](axon::dtype::f64 b, axon::dtype::f64 p) { return std::pow(b, p); });    
}

Tensor axon::math::maximum(const Tensor& a, const Tensor& b) {
    return axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, [](axon::dtype::f64 x, axon::dtype::f64 y) { return (x > y) ? x : y; });
}

Tensor axon::math::minimum(const Tensor& a, const Tensor& b) {
    return axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, [](axon::dtype::f64 x, axon::dtype::f64 y) { return (x > y) ? y : x; });
}
