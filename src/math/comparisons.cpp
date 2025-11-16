// src/math/comparisons.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/TensorMath.hpp"
#include "../../include/private/OperationHelpers.hpp"

Tensor axon::math::gt(const Tensor& a, const Tensor& b) {
    return axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, [](axon_dtype::f64 val_a, axon_dtype::f64 val_b) { return val_a > val_b ? 1.0 : 0.0; });
}

Tensor axon::math::lt(const Tensor& a, const Tensor& b) {
    return axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, [](axon_dtype::f64 val_a, axon_dtype::f64 val_b) { return val_a < val_b ? 1.0 : 0.0; });
}

Tensor axon::math::eq(const Tensor& a, const Tensor& b) {
    const axon_dtype::f64 eps = 1e-9;
    return axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, [eps](axon_dtype::f64 val_a, axon_dtype::f64 val_b) { return std::abs(val_a - val_b) < eps ? 1.0 : 0.0; });
}
