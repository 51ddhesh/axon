// src/math/unary_ops.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/TensorMath.hpp"
#include "../../include/private/OperationHelpers.hpp"

Tensor axon::math::sin(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::sin(val); });
}

Tensor axon::math::cos(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::cos(val); });
}

Tensor axon::math::tan(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::tan(val); });
}

Tensor axon::math::sqrt(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::sqrt(val); });
}

Tensor axon::math::exp(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::exp(val); });
}

Tensor axon::math::ln(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::log(val); });
}

Tensor axon::math::log10(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::log10(val); });
}

Tensor axon::math::log2(const Tensor& input) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(input, [](axon::dtype::f64 val) { return std::log2(val); });
}

Tensor axon::math::pow(const Tensor& base, axon::dtype::f64 exponent) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(base, [exponent](axon::dtype::f64 val) { return std::pow(val, exponent); });
}

Tensor axon::math::pow(axon::dtype::f64 base, const Tensor& exponent) {
    return axon::private_helpers::unary_op_helper::_apply_unary_op(exponent, [base](axon::dtype::f64 val) { return std::pow(base, val); });
}
