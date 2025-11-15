// src/math/comparisons.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/TensorMath.hpp"
#include <functional>

// Anonymous namespacae for the helper functions
namespace {
    // Helper function for binary operations like a + b or a > b etc.
    Tensor _apply_op_bin(const Tensor& a, const Tensor& b, std::function<axon_dtype::f64(axon_dtype::f64, axon_dtype::f64)> op) {
        bool rows_compatible = ((a.rows() == b.rows()) || (a.rows() == 1) || (b.rows() == 1));
        bool cols_compatible = ((a.cols() == b.cols()) || (a.cols() == 1) || (b.cols() == 1));

        if (!(rows_compatible && cols_compatible)) {
            throw std::invalid_argument("The shape of the Tensors is not compatible");
        }

        size_t result_rows = std::max(a.rows(), b.rows());
        size_t result_cols = std::max(a.cols(), b.cols());
        Tensor result(result_rows, result_cols);

        size_t a_row_stride = (a.rows() == 1) ? 0 : 1;
        size_t a_col_stride = (a.cols() == 1) ? 0 : 1;
        size_t b_row_stride = (b.rows() == 1) ? 0 : 1;
        size_t b_col_stride = (b.cols() == 1) ? 0 : 1;

        size_t a_r = 0, a_c = 0;
        size_t b_r = 0, b_c = 0;

        for (size_t i = 0; i < result_rows; i++) {
            a_c = 0;
            b_c = 0;
            for (size_t j = 0; j < result_cols; j++) {
                result(i, j) = op(a(a_r, a_c), b(b_r, b_c));
                a_c += a_col_stride;
                b_c += b_col_stride;
            }
            a_r += a_row_stride;
            b_r += b_row_stride;
        }
        return result;
    }
}


Tensor axon::math::gt(const Tensor& a, const Tensor& b) {
    return _apply_op_bin(a, b, [](axon_dtype::f64 val_a, axon_dtype::f64 val_b) { return val_a > val_b ? 1.0 : 0.0; });
}

Tensor axon::math::lt(const Tensor& a, const Tensor& b) {
    return _apply_op_bin(a, b, [](axon_dtype::f64 val_a, axon_dtype::f64 val_b) { return val_a < val_b ? 1.0 : 0.0; });
}

Tensor axon::math::eq(const Tensor& a, const Tensor& b) {
    const axon_dtype::f64 eps = 1e-9;
    return _apply_op_bin(a, b, [eps](axon_dtype::f64 val_a, axon_dtype::f64 val_b) { return std::abs(val_a - val_b) < eps ? 1.0 : 0.0; });
}
