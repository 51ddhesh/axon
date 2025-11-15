#include "../../include/TensorMath.hpp"

Tensor axon::math::gt(const Tensor& a, const Tensor& b) {
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
            result(i, j) = (a(a_r, a_c) > b(b_r, b_c)) ? 1.0 : 0.0;
            a_c += a_col_stride;
            b_c += b_col_stride;
        }
        a_r += a_row_stride;
        b_r += b_row_stride;
    }

    return result;
}

Tensor axon::math::lt(const Tensor& a, const Tensor& b) {
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
            result(i, j) = (a(a_r, a_c) < b(b_r, b_c)) ? 1.0 : 0.0;
            a_c += a_col_stride;
            b_c += b_col_stride;
        }
        a_r += a_row_stride;
        b_r += b_row_stride;
    }

    return result;
}

Tensor axon::math::eq(const Tensor& a, const Tensor& b) {
    bool rows_compatible = ((a.rows() == b.rows()) || (a.rows() == 1) || (b.rows() == 1));
    bool cols_compatible = ((a.cols() == b.cols()) || (a.cols() == 1) || (b.cols() == 1));

    if (!(rows_compatible && cols_compatible)) {
        throw std::invalid_argument("The shape of the Tensors is not compatible");
    }

    const double eps = 1e-9;

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
            double res = std::abs(a(a_r, a_c) - b(b_r, b_c));
            result(i, j) = (res < eps) ? 1.0 : 0.0;
            a_c += a_col_stride;
            b_c += b_col_stride;
        }
        a_r += a_row_stride;
        b_r += b_row_stride;
    }

    return result;
}
