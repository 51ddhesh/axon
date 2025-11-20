// include/private/OperationHelpers.hpp
// github.com/51ddhesh/axon
// MIT License

// * NOTE: This file holds private helper functions and is not part of the public API 

#ifndef AXON_PRIVATE_OPERATION_HELPERS_HPP
#define AXON_PRIVATE_OPERATION_HELPERS_HPP

#include "../Tensor.hpp"
#include <algorithm>
#include <stdexcept>

namespace axon {
    namespace private_helpers {
        namespace unary_op_helper {
            template <typename UnaryOp>
            Tensor _apply_unary_op(const Tensor& a, UnaryOp op) {
                Tensor result(a.rows(), a.cols());
                for (size_t i = 0; i < a.get_size(); i++) {
                    result(i) = op(a(i));
                }
                return result;
            }
        } // namespace unary_op_helper

        namespace binary_op_helper {
            template <typename BinaryOp>
            Tensor _apply_binary_op(const Tensor& a, const Tensor& b, BinaryOp op) {
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
        } // namespace binary_op_helper

        namespace other {
            inline Tensor sum_to_shape(const Tensor& input, const std::vector<size_t>& shape) {
                if (input.getShape() == shape) return input;
                Tensor result = input;
                if (result.rows() != shape[0]) {
                    result = result.sum(0);
                }
                if (result.cols() != shape[1]) {
                    result = result.sum(1);
                }
                return result;
            }
        } // namespace other

    } // namespace private
} // namespace axon

#endif // AXON_PRIVATE_OPERATION_HELPERS_HPP