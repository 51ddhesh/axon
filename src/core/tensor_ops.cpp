// src/tensor_ops.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Tensor.hpp"

// Anonymous namespace for private helper
namespace {
    template <typename Functor>
    Tensor _apply_op_bin(const Tensor& a, const Tensor& b, Functor op) {
        // Check for compatibility
        bool rows_compatible = ((a.rows() == b.rows()) || (a.rows() == 1) || (b.rows() == 1));
        bool cols_compatible = ((a.cols() == b.cols()) || (a.cols() == 1) || (b.cols() == 1));

        if (!(rows_compatible && cols_compatible)) {
            throw std::invalid_argument("The shape of the Tensors is not compatible");
        }

        // Form the resultant tensor
        size_t result_rows = std::max(a.rows(), b.rows());
        size_t result_cols = std::max(a.cols(), b.cols());
        Tensor result(result_rows, result_cols);

        // Find the strides 
        size_t a_row_stride = (a.rows() == 1) ? 0 : 1;
        size_t a_col_stride = (a.cols() == 1) ? 0 : 1;
        size_t b_row_stride = (b.rows() == 1) ? 0 : 1;
        size_t b_col_stride = (b.cols() == 1) ? 0 : 1;

        // Initialize the indices
        size_t a_r = 0, a_c = 0;
        size_t b_r = 0, b_c = 0;

        // Actual operation
        for (size_t i = 0; i < result_rows; i++) {
            // Reset the column indices before entering the column loop
            a_c = 0;
            b_c = 0;
            for (size_t j = 0; j < result_cols; j++) {
                // Perform the op
                result(i, j) = op(a(a_r, a_c), b(b_r, b_c));
                // advance the column indices
                a_c += a_col_stride;
                b_c += b_col_stride;
            }
            // Advance the row indices
            a_r += a_row_stride;
            b_r += b_row_stride;
        }
        return result;
    }
} // namespace 


Tensor Tensor::operator+ (const Tensor& other_) const {
    return _apply_op_bin((*this), other_, std::plus<axon_dtype::f64>());
}

Tensor Tensor::operator+= (const Tensor& other_) {
    (*this) = (*this) + other_;
    return (*this);
}

Tensor Tensor::operator- (const Tensor& other_) const {
    return _apply_op_bin((*this), other_, std::minus<axon_dtype::f64>());
}

Tensor Tensor::operator-= (const Tensor& other_) {
    (*this) = (*this) - other_;
    return (*this);
}

Tensor Tensor::operator* (const Tensor& other_) const {
    return _apply_op_bin((*this), other_, std::multiplies<axon_dtype::f64>());
}

Tensor Tensor::operator*= (const Tensor& other_) {
    (*this) = (*this) * other_;
    return (*this);
}

// ! NOTE: `std::divides` does not have an in-built assert to check for the divisor being zero
// * Dividing by zero will result in `inf`
Tensor Tensor::operator/ (const Tensor& other_) const {
    return _apply_op_bin((*this), other_, std::divides<axon_dtype::f64>());
}

Tensor Tensor::operator/= (const Tensor& other_) {
    (*this) = (*this) / other_;
    return (*this);
}

// Tensor Ops with Scalars

// Scalar Addition
Tensor Tensor::operator+ (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] + val_;
    }
    return result;
}

// Scalar Subtraction
Tensor Tensor::operator- (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] - val_;
    }
    return result;
}

// Scalar Multiplication
Tensor Tensor::operator* (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] * val_;
    }
    return result;
}

// Scalar Division
Tensor Tensor::operator/ (const double val_) const {
    assert(val_ != 0.0);
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] / val_;
    }
    return result;
}

Tensor Tensor::operator+=(const double val_) {
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] += val_;
    }
    return *this;
}

Tensor Tensor::operator-=(const double val_) {
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] -= val_;
    }
    return *this;
}

Tensor Tensor::operator*=(const double val_) {
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] *= val_;
    }
    return *this;
}

Tensor Tensor::operator/=(const double val_) {
    assert(val_ != 0.0);
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] /= val_;
    }
    return *this;
}

axon_dtype::f64 frobenius_inner_product(const Tensor& a, const Tensor& b) {
    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("The shape must match for Frobenius Inner Product");
    }

    axon_dtype::f64 result = 0.0;
    for (size_t i = 0; i < a.get_size(); i++) {
        result += a(i) * b(i);
    }

    return result;
}


axon_dtype::f64 dot(const Tensor& a, const Tensor& b) {
    if (a.get_size() != b.get_size()) {
        throw std::invalid_argument("The number of elements must be same for both Tensors to perform a dot product");
    }
    axon_dtype::f64 result = 0.0;

    for (size_t i = 0; i < a.get_size(); i++) {
        result += a(i) * b(i);
    }

    return result;
}

// cache-friendly matmul
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t a_rows = a.rows();
    size_t a_cols = a.cols(); // shared dim
    size_t b_cols = b.cols();

    Tensor result = Tensor::zeros(a_rows, b_cols);

    // cache friendly loop
    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t k = 0; k < a_cols; ++k) {
            // Fetch a(i, k) once and reuse it across the inner loop
            double a_val = a(i, k); 
            for (size_t j = 0; j < b_cols; ++j) {
                result(i, j) += a_val * b(k, j);
            }
        }
    }

    return result;
}

// Transpose
Tensor Tensor::T() const {
    Tensor result(cols(), rows());
    for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < cols(); j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

