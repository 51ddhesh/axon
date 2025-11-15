// src/tensor_ops.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Tensor.hpp"

Tensor Tensor::operator+ (const Tensor& other_) const {
    // Check for compatibility
    bool rows_compatible = ((this -> rows() == other_.rows()) || (this -> rows() == 1) || (other_.rows() == 1));
    bool cols_compatible = ((this -> cols() == other_.cols()) || (this -> cols() == 1) || (other_.cols() == 1));
    
    if (!(rows_compatible && cols_compatible)) {
        throw std::invalid_argument("Tensors not compatible for addition");
    }

    // Form the resultant Tensor
    size_t result_rows = std::max(this -> rows(), other_.rows());
    size_t result_cols = std::max(this -> cols(), other_.cols());
    
    Tensor result(result_rows, result_cols);

    // Define the strides
    size_t this_row_stride = (this -> rows() == 1) ? 0 : 1;
    size_t this_col_stride = (this -> cols() == 1) ? 0 : 1;
    size_t other_row_stride = (other_.rows() == 1) ? 0 : 1;
    size_t other_col_stride = (other_.cols() == 1) ? 0 : 1;

    // Define the indices 
    size_t this_r = 0, this_c = 0;
    size_t other_r = 0, other_c = 0;

    for (size_t i = 0; i < result_rows; i++) {
        // Reset the column indices after the column loop starts
        this_c = 0;
        other_c = 0;

        for (size_t j = 0; j < result_cols; j++) {
            result(i, j) = (*this)(this_r, this_c) + other_(other_r, other_c);

            // Update the column indices after addition with the stride
            this_c += this_col_stride;
            other_c += other_col_stride;
        }

        // Update the row indices with the stride after the column loop completes
        this_r += this_row_stride;
        other_r += other_row_stride;
    }

    return result;
}

Tensor Tensor::operator+= (const Tensor& other_) {
    (*this) = (*this) + other_;
    return (*this);
}

Tensor Tensor::operator- (const Tensor& other_) const {
    bool rows_compatible = ((this -> rows() == other_.rows()) || (this -> rows() == 1) || (other_.rows() == 1));
    bool cols_compatible = ((this -> cols() == other_.cols()) || (this -> cols() == 1) || (other_.cols() == 1));

    if(!(rows_compatible && cols_compatible)) {
        throw std::invalid_argument("Tensors are not compatible for subtraction");
    }

    size_t result_rows = std::max(this -> rows(), other_.rows());
    size_t result_cols = std::max(this -> cols(), other_.cols());

    Tensor result(result_rows, result_cols);

    size_t this_row_stride = (this -> rows() == 1) ? 0 : 1;
    size_t this_col_stride = (this -> cols() == 1) ? 0 : 1;
    size_t other_row_stride = (other_.rows() == 1) ? 0 : 1;
    size_t other_col_stride = (other_.cols() == 1) ? 0 : 1;
    
    size_t this_r = 0, this_c = 0;
    size_t other_r = 0, other_c = 0;

    for (size_t i = 0; i < result_rows; i++) {
        this_c = 0;
        other_c = 0;

        for (size_t j = 0; j < result_cols; j++) {
            result(i, j) = (*this)(this_r, this_c) - other_(other_r, other_c);

            this_c += this_col_stride;
            other_c += other_col_stride;
        }

        this_r += this_row_stride;
        other_r += other_row_stride;
    }

    return result;
}

Tensor Tensor::operator-= (const Tensor& other_) {
    (*this) = (*this) - other_;
    return (*this);
}

Tensor Tensor::operator* (const Tensor& other_) const {
    bool rows_compatible = ((this -> rows() == other_.rows()) || (this -> rows() == 1) || (other_.rows() == 1));
    bool cols_compatible = ((this -> cols() == other_.cols()) || (this -> cols() == 1) || (other_.cols() == 1));
    
    if (!(rows_compatible && cols_compatible)) {
        throw std::invalid_argument("Tensors not compatible for addition");
    }

    size_t result_rows = std::max(this -> rows(), other_.rows());
    size_t result_cols = std::max(this -> cols(), other_.cols());
    
    Tensor result(result_rows, result_cols);

    size_t this_row_stride = (this -> rows() == 1) ? 0 : 1;
    size_t this_col_stride = (this -> cols() == 1) ? 0 : 1;
    size_t other_row_stride = (other_.rows() == 1) ? 0 : 1;
    size_t other_col_stride = (other_.cols() == 1) ? 0 : 1;

    size_t this_r = 0, this_c = 0;
    size_t other_r = 0, other_c = 0;

    for (size_t i = 0; i < result_rows; i++) {
        this_c = 0;
        other_c = 0;

        for (size_t j = 0; j < result_cols; j++) {
            result(i, j) = (*this)(this_r, this_c) * other_(other_r, other_c);

            this_c += this_col_stride;
            other_c += other_col_stride;
        }

        this_r += this_row_stride;
        other_r += other_row_stride;
    }

    return result;
}

Tensor Tensor::operator*= (const Tensor& other_) {
    (*this) = (*this) * other_;
    return (*this);
}

Tensor Tensor::operator/ (const Tensor& other_) const {
    bool rows_compatible = ((this -> rows() == other_.rows()) || (this -> rows() == 1) || (other_.rows() == 1));
    bool cols_compatible = ((this -> cols() == other_.cols()) || (this -> cols() == 1) || (other_.cols() == 1));
    
    if (!(rows_compatible && cols_compatible)) {
        throw std::invalid_argument("Tensors not compatible for addition");
    }

    size_t result_rows = std::max(this -> rows(), other_.rows());
    size_t result_cols = std::max(this -> cols(), other_.cols());
    
    Tensor result(result_rows, result_cols);

    size_t this_row_stride = (this -> rows() == 1) ? 0 : 1;
    size_t this_col_stride = (this -> cols() == 1) ? 0 : 1;
    size_t other_row_stride = (other_.rows() == 1) ? 0 : 1;
    size_t other_col_stride = (other_.cols() == 1) ? 0 : 1;

    size_t this_r = 0, this_c = 0;
    size_t other_r = 0, other_c = 0;

    for (size_t i = 0; i < result_rows; i++) {
        this_c = 0;
        other_c = 0;

        for (size_t j = 0; j < result_cols; j++) {
            assert(other_(other_r, other_c) != 0.0);
            result(i, j) = (*this)(this_r, this_c) / other_(other_r, other_c);

            this_c += this_col_stride;
            other_c += other_col_stride;
        }

        this_r += this_row_stride;
        other_r += other_row_stride;
    }

    return result;
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

