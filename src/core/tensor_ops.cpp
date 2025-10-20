// src/tensor_ops.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Tensor.hpp"

// Element wise Tensor Ops

// Element-wise Tensor addition
Tensor Tensor::operator+ (const Tensor& other_) const {
    if (getShape() != other_.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }
    Tensor result(rows(), cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] + other_._data[i];
    }
    return result;
}

// Element-wise Tensor subtraction
Tensor Tensor::operator- (const Tensor& other_) const {
    if (getShape() != other_.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }
    Tensor result(rows(), cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] - other_._data[i];
    }
    return result;
}

// Element-wise Tensor Multiplication 
// ! THIS IS NOT `MATMUL`
Tensor Tensor::operator* (const Tensor& other_) const {
    if (getShape() != other_.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }
    Tensor result(rows(), cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] * other_._data[i];
    }
    return result;
}

// Element-wise Tensor division
Tensor Tensor::operator/ (const Tensor& other_) const {
    if (getShape() != other_.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }
    Tensor result(rows(), cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] / other_._data[i];
    }
    return result;
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

// * Compound Operations

// Element Wise Compound Addition
Tensor Tensor::operator+=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] += other_._data[i];
    }
    return *this;
}

Tensor Tensor::operator-=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] -= other_._data[i];
    }
    return *this;
}

Tensor Tensor::operator*=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] *= other_._data[i];
    }
    return *this;
}

Tensor Tensor::operator/=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> _data.size(); i++) {
        assert(other_._data[i] != 0.0);
        this -> _data[i] /= other_._data[i];
    }
    return *this;
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
        result += a.getData()[i] * b.getData()[i];
    }

    return result;
}


axon_dtype::f64 dot(const Tensor& a, const Tensor& b) {
    if (a.get_size() != b.get_size()) {
        throw std::invalid_argument("The number of elements must be same for both Tensors to perform a dot product");
    }
    axon_dtype::f64 result = 0.0;

    for (size_t i = 0; i < a.get_size(); i++) {
        result += a.getData()[i] * b.getData()[i];
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
// TODO:
Tensor Tensor::T() {
    
}

