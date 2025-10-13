// src/tensor_ops.cpp
// github.com/51ddhesh
// MIT License

#include "../include/Tensor.hpp"

// Element wise Tensor Ops

// Element-wise Tensor addition
Tensor Tensor::operator+ (const Tensor& other_) const {
    if (getShape() != other_.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }
    Tensor result(rows(), cols());
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = this -> data[i] + other_.data[i];
    }
    return result;
}

// Element-wise Tensor subtraction
Tensor Tensor::operator- (const Tensor& other_) const {
    if (getShape() != other_.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }
    Tensor result(rows(), cols());
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = this -> data[i] - other_.data[i];
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
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = this -> data[i] * other_.data[i];
    }
    return result;
}

// Element-wise Tensor division
Tensor Tensor::operator/ (const Tensor& other_) const {
    if (getShape() != other_.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match");
    }
    Tensor result(rows(), cols());
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = this -> data[i] / other_.data[i];
    }
    return result;
}

// Tensor Ops with Scalars

// Scalar Addition
Tensor Tensor::operator+ (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> data.size(); i++) {
        result.data[i] = this -> data[i] + val_;
    }
    return result;
}

// Scalar Subtraction
Tensor Tensor::operator- (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> data.size(); i++) {
        result.data[i] = this -> data[i] - val_;
    }
    return result;
}

// Scalar Multiplication
Tensor Tensor::operator* (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> data.size(); i++) {
        result.data[i] = this -> data[i] * val_;
    }
    return result;
}

// Scalar Division
Tensor Tensor::operator/ (const double val_) const {
    assert(val_ != 0.0);
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> data.size(); i++) {
        result.data[i] = this -> data[i] / val_;
    }
    return result;
}

// * Compound Operations

// Element Wise Compound Addition
Tensor Tensor::operator+=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> data.size(); i++) {
        this -> data[i] += other_.data[i];
    }
    return *this;
}

Tensor Tensor::operator-=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> data.size(); i++) {
        this -> data[i] -= other_.data[i];
    }
    return *this;
}

Tensor Tensor::operator*=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> data.size(); i++) {
        this -> data[i] *= other_.data[i];
    }
    return *this;
}

Tensor Tensor::operator/=(const Tensor& other_) {
    if (this -> getShape() != other_.getShape()) {
        throw std::invalid_argument("The shape of the two tensors must match");
    }
    for (size_t i = 0; i < this -> data.size(); i++) {
        assert(other_.data[i] != 0.0);
        this -> data[i] /= other_.data[i];
    }
    return *this;
}


Tensor Tensor::operator+=(const double val_) {
    for (size_t i = 0; i < this -> data.size(); i++) {
        this -> data[i] += val_;
    }
    return *this;
}

Tensor Tensor::operator-=(const double val_) {
    for (size_t i = 0; i < this -> data.size(); i++) {
        this -> data[i] -= val_;
    }
    return *this;
}

Tensor Tensor::operator*=(const double val_) {
    for (size_t i = 0; i < this -> data.size(); i++) {
        this -> data[i] *= val_;
    }
    return *this;
}

Tensor Tensor::operator/=(const double val_) {
    assert(val_ != 0.0);
    for (size_t i = 0; i < this -> data.size(); i++) {
        this -> data[i] /= val_;
    }
    return *this;
}

// axon::f64 dot(const Tensor& a, const Tensor& b) {
//     if (!(a.getShape() == b.getShape() && a.rows() == b.rows() && a.cols() == b.cols())) {
//         throw std::invalid_argument("The shape of the two tensors must match");
//     }
//     axon::f64 dot_result = 0;
//     for (size_t i = 0; i < a.getData().size(); i++) {
//         dot_result += a.getData()[i] * b.getData()[i];
//     }
//     return dot_result;
// }


axon::f64 frobenius_inner_product(const Tensor& a, const Tensor& b) {
    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("The shape must match for Frobenius Inner Product");
    }

    axon::f64 result = 0.0;
    for (size_t i = 0; i < a.get_size(); i++) {
        result += a.getData()[i] * b.getData()[i];
    }

    return result;
}


axon::f64 dot(const Tensor& a, const Tensor& b) {
    if (a.get_size() != b.get_size()) {
        throw std::invalid_argument("The number of elements must be same for both Tensors to perform a dot product");
    }
    axon::f64 result = 0.0;

    for (size_t i = 0; i < a.get_size(); i++) {
        result += a.getData()[i] * b.getData()[i];
    }

    return result;
}


