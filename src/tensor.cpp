// src/tensor.cpp
// github.com/51ddhesh
// MIT License

#include "../include/tensor.hpp"

void Tensor::print_tensor() {
    auto data = getData();
    std::cout << "Tensor([";
    for (auto& i : data) std::cout << i << ", ";
    std::cout << "])" << std::endl;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (getSize() != other.getSize()) {
        throw std::range_error("Size of both Tensors must be same");
    }
    // element-wise addition
    std::vector<double> t(getSize(), 0.0);
    for (size_t i = 0; i < getSize(); i++) {
        t[i] = getData()[i] + other.getData()[i];
    }
    return Tensor(t);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (getSize() != other.getSize()) {
        throw std::range_error("Size of both Tensors must be same");
    }
    std::vector<double> t(getSize(), 0.0);
    for (size_t i = 0; i < getSize(); i++) {
        t[i] = getData()[i] - other.getData()[i];
    }
    return Tensor(t);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (getSize() != other.getSize()) {
        throw std::range_error("Size of both Tensors must be same");
    }
    std::vector<double> t(getSize(), 0.0);
    for (size_t i = 0; i < getSize(); i++) {
        t[i] = getData()[i] * other.getData()[i];
    }
    return Tensor(t);
}

void print(const Tensor& t) {
    auto data = t.getData();
    std::cout << "Tensor([";
    for (auto& i : data) std::cout << i << ", ";
    std::cout << "])" << std::endl;
}

axon::f64 dot(const Tensor& a, const Tensor& b) {
    if (a.getSize() != b.getSize()) {
        throw std::range_error("Size of both Tensors must be same");
    }

    axon::f64 dot_prod = 0.0;

    for (size_t i = 0; i < a.getSize(); i++) {
        dot_prod += a.getData()[i] * b.getData()[i];
    }

    return dot_prod;
}

Tensor Tensor::operator+(const double val_) const {
    std::vector<double> temp_(getSize(), 0.0);
    for (size_t i = 0; i < getSize(); i++) {
        temp_[i] = getData()[i] + val_;
    }
    return Tensor(temp_);
}

Tensor Tensor::operator-(const double val_) const {
    std::vector<double> temp_ (getSize(), 0.0);
    for (size_t i = 0; i < getSize(); i++) {
        temp_[i] = getData()[i] - val_;
    }
    return Tensor(temp_);
}

Tensor Tensor::operator*(const double val_) const {
    std::vector<double> temp_(getSize(), 0.0);
    for (size_t i = 0; i < getSize(); i++) {
        temp_[i] = getData()[i] * val_;
    }
    return Tensor(temp_);
}

