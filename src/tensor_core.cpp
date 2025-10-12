// src/tensor_core.cpp
// github.com/51ddhesh
// MIT License

#include "../include/Tensor.hpp"
#include <stdexcept>
#include <numeric>

// Default Constructor
Tensor::Tensor() : data(), shape({0, 0}) {}

Tensor::Tensor(size_t rows_, size_t cols_) : shape({rows_, cols_}) {
    data.resize(rows_ * cols_, 0.0);
}

Tensor::Tensor(size_t rows_, size_t cols_, double init_val) : shape({rows_, cols_}) {
    data.resize(rows_ * cols_, init_val);
}

Tensor::Tensor(std::initializer_list<double> init_list) {
    shape = {1, init_list.size()};
    data = init_list;
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> init_list) {
    if (init_list.size() == 0 || init_list.begin() -> size() == 0) {
        shape = {0, 0};
        return;
    }

    size_t rows_ = init_list.size();
    size_t cols_ = init_list.begin() -> size();

    shape = {rows_, cols_};
    data.reserve(rows_ * cols_);

    for (const auto& row_list : init_list) {
        if (row_list.size() != cols_) {
            throw std::invalid_argument("All rows in init_list must have same size.");
        }
        data.insert(data.end(), row_list.begin(), row_list.end());
    }
}

double& Tensor::operator() (size_t row_, size_t col_) {
    if (row_ >= rows() || col_ >= cols()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data[row_ * cols() + col_];
}

const double& Tensor::operator() (size_t row_, size_t col_) const {
    if (row_ >= rows() || col_ >= cols()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data[row_ * cols() + col_];
}

