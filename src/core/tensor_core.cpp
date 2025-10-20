// src/tensor_core.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Tensor.hpp"
#include <stdexcept>
#include <numeric>

// Default Constructor
Tensor::Tensor() : _data(), _shape({0, 0}) {
    compute_strides();
}

Tensor::Tensor(size_t rows_, size_t cols_) : _shape({rows_, cols_}) {
    _data.resize(rows_ * cols_, 0.0);
    compute_strides();
}

Tensor::Tensor(size_t rows_, size_t cols_, double init_val) : _shape({rows_, cols_}) {
    _data.resize(rows_ * cols_, init_val);
    compute_strides();
}

Tensor::Tensor(std::initializer_list<double> init_list) {
    _shape = {1, init_list.size()};
    _data = init_list;
    compute_strides();
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> init_list) {
    if (init_list.size() == 0 || init_list.begin() -> size() == 0) {
        _shape = {0, 0};
        return;
    }

    size_t rows_ = init_list.size();
    size_t cols_ = init_list.begin() -> size();

    _shape = {rows_, cols_};
    _data.reserve(rows_ * cols_);

    for (const auto& row_list : init_list) {
        if (row_list.size() != cols_) {
            throw std::invalid_argument("All rows in init_list must have same size.");
        }
        _data.insert(_data.end(), row_list.begin(), row_list.end());
    }

    compute_strides();
}

void Tensor::compute_strides() {
    if (_shape.empty()) {
        _strides.clear();
        return;
    }
    _strides.resize(_shape.size());
    _strides.back() = 1;
    for (int i = _shape.size() - 2; i >= 0; i--) {
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
}

double& Tensor::operator() (size_t row_, size_t col_) {
    if (row_ >= rows() || col_ >= cols()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return _data[row_ * _strides[0] + col_ * _strides[1]];
}

const double& Tensor::operator() (size_t row_, size_t col_) const {
    if (row_ >= rows() || col_ >= cols()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return _data[row_ * _strides[0] + col_ * _strides[1]];
}

double& Tensor::operator() (size_t index_) {
    if (index_ >= this -> get_size()) {
        throw std::out_of_range("Index out of range");
    }
    return _data[index_];
}

const double& Tensor::operator() (size_t index_) const {
    if (index_ >= this -> get_size()) {
        throw std::out_of_range("Index out of range");
    }
    return _data[index_];
}

