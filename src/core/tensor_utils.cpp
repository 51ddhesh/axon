// src/tensor_utils.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Tensor.hpp"
#include "../../utils/random_.hpp"
#include <iomanip>

void Tensor::print_tensor() const {
    std::cout << "Tensor([";
    for (size_t i = 0; i < rows(); i++) {
        if (i > 0) {
            std::cout << ' ';
        }
        if (i != 0) std::cout << "       ";
        std::cout << '[';
        for (size_t j = 0; j < cols(); j++) {
            std::cout << std::fixed << std::setprecision(4) << (*this)(i, j);
            if (j < cols() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ']';
        if (i < rows() - 1) {
            std::cout << ",\n";
        }
    }
    std::cout << "])" << std::endl;
}

// Wrapper over `Tensor::print_tensor
void print(const Tensor& t) {
    t.print_tensor();
}

Tensor Tensor::zeros(size_t rows_, size_t cols_) {
    return Tensor(rows_, cols_);
}

Tensor Tensor::ones(size_t rows_, size_t cols_) {
    return Tensor(rows_, cols_, 1.0);
}

Tensor Tensor::randint(size_t rows_, size_t cols_) {
    int min_ = -1000;
    int max_ = 1000;
    Tensor random_int_tensor(rows_, cols_);
    for (size_t i = 0; i < random_int_tensor._data.size(); i++) {
        random_int_tensor(i) = axon_random::random_int(min_, max_);
    }
    return random_int_tensor;
}

Tensor Tensor::randint(size_t rows_, size_t cols_, std::vector<axon_dtype::i32> limits_) {
    assert(limits_.size() == 2);
    Tensor random_int_tensor(rows_, cols_);
    for (size_t i = 0; i < random_int_tensor._data.size(); i++) {
        random_int_tensor(i) = axon_random::random_int(limits_[0], limits_[1]);
    }
    return random_int_tensor;
}

Tensor Tensor::randn(size_t rows_, size_t cols_, std::vector<axon_dtype::f64> limits_) {
    assert(limits_.size() == 2);
    Tensor random_tensor(rows_, cols_);
    for (size_t i = 0; i < random_tensor.get_size(); i++) {
        random_tensor(i) = axon_random::random_double(limits_[0], limits_[1]);
    }
    return random_tensor;
}

Tensor Tensor::randn(size_t rows_, size_t cols_) {
    axon_dtype::f64 min_ = 0.0, max_ = 1.0;
    Tensor random_tensor(rows_, cols_);
    for (size_t i = 0; i < random_tensor.get_size(); i++) {
        random_tensor(i) = axon_random::random_double(min_, max_);
    }
    return random_tensor;
}

Tensor Tensor::row(std::initializer_list<axon_dtype::f64> init_list) {
    return Tensor(init_list);
}

Tensor Tensor::column(std::initializer_list<axon_dtype::f64> init_list) {
    Tensor result(init_list.size(), 1);
    size_t i = 0;
    for (const axon_dtype::f64& val : init_list) {
        result(i, 0) = val;
        i++;
    }
    return result;
}

