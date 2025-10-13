// src/tensor_utils.cpp
// github.com/51ddhesh
// MIT License

#include "../include/Tensor.hpp"
#include "../utils/random_.hpp"
#include <iomanip>

void Tensor::print_tensor() {
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
    const_cast<Tensor&>(t).print_tensor();
}

Tensor Tensor::zeros(size_t rows_, size_t cols_) {
    return Tensor(rows_, cols_);
}

Tensor Tensor::ones(size_t rows_, size_t cols_) {
    return Tensor(rows_, cols_, 1.0);
}

Tensor Tensor::randn(size_t rows_, size_t cols_) {
    double min_ = 0.0;
    double max_ = 1.0;
    Tensor random_tensor_(rows_, cols_);
    for (size_t i = 0; i < random_tensor_.data.size(); i++) {
        random_tensor_.data[i] = axon_random::random_double(min_, max_);
    }
    return random_tensor_;
}

Tensor Tensor::randn(size_t rows_, size_t cols_, double min_ = 0.0, double max_ = 1.0) {
    Tensor random_tensor_(rows_, cols_);
    for (size_t i = 0; i < random_tensor_.data.size(); i++) {
        random_tensor_.data[i] = axon_random::random_double(min_, max_);
    }
    return random_tensor_;
}

Tensor Tensor::randint(size_t rows_, size_t cols_) {
    int min_ = -1000;
    int max_ = 1000;
    Tensor random_int_tensor(rows_, cols_);
    for (size_t i = 0; i < random_int_tensor.data.size(); i++) {
        random_int_tensor.data[i] = axon_random::random_double(min_, max_);
    }
    return random_int_tensor;
}

Tensor Tensor::randint(size_t rows_, size_t cols_, int min_ = -1000, int max_ = 1000) {
    Tensor random_int_tensor(rows_, cols_);
    for (size_t i = 0; i < random_int_tensor.data.size(); i++) {
        random_int_tensor.data[i] = axon_random::random_double(min_, max_);
    }
    return random_int_tensor;
}
