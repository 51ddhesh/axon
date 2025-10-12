// src/tensor_utils.cpp
// github.com/51ddhesh
// MIT License

#include "../include/Tensor.hpp"
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
            std::cout << std::fixed << std::setprecision(2) << (*this)(i, j);
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

