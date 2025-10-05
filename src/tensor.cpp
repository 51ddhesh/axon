// src/tensor.cpp
// github.com/51ddhesh
// MIT License

#include "../include/tensor.hpp"


// ! TO REMOVE:
// ! Causes errors with undefined references due to the templates
// template <typename T, typename U>
// axon::i64 dot(const Tensor<double>& a, const Tensor<U>& b) {
//     if (a.getData().size() != b.getData().size()) {
//         throw std::range_error("The size of both Tensors must be the same");
//     }
//     axon::i64 dot = 0;
//     size_t n = a.getData().size();
    
//     for (size_t i = 0; i < n; i++) {
//         dot += int(double(a.data[i]) * double(b.data[i]));
//     }
    
//     return dot;
// }
