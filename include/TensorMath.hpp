// include/TensorMath.hpp
// github.com/51ddhesh/axon
// MIT License


#ifndef AXON_TENSOR_MATH_HPP
#define AXON_TENSOR_MATH_HPP

#include "Tensor.hpp"

namespace axon {
    namespace math {
        // The below section of the functions is defined in src/math/comparisons.cpp

        // Returns a tensor with the same shape as `a` and `b`.
        // For each element in a and b, if a[i] > b[i], then c[i] = 1
        Tensor gt(const Tensor& a, const Tensor& b);
        
        // Returns a tensor with the same shape as `a` and `b`.
        // For each element in a and b, if a[i] < b[i], then c[i] = 1
        Tensor lt(const Tensor& a, const Tensor& b);
        
        // Returns a tensor with the same shape as `a` and `b`.
        // For each element in a and b, if a[i] == b[i], then c[i] = 1
        // Since floating point comparison is slightly inaccurate
        // The difference between a[i] and b[i] must be <= 1e-9
        Tensor eq(const Tensor& a, const Tensor& b); 
    }
}

#endif // AXON_TENSOR_MATH_HPP