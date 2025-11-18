// include/TensorMath.hpp
// github.com/51ddhesh/axon
// MIT License


#ifndef AXON_TENSOR_MATH_HPP
#define AXON_TENSOR_MATH_HPP

#include "Tensor.hpp"
#include <cmath>

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

        // The below section is defined in src/math/unary_ops.cpp
        
        // Calculates sine of all the elements of `input` 
        // Returns a tensor with same shape as `input`
        Tensor sin(const Tensor& input);
    
        // Calculates the cosine of all the elements of `input`
        // Returns a `Tensor` with the same shape as `input`
        Tensor cos(const Tensor& input);
        
        // Calculates the tangent of all the elements of `input`
        // Returns a `Tensor` with the same shape as `input`
        Tensor tan(const Tensor& input);
 
        // Calculates the square root of all the elements of `input` 
        // Returns a `Tensor` with the same shape as `input`
        Tensor sqrt(const Tensor& input);
        
        // Calculates the exponential of all the elements of `input`
        // Returns a `Tensor` with the same shape as `input`
        Tensor exp(const Tensor& input);
        
        // Calculates the natural logarithm of all the elements of `input`
        // Returns a `Tensor` with the same shape as `input`
        Tensor ln(const Tensor& input);
        
        // Calculates the logarithm (base 10) of all the elements of `input`
        // Returns a `Tensor` with the same shape as `input`
        Tensor log10(const Tensor& input);

        // Calculates the logarithm (base 2) of all the elements of `input`
        // Returns a `Tensor` with the same shape as `input`
        Tensor log2(const Tensor& input);
    
        // Raises all elements of `input` (base) to the given `exponent`
        // Returns a `Tensor` with the same shape as `input`
        Tensor pow(const Tensor& base, axon::dtype::f64 exponent);
        
        // `pow(base, i) : for i in exponent` 
        // Returns a `Tensor` with the same shape as `exponent`
        Tensor pow(axon::dtype::f64 base, const Tensor& exponent);
        
        // Raises the corresponding element of the `base` to the corresponding element in the `power`
        Tensor pow(const Tensor& base, const Tensor& power);   
    
        // returns the maximum element from each `Tensor` in a `Tensor`
        Tensor maximum(const Tensor& a, const Tensor& b);

        // returns the minimum element from each input tensor in a `Tensor`
        Tensor minimum(const Tensor& a, const Tensor& b);
    }
}

#endif // AXON_TENSOR_MATH_HPP