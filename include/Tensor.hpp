// include/Tensor.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <iostream>
#include <vector>
#include <initializer_list>
#include <assert.h>

namespace axon {
    using i64 = long long;
    using f64 = double;
}

class Tensor {
private:
    std::vector<axon::f64> _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;

    void compute_strides();

public:

    // * Constructors

    // Default Constructor
    Tensor();
    // Init a Tensor with given rows and columns
    Tensor(size_t rows, size_t cols);
    // Init a Tensor with given rows and columns and values
    Tensor(size_t rows, size_t cols, double init_val);
    // Init a 1D Tensor with a `initializer_list`
    Tensor(std::initializer_list<double> init_list);
    // Init a 2D Tensor with a `initializer_list`
    Tensor(std::initializer_list<std::initializer_list<double>> init_list);


    // * Getters

    // Fetch data
    const std::vector<double>& getData() const {
        return this -> _data;
    }

    // Fetch shape
    const std::vector<size_t>& getShape() const {
        return this -> _shape;
    }

    // Fetch rows
    size_t rows() const {
        return _shape.empty() ? 0 : _shape[0];
    }

    // Fetch columns
    size_t cols() const {
        return _shape.size() < 2 ? 0 : _shape[1];
    }

    // Get size
    size_t get_size() const {
        return this -> _data.size();
    }

    // Get the strides
    const std::vector<size_t> get_strides() const {
        return this -> _strides;
    }



    // * Accessor Operator Overloads

    // Non-`const` accessor to modify elements
    double& operator() (size_t rows_, size_t cols_);

    // `const` accessor to perform read-only operations
    const double& operator() (size_t rows_, size_t cols_) const;

    // * Static factory methods similar to `numpy` and `torch`

    // Function similar to numpy.zeros 
    static Tensor zeros(size_t rows_, size_t cols_);

    // Function similar to numpy.ones
    static Tensor ones(size_t rows_, size_t cols_);

    // Get a random-valued (between 0 and 1 by default) Tensor
    static Tensor randn(size_t rows_, size_t cols_);
    static Tensor randn(size_t rows_, size_t cols_, double min_, double max_);

    // Get a random-valued (between -1000 and 1000 by default) Tensor (integer-valued)
    static Tensor randint(size_t rows_, size_t cols_);
    static Tensor randint(size_t rows_, size_t cols_, int min_, int max_);

    // Transpose
    static Tensor T(const Tensor& a);


    // * IO

    void print_tensor();

    // * Tensor Ops

    // Operations with other Tensors (Element wise)
    Tensor operator+(const Tensor& other) const;

    Tensor operator-(const Tensor& other) const;
    
    Tensor operator*(const Tensor& other) const;

    Tensor operator/(const Tensor& other) const;

    // Operations with scalars
    Tensor operator+(const double val_) const;
    
    Tensor operator-(const double val_) const;
    
    Tensor operator*(const double val_) const;

    Tensor operator/(const double val_) const;

    // Compound Operations 
    // Element wise
    Tensor operator+=(const Tensor& other);
    Tensor operator-=(const Tensor& other);
    Tensor operator*=(const Tensor& other);
    Tensor operator/=(const Tensor& other);
    // Ops witch scalars 
    Tensor operator+=(const double val_);
    Tensor operator-=(const double val_);
    Tensor operator*=(const double val_);
    Tensor operator/=(const double val_);

};


void print(const Tensor& t);

axon::f64 frobenius_inner_product(const Tensor& a, const Tensor& b);

axon::f64 dot(const Tensor& a, const Tensor& b); 

Tensor matmul(const Tensor& a, const Tensor& b);

#endif // AXON_TENSOR_HPP