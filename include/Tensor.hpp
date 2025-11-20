// include/Tensor.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <iostream>
#include <vector>
#include <initializer_list>
#include <assert.h>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_set>

namespace axon {
    namespace dtype {
        using i32 = int;
        using i64 = long long;
        using f32 = float;
        using f64 = double;
    }

    namespace constants {
        constexpr dtype::f64 eps = 1e-9;
        constexpr dtype::f64 pi = M_PI;
        constexpr dtype::f64 e = M_E;
    }
}

class Tensor {
private:
    std::vector<axon::dtype::f64> _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;

    // The parent Tensors used to create this tensor
    std::vector<Tensor*> _prev;
    // Compute the gradient for parent Tensors
    std::function<void(Tensor*)> _backward_fn;
    // shared_ptr to a Tensor that will hold the gradient
    // Handles memory management on its own
    std::shared_ptr<Tensor> _grad;

    void compute_strides();
    void build_topo(std::vector<Tensor*>& sorted, std::unordered_set<Tensor*>& visited);

    friend Tensor operator+ (const Tensor& a, const Tensor& b);    
    friend Tensor operator- (const Tensor& a, const Tensor& b);
    friend Tensor operator* (const Tensor& a, const Tensor& b);
    friend Tensor operator/ (const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

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

    // Operator to modify element with its index, directly from this -> _data
    double& operator() (size_t index_);

    // Operator to get a const, non-mutable reference with its index in this -> _data
    const double& operator() (size_t index_) const;
    
    
    // * Static factory methods similar to `numpy` and `torch`

    // Function similar to numpy.zeros 
    static Tensor zeros(size_t rows_, size_t cols_);

    // Function similar to numpy.ones
    static Tensor ones(size_t rows_, size_t cols_);

    // Get a random-valued (between 0 and 1 by default) Tensor
    static Tensor randn(size_t rows_, size_t cols_);
    static Tensor randn(size_t rows_, size_t cols_, std::vector<axon::dtype::f64> limits_);

    // Get a random-valued (between -1000 and 1000 by default) Tensor (integer-valued)
    static Tensor randint(size_t rows_, size_t cols_);
    static Tensor randint(size_t rows_, size_t cols_, std::vector<axon::dtype::i32> limits_);

    // Transpose
    Tensor T() const;

    // Create a column vector
    static Tensor column(std::initializer_list<axon::dtype::f64> init_list);

    // Create a row vector
    static Tensor row(std::initializer_list<axon::dtype::f64> init_list);


    // * IO

    void print_tensor() const;

    // * Utils

    // Return the sum of the elements of the `Tensor`
    axon::dtype::f64 sum() const;

    // Returns a column or a row as sum
    // use index = 0 for row-wise (collapse rows)
    // use index = 1 for column-wise (collapse columns)
    // 
    //    index = 1                     index = 0
    // 
    //  [[1, 2, 3],    [[6],       [[1, 2, 3], 
    //   [4, 5, 6], =>  [15],  or   [4, 5, 6],  => [[12, 15, 18]] 
    //   [7, 8, 9]]     [24]]       [7, 8, 9]]
    // 
    Tensor sum(int axis) const;


    // * Tensor Ops

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


    // autograd
    void backward();
    // Get the grad of the Tensor
    inline Tensor grad() const {
        return _grad ? *_grad : Tensor::zeros(rows(), cols());
    }

};

// Operators to handle scalar + tensor operations

Tensor operator+ (axon::dtype::f64 scalar, const Tensor& tensor);
Tensor operator- (axon::dtype::f64 scalar, const Tensor& tensor);
Tensor operator* (axon::dtype::f64 scalar, const Tensor& tensor);
Tensor operator/ (axon::dtype::f64 scalar, const Tensor& tensor);

// Operator to check if two tensors are equal
bool operator== (const Tensor& a, const Tensor& b);

void print(const Tensor& t);

template <typename _dtype>
inline void print(_dtype val) {
    if (std::is_same_v<_dtype, int> || std::is_same_v<_dtype, float> || std::is_same_v<_dtype, double>) {
        std::cout << val << std::endl;
    }
}

axon::dtype::f64 frobenius_inner_product(const Tensor& a, const Tensor& b);

axon::dtype::f64 dot(const Tensor& a, const Tensor& b); 

Tensor matmul(const Tensor& a, const Tensor& b);

#endif // AXON_TENSOR_HPP