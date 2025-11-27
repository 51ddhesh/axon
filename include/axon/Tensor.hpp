// include/axon/Tensor.hpp
// github.com/51ddhesh/axon
// MIT License

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <memory>
#include <vector>
#include <initializer_list>
#include <functional>
#include <iostream>

namespace axon {


class Tensor; // forward declaration


// Tensor Storage
struct TensorStorage {
    std::vector<double> data;
    std::vector<double> grad;

    explicit TensorStorage(size_t size);
    ~TensorStorage() = default;
};


// The Tensor

class Tensor {

private:
    // pointer to the shared memory blob
    std::shared_ptr<TensorStorage> storage_;
    
    // Metadata for the view of the Tensor
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;

    // Graph metadata (the nodes)
    std::vector<Tensor> prev_;

    // helpers
    void compute_strides();

    // private constructor 
    Tensor(std::shared_ptr<TensorStorage> storage, 
           const std::vector<size_t>& shape,
           const std::vector<size_t>& strides,
           size_t offset,
           const std::vector<Tensor>& prev = {}
    );

    using BackwardFn = std::function<void()>;
    BackwardFn grad_fn_;

public:

    // * GRAD
    void set_backward_fn(BackwardFn fn) {
        grad_fn_ = fn;
    }

    void backward();
    void zero_grad();

    // * CONSTRUCTORS
    // Empty / default
    Tensor();
    // Create a new tensor and storage of specific shape, initialized to val
    explicit Tensor(const std::vector<size_t>& shape, double val = 0.0);
    // Create a 1D tensor from initializer list
    Tensor(std::initializer_list<double> list);

    // * ACCESSORS
    const std::vector<size_t>& shape() const {
        return shape_;
    }

    const std::vector<size_t>& strides() const {
        return strides_;
    }

    size_t size() const;

    size_t offset() const {
        return offset_;
    }

    // * POINTERS
    double* data_ptr();
    const double* data_ptr() const;
    double* grad_ptr();

    // * MEMORY LAYOUT
    bool is_contiguous() const;
    Tensor contiguous() const;

    // * N-DIM 
    // Get reference
    double& operator() (const std::vector<size_t>& coords);
    double operator() (const std::vector<size_t>& coords) const;

    // Views (Zero Copy)
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor permute(const std::vector<size_t>& dims) const;
    Tensor transpose(size_t dim0, size_t dim1) const;


    // * FACTORY FUNCTIONS
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);

    // * UTILS
    void print() const;
    void print_meta() const;

    // * REDUCTIONS
    // Returns the sum of all the elements in the Tensor
    Tensor sum() const;

    // Returns the sum along a specific dimension
    Tensor sum(size_t dim) const;

    // * MATH OPERATORS
    Tensor operator+ (const Tensor& other) const;
    Tensor operator* (const Tensor& other) const;
    Tensor operator- (const Tensor& other) const;
    Tensor operator- () const;


    // * OPERATIONS 

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const;

    // * ACTIVATIONS
    Tensor relu() const;
    Tensor sigmoid() const;
};

} // namespace axon

#endif // AXON_TENSOR_HPP
