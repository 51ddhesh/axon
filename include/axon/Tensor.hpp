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
           const std::vector<size_t>& stides,
           size_t offset,
           const std::vector<Tensor>& prev = {}
    );

public:
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

    // * FACTORY FUNCTIONS
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);

    // * UTILS
    void print() const;
    void print_meta() const;
};

} // namespace axon

#endif // AXON_TENSOR_HPP
