// src/Tensor.cpp
// github.com/51ddhesh/axon
// MIT License

#include "axon/Tensor.hpp"
#include <numeric>
#include <iomanip>
#include <stdexcept>

namespace axon {

// TensorStorage implementation
TensorStorage::TensorStorage(size_t size) {
    data.resize(size);
    // init the gradients to zero
    grad.resize(size, 0.0); 
}

// * Tensor Stride implementation

// Row Major strides
void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    size_t stride = 1;
    size_t shape_size = shape_.size();
    for (int i = shape_size - 1; i >= 0; i--) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

// * CONSTRUCTORS

// Default
Tensor::Tensor() : offset_(0) {}

// Shape + Fill value
Tensor::Tensor(const std::vector<size_t>& shape, double val) : 
    shape_(shape), offset_(0) {
        compute_strides();
        size_t total_size = size();

        // allocate memory on heap
        storage_ = std::make_shared<TensorStorage>(total_size);
        std::fill(storage_ -> data.begin(), storage_ -> data.end(), val);
}

// Constructor with 1D initializer list
Tensor::Tensor(std::initializer_list<double> list) :
    shape_({list.size()}), offset_(0) {
        compute_strides();
        storage_ = std::make_shared<TensorStorage>(list.size());
        // copy the data
        std::copy(list.begin(), list.end(), storage_ -> data.begin());
}

// Private constructor for the views
Tensor::Tensor(std::shared_ptr<TensorStorage> storage, 
               const std::vector<size_t>& shape,
               const std::vector<size_t>& strides,
               size_t offset,
               const std::vector<Tensor>& prev
) : storage_(storage), shape_(shape), strides_(strides), offset_(offset), prev_(prev) {}

// * Utils
size_t Tensor::size() const {
    if (shape_.empty()) return 0;
    size_t s = 1;
    for (auto dim : shape_) s *= dim;
    return s;
}

void Tensor::print_meta() const {
    std::cout << "Tensor<";
    if (!storage_) {
        std::cout << "Empty>" << std::endl;
        return;
    }

    std::cout << "Shape:(";
    for (size_t s : shape_) std::cout << s << ",";
    std::cout << ") ";
    
    std::cout << "Stride:(";
    for (size_t s : strides_) std::cout << s << ",";
    std::cout << ") ";

    std::cout << "Offset:" << offset_ << " ";
    std::cout << "StoragePtr:" << storage_.get();
    std::cout << ">" << std::endl;
}

// ! Assumes is_contiguous
// TODO: Check if is_contiguous
void Tensor::print() const {
    if (!storage_) return;
    const double* p = data_ptr();
    size_t total_size = size();
    std::cout << "Tensor: [";
    for (size_t i = 0; i < total_size; i++) {
        std::cout << p[i] << ',';
    }
    std::cout << ']' << std::endl;
}

// * ACCESSORS
double* Tensor::data_ptr() {
    if (!storage_) return nullptr;
    return storage_ -> data.data() + offset_;
}

const double* Tensor::data_ptr() const {
    if (!storage_) return nullptr;
    return storage_ -> data.data() + offset_;
}

double* Tensor::grad_ptr() {
    if (!storage_) return nullptr;
    return storage_ -> grad.data() + offset_;
}

// * FACTORY METHODS
Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    return Tensor(shape, 0.0);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    return Tensor(shape, 1.0);
}

// * N-DIM Access
double& Tensor::operator() (const std::vector<size_t>& coords) {
    if (coords.size() != shape_.size()) {
        throw std::runtime_error("Ranks mismatch...");
    }

    size_t idx = offset_;
    size_t coords_size = coords.size();
    for (size_t i = 0; i < coords_size; i++) {
        idx += coords[i] * strides_[i];
    }
    return storage_ -> data[idx];
}

double Tensor::operator() (const std::vector<size_t>& coords) const {
    if (coords.size() != shape_.size()) {
        throw std::runtime_error("Ranks mismatch...");
    }

    size_t idx = offset_;
    size_t coords_size = coords.size();
    for (size_t i = 0; i < coords_size; i++) {
        idx += coords[i] * strides_[i];
    }
    return storage_ -> data[idx];
}

// * VIEWS
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t current_size = size();
    size_t new_size = 1;
    for (size_t s : new_shape) new_size *= s;

    if (current_size != new_size) throw std::runtime_error("Reshape size mismatch");

    // TODO: check is_contiguous
    Tensor t(storage_, new_shape, {}, 0, prev_);
    t.compute_strides();
    return t;
}

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    if (dims.size() != shape_.size()) {
        throw std::runtime_error("Permute rank mismatch");
    }

    size_t shape_size = shape_.size();
    size_t dim_size = dims.size();

    std::vector<size_t> new_shape(shape_size);
    std::vector<size_t> new_strides(shape_size);

    for (size_t i = 0; i < dim_size; i++) {
        new_shape[i] = shape_[dims[i]];
        new_strides[i] = strides_[dims[i]];
    }

    return Tensor(storage_, new_shape, new_strides, offset_, prev_);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    std::vector<size_t> dims(shape_.size());
    std::iota(dims.begin(), dims.end(), 0);
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

} // namespace axon
