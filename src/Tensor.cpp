// src/Tensor.cpp
// github.com/51ddhesh/axon
// MIT License

#include "axon/Tensor.hpp"
#include <numeric>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>

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

void Tensor::print() const {
    size_t total_elements = size();
    
    if (is_contiguous()) {
        const double* ptr = data_ptr();
        std::cout << "Tensor (Contiguous): [ ";
        for(size_t i=0; i<total_elements; ++i) {
            std::cout << ptr[i] << " ";
        }
        std::cout << "]" << std::endl;
        return;
    }

    // Slow path for Strided/Views
    std::cout << "Tensor (Strided): [ ";
    
    std::vector<size_t> coords(shape_.size(), 0);
    for (size_t i = 0; i < total_elements; ++i) {
        std::cout << (*this)(coords) << " "; 

        for (int dim = shape_.size() - 1; dim >= 0; --dim) {
            coords[dim]++;
            if (coords[dim] < shape_[dim]) break;
            coords[dim] = 0;
        }
    }
    std::cout << "]" << std::endl;
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

    if (!is_contiguous()) {
        Tensor c = this -> contiguous();
        return c.reshape(new_shape);
    }

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

// * MEMORY LAYOUT
bool Tensor::is_contiguous() const {
    size_t z = 1;
    size_t shape_size = shape_.size();
    for (int i = shape_size - 1; i >= 0; i--) {
        if (strides_[i] != z) return false;
        z *= shape_[i];
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor compact = Tensor(shape_);

    size_t total_elements = size();
    std::vector<size_t> coords(shape_.size(), 0);

    double* target_ptr = compact.data_ptr();
    
    for (size_t i = 0; i < total_elements; i++) {
        double val = (*this)(coords);
        target_ptr[i] = val;

        for (int dim = shape_.size() - 1; dim >= 0; dim--) {
            coords[dim]++;
            if (coords[dim] < shape_[dim]) {
                break;
            } else {
                coords[dim] = 0;
            }
        }
    }
    return compact;
}

void Tensor::zero_grad() {
    if (storage_) {
        std::fill(storage_ -> grad.begin(), storage_ -> grad.end(), 0.0);
    }
}

void Tensor::backward() {
    std::vector<Tensor> order;
    std::unordered_set<TensorStorage*> visited;

    std::function<void(const Tensor&)> build_order = 
        [&](const Tensor& node) {
            if (!node.storage_) {
                return;
            }

            if (visited.count(node.storage_.get())) {
                return;
            }

            // perform DFS
            visited.insert(node.storage_.get());
            for (const auto& parent : node.prev_) {
                build_order(parent);
            }
            order.push_back(node);
        };

    build_order(*this);
    std::reverse(order.begin(), order.end());

    // seed the gradient at the start (dL/dLoss = 1.0)
    // assuming it is a scalar loss
    if (this -> size() != 1) {
        std::cerr << "Warning: Calling backward() on non-scalar Tensor. Seeding with 1.0" << std::endl;
    }

    std::fill(storage_ -> grad.begin(), storage_ -> grad.end(), 1.0);

    // Execute the chain rule
    for (Tensor& node : order) {
        if (node.grad_fn_) {
            node.grad_fn_();
        }
    }
}

} // namespace axon
