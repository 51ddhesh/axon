#pragma once

#include "tensor_impl.hpp"
#include <memory>
#include <vector>

namespace axon::tensor {

class Tensor {
private:
    std::shared_ptr<TensorImpl> impl_;

public:
    // Null tensor
    Tensor() = default;

    // Create contiguous tensor
    Tensor(const std::vector<size_t>& shape, core::Device device = core::CPU()) {
        impl_ = std::make_shared<TensorImpl>(shape, device);
    }

    // Create view into existing storage
    Tensor(core::Storage storage, size_t offset, 
           const std::vector<size_t>& shape, 
           const std::vector<size_t>& strides) {
        impl_ = std::make_shared<TensorImpl>(std::move(storage), offset, shape, strides);
    }

    // Default copy and move semantics
    // Copying a Tensor shallow-copies the underlying TensorImpl pointer
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    ~Tensor() = default;

    // Check validity
    bool is_valid() const { return impl_ != nullptr; }

    // Accessors
    const std::vector<size_t>& shape() const { 
        if (!impl_) throw std::runtime_error("Cannot access shape of null tensor");
        return impl_->sizes(); 
    }
    
    const std::vector<size_t>& stride() const { 
        if (!impl_) throw std::runtime_error("Cannot access stride of null tensor");
        return impl_->strides(); 
    }

    size_t dim() const { return impl_ ? impl_->dim() : 0; }
    size_t numel() const { return impl_ ? impl_->numel() : 0; }
    core::Device device() const { return impl_ ? impl_->device() : core::CPU(); }
    bool is_contiguous() const { return impl_ ? impl_->is_contiguous() : false; }

    // Data access
    float* data() { return impl_ ? impl_->data() : nullptr; }
    const float* data() const { return impl_ ? impl_->data() : nullptr; }

    // Storage access (e.g. for creating views)
    core::Storage& storage() { 
        if (!impl_) throw std::runtime_error("Null tensor has no storage");
        return impl_->storage(); 
    }
    const core::Storage& storage() const { 
        if (!impl_) throw std::runtime_error("Null tensor has no storage");
        return impl_->storage(); 
    }
    size_t storage_offset() const { return impl_ ? impl_->storage_offset() : 0; }

    // Comparison for testing pointer sharing
    bool is_same(const Tensor& other) const {
        return impl_ == other.impl_;
    }
};

} // namespace axon::tensor
