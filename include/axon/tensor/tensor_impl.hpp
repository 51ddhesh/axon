#pragma once

#include "axon/core/storage.hpp"
#include <vector>
#include <numeric>
#include <stdexcept>

namespace axon::tensor {

class TensorImpl {
private:
    core::Storage storage_;
    size_t storage_offset_ = 0;
    std::vector<size_t> sizes_;
    std::vector<size_t> strides_;
    bool requires_grad_ = false;

public:
    TensorImpl() = default;

    // Create new contiguous storage
    TensorImpl(const std::vector<size_t>& sizes, core::Device device)
        : sizes_(sizes) {
        
        // Calculate strides for contiguous memory (C-order)
        strides_.resize(sizes_.size());
        size_t current_stride = 1;
        for (int i = static_cast<int>(sizes_.size()) - 1; i >= 0; --i) {
            strides_[i] = current_stride;
            current_stride *= sizes_[i];
        }

        size_t total_elements = current_stride;
        if (total_elements > 0) {
            storage_ = core::Storage(total_elements, device);
        } else {
            storage_ = core::Storage();
        }
    }

    // View into existing storage
    TensorImpl(core::Storage storage, size_t offset, 
               const std::vector<size_t>& sizes, 
               const std::vector<size_t>& strides)
        : storage_(std::move(storage)), storage_offset_(offset), 
          sizes_(sizes), strides_(strides) {}

    ~TensorImpl() = default;

    // Accessors
    const std::vector<size_t>& sizes() const { return sizes_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t storage_offset() const { return storage_offset_; }
    core::Device device() const { return storage_.device(); }
    const core::Storage& storage() const { return storage_; }
    core::Storage& storage() { return storage_; }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool req) { requires_grad_ = req; }

    size_t numel() const {
        if (sizes_.empty()) return 0;
        size_t n = 1;
        for (size_t s : sizes_) {
            n *= s;
        }
        return n;
    }

    size_t dim() const { return sizes_.size(); }

    bool is_contiguous() const {
        size_t expected_stride = 1;
        for (int i = static_cast<int>(sizes_.size()) - 1; i >= 0; --i) {
            if (sizes_[i] == 1) continue; // Broadcasted/unit dimensions can have any stride
            if (strides_[i] != expected_stride) return false;
            expected_stride *= sizes_[i];
        }
        return true;
    }

    float* data() { 
        return storage_.data() ? storage_.data() + storage_offset_ : nullptr; 
    }
    
    const float* data() const { 
        return storage_.data() ? storage_.data() + storage_offset_ : nullptr; 
    }
};

} // namespace axon::tensor
