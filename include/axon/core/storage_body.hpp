#pragma once

#include "device.hpp"
#include "arena.hpp"
#include <cstddef>
#include <stdexcept>
#include <cstring>

namespace axon::core {

class StorageBody {
private:
    float* data_;
    size_t size_;
    Device device_;
    Arena* arena_;

public:
    StorageBody(size_t size, Device device, Arena* arena = nullptr);
    ~StorageBody();

    // Non-copyable
    StorageBody(const StorageBody&) = delete;
    StorageBody& operator=(const StorageBody&) = delete;

    // Access
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    Device device() const { return device_; }
    Arena* arena() const { return arena_; }

    // Clone - create a new independent copy
    StorageBody* clone() const {
        StorageBody* new_body = new StorageBody(size_, device_, arena_);
        if (device_.is_cpu()) {
            std::memcpy(new_body->data_, data_, size_ * sizeof(float));
        } else {
#ifdef __CUDACC__
            cudaMemcpy(new_body->data_, data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
        }
        return new_body;
    }

    // Operations
    void zero();
    void fill(float value);
    void resize(size_t new_size);

private:
    void allocate();
    void deallocate();
};

// Implementation
inline StorageBody::StorageBody(size_t size, Device device, Arena* arena)
    : size_(size), device_(device), arena_(arena) {
    allocate();
}

inline StorageBody::~StorageBody() {
    deallocate();
}

inline void StorageBody::allocate() {
    if (arena_) {
        data_ = static_cast<float*>(arena_->allocate(size_ * sizeof(float)));
        zero();
        return;
    }

    if (device_.is_cpu()) {
#ifdef __CUDACC__
        cudaMalloc(&data_, size_ * sizeof(float));
        cudaMemset(data_, 0, size_ * sizeof(float));
#else
        data_ = static_cast<float*>(std::aligned_alloc(256, size_ * sizeof(float)));
        if (!data_) throw std::bad_alloc();
        std::memset(data_, 0, size_ * sizeof(float));
#endif
    } else {
#ifdef __CUDACC__
        cudaMalloc(&data_, size_ * sizeof(float));
        cudaMemset(data_, 0, size_ * sizeof(float));
#else
        throw std::runtime_error("CUDA not available - rebuild with CUDA_ENABLED=ON");
#endif
    }
}

inline void StorageBody::deallocate() {
    if (data_ && !arena_) {
        if (device_.is_cpu()) {
            std::free(data_);
        } else {
#ifdef __CUDACC__
            cudaFree(data_);
#endif
        }
        data_ = nullptr;
    } else if (data_ && arena_) {
        data_ = nullptr; // Let arena manage it
    }
}

inline void StorageBody::zero() {
    if (device_.is_cpu()) {
        std::memset(data_, 0, size_ * sizeof(float));
    } else {
#ifdef __CUDACC__
        cudaMemset(data_, 0, size_ * sizeof(float));
#endif
    }
}

inline void StorageBody::fill(float value) {
    if (device_.is_cpu()) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    } else {
#ifdef __CUDACC__
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
#endif
    }
}

inline void StorageBody::resize(size_t new_size) {
    if (new_size <= size_) return;

    float* old_data = data_;
    size_t old_size = size_;
    
    size_ = new_size;
    allocate(); // allocates new_size into data_ and zeros it
    
    // Copy old data
    if (old_data) {
        if (device_.is_cpu()) {
            std::memcpy(data_, old_data, old_size * sizeof(float));
        } else {
#ifdef __CUDACC__
            cudaMemcpy(data_, old_data, old_size * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
        }
        
        // Temporarily swap back to old_data to deallocate it properly
        float* new_data_ptr = data_;
        data_ = old_data;
        deallocate();
        data_ = new_data_ptr;
    }
}

} // namespace axon::core