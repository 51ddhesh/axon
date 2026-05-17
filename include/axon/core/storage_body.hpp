#pragma once

#include "device.hpp"
#include <cstddef>
#include <atomic>
#include <stdexcept>
#include <cstring>

namespace axon::core {

class StorageBody {
private:
    float* data_;
    size_t size_;
    Device device_;
    std::atomic<int> ref_count_;

public:
    StorageBody(size_t size, Device device);
    ~StorageBody();

    // Non-copyable
    StorageBody(const StorageBody&) = delete;
    StorageBody& operator=(const StorageBody&) = delete;

    // Access
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    Device device() const { return device_; }

    // Reference counting
    void inc_ref() { ++ref_count_; }
    void dec_ref() { 
        if (--ref_count_ == 0) {
            delete this; 
        }
    }
    int ref_count() const { return ref_count_.load(); }

    // Clone - create a new independent copy
    StorageBody* clone() const {
        StorageBody* new_body = new StorageBody(size_, device_);
        std::memcpy(new_body->data_, data_, size_ * sizeof(float));
        return new_body;
    }

    // Operations
    void zero();
    void fill(float value);

private:
    void allocate();
    void deallocate();
};

// Implementation
inline StorageBody::StorageBody(size_t size, Device device)
    : size_(size), device_(device), ref_count_(1) {
    allocate();
}

inline StorageBody::~StorageBody() {
    deallocate();
}

inline void StorageBody::allocate() {
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
    if (data_) {
        if (device_.is_cpu()) {
            std::free(data_);
        } else {
#ifdef __CUDACC__
            cudaFree(data_);
#endif
        }
        data_ = nullptr;
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

} // namespace axon::core