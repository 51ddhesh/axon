#pragma once

#include "device.hpp"
#include "storage_body.hpp"
#include <memory>

namespace axon::core {

class Storage {
private:
    std::shared_ptr<StorageBody> body_;

public:
    Storage() = default;
    Storage(size_t size, Device device, Arena* arena = nullptr);
    explicit Storage(StorageBody* body);

    Storage(const Storage& other);
    Storage(Storage&& other) noexcept;
    Storage& operator=(const Storage& other);
    Storage& operator=(Storage&& other) noexcept;

    // Read-only access - no copy triggered
    const float* data() const { return body_ ? body_->data() : nullptr; }
    
    // Mutable access - triggers copy-on-write when shared
    float* data() { 
        if (body_ && !is_unique()) {
            body_.reset(body_->clone());
        }
        return body_ ? body_->data() : nullptr; 
    }

    size_t size() const { return body_ ? body_->size() : 0; }
    Device device() const { return body_ ? body_->device() : CPU(); }

    bool is_valid() const { return body_ != nullptr; }
    bool is_unique() const { return body_ && body_.use_count() == 1; }

    // Operations - auto copy-on-write only when shared
    void zero() {
        if (body_ && !is_unique()) {
            body_.reset(body_->clone());
        }
        if (body_) body_->zero();
    }

    void fill(float value) {
        if (body_ && !is_unique()) {
            body_.reset(body_->clone());
        }
        if (body_) body_->fill(value);
    }

    void resize(size_t new_size) {
        if (!body_ || new_size <= size()) return;
        
        if (is_unique()) {
            body_->resize(new_size);
        } else {
            auto new_body = std::make_shared<StorageBody>(new_size, device(), body_->arena());
            // StorageBody::allocate zeros memory, so just copy old data over
            if (device().is_cpu()) {
                std::memcpy(new_body->data(), body_->data(), size() * sizeof(float));
            } else {
#ifdef __CUDACC__
                cudaMemcpy(new_body->data(), body_->data(), size() * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
            }
            body_ = new_body;
        }
    }
};

inline Storage::Storage(size_t size, Device device, Arena* arena) {
    body_ = std::make_shared<StorageBody>(size, device, arena);
}

inline Storage::Storage(StorageBody* body) : body_(body) {}

inline Storage::Storage(const Storage& other) : body_(other.body_) {}

inline Storage::Storage(Storage&& other) noexcept : body_(std::move(other.body_)) {}

inline Storage& Storage::operator=(const Storage& other) {
    body_ = other.body_;
    return *this;
}

inline Storage& Storage::operator=(Storage&& other) noexcept {
    body_ = std::move(other.body_);
    return *this;
}

} // namespace axon::core